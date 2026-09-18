"""B5 hybrid multimodal GNN model.

This model is intentionally heavier than the B4 GNN baseline. It combines:

- a deeper pure-PyTorch molecular graph encoder
- attention graph pooling
- Morgan fingerprints as an explicit chemical prior
- PCA-reduced cell-line expression features
- multiplicative fusion terms between drug and cell representations

B5 is still dependency-light: it uses PyTorch, RDKit via the existing
fingerprint/graph builders, NumPy, pandas, and scikit-learn.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - import-time dependency guard.
    raise RuntimeError(
        "B5 hybrid GNN requires PyTorch. Install the gpu extra or torch directly."
    ) from exc

from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.features.graphs import ATOM_FEATURE_DIM, MolecularGraph, build_graph_map
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b3 import resolve_torch_device
from mcdrp.models.gnn_b4 import fit_cell_scaler, fit_target_scaler, split_rows
from mcdrp.results.predictions import prediction_frame, write_predictions
from mcdrp.splits.make_splits import DEFAULT_SPLITS

logger = logging.getLogger(__name__)

EXPRESSION_PATH = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"


@dataclass
class HybridGraphBatch:
    """Batched graph, fingerprint, cell, and target tensors."""

    node_features: torch.Tensor
    adjacency: torch.Tensor
    mask: torch.Tensor
    fingerprints: torch.Tensor
    cell_features: torch.Tensor
    targets: torch.Tensor


class HybridDrugResponseDataset(Dataset[dict[str, Any]]):
    """Dataset returning graph + fingerprint + cell features for each response."""

    def __init__(
        self,
        rows: pd.DataFrame,
        graph_map: dict[str, MolecularGraph],
        fingerprint_map: dict[str, np.ndarray],
        cell_feature_map: dict[str, np.ndarray],
        *,
        cell_scaler: StandardScaler,
        target_scaler: StandardScaler,
        target: str,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.graphs: list[MolecularGraph] = []
        self.fingerprints: list[np.ndarray] = []
        self.cell_features: list[np.ndarray] = []
        self.targets: list[np.float32] = []

        n_cell_features = cell_scaler.n_features_in_
        n_fingerprint_bits = len(next(iter(fingerprint_map.values())))
        for row in self.rows.itertuples(index=False):
            drug_id = str(row.drug_id)
            self.graphs.append(graph_map[drug_id])
            self.fingerprints.append(
                fingerprint_map.get(
                    drug_id,
                    np.zeros(n_fingerprint_bits, dtype=np.float32),
                ).astype(np.float32)
            )
            cell = cell_feature_map.get(str(row.depmap_id))
            if cell is None:
                cell = np.zeros(n_cell_features, dtype=np.float32)
            cell_scaled = cell_scaler.transform(cell.reshape(1, -1))[0].astype(np.float32)
            self.cell_features.append(cell_scaled)
            y_scaled = target_scaler.transform([[getattr(row, target)]])[0, 0]
            self.targets.append(np.float32(y_scaled))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        graph = self.graphs[idx]
        return {
            "node_features": graph.node_features,
            "adjacency": graph.adjacency,
            "fingerprints": self.fingerprints[idx],
            "cell_features": self.cell_features[idx],
            "target": self.targets[idx],
        }


def collate_hybrid_graph_batch(samples: list[dict[str, Any]]) -> HybridGraphBatch:
    """Pad variable-size molecular graphs and stack tabular features."""

    batch_size = len(samples)
    max_nodes = max(sample["node_features"].shape[0] for sample in samples)
    node_features = np.zeros((batch_size, max_nodes, ATOM_FEATURE_DIM), dtype=np.float32)
    adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros((batch_size, max_nodes), dtype=np.float32)
    fingerprints = np.stack([sample["fingerprints"] for sample in samples]).astype(
        np.float32
    )
    cell_features = np.stack([sample["cell_features"] for sample in samples])
    targets = np.asarray([sample["target"] for sample in samples], dtype=np.float32)

    for idx, sample in enumerate(samples):
        n_nodes = sample["node_features"].shape[0]
        node_features[idx, :n_nodes] = sample["node_features"]
        adjacency[idx, :n_nodes, :n_nodes] = sample["adjacency"]
        mask[idx, :n_nodes] = 1.0

    return HybridGraphBatch(
        node_features=torch.from_numpy(node_features),
        adjacency=torch.from_numpy(adjacency),
        mask=torch.from_numpy(mask),
        fingerprints=torch.from_numpy(fingerprints),
        cell_features=torch.from_numpy(cell_features),
        targets=torch.from_numpy(targets),
    )


class ResidualGraphBlock(nn.Module):
    """Residual mean-aggregation graph block."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim)
        self.residual = (
            nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.bmm(adjacency, x)
        message = self.self_linear(x) + self.neighbor_linear(neighbors)
        return self.dropout(self.activation(self.norm(message + self.residual(x))))


class AttentiveGraphEncoder(nn.Module):
    """Encode atom graphs with residual GNN layers and attention pooling."""

    def __init__(
        self,
        *,
        atom_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 256,
        n_layers: int = 5,
        output_dim: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        layers: list[ResidualGraphBlock] = []
        input_dim = atom_dim
        for _ in range(n_layers):
            layers.append(ResidualGraphBlock(input_dim, hidden_dim, dropout))
            input_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = node_features
        for layer in self.layers:
            x = layer(x, adjacency)
            x = x * mask.unsqueeze(-1)

        masked_x = x * mask.unsqueeze(-1)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = masked_x.sum(dim=1) / counts
        max_pool = masked_x.masked_fill(mask.unsqueeze(-1).eq(0), -1e9).max(dim=1).values

        attn_logits = self.attention(x).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask.eq(0), -1e9)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        attention_pool = (x * attn).sum(dim=1)

        return self.output(torch.cat([mean_pool, max_pool, attention_pool], dim=1))


class MLPEncoder(nn.Module):
    """Layer-normalized feed-forward encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden_dim
        layers.extend(
            [
                nn.Linear(previous, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class HybridGNNRegressor(nn.Module):
    """Graph + fingerprint + cell-line multimodal regressor."""

    def __init__(
        self,
        *,
        cell_dim: int,
        fingerprint_dim: int,
        graph_hidden_dim: int = 256,
        graph_layers: int = 5,
        graph_output_dim: int = 256,
        fingerprint_hidden_dim: int = 512,
        fingerprint_output_dim: int = 256,
        cell_hidden_dim: int = 256,
        shared_dim: int = 256,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.graph_encoder = AttentiveGraphEncoder(
            hidden_dim=graph_hidden_dim,
            n_layers=graph_layers,
            output_dim=graph_output_dim,
            dropout=dropout,
        )
        self.fingerprint_encoder = MLPEncoder(
            fingerprint_dim,
            (fingerprint_hidden_dim,),
            fingerprint_output_dim,
            dropout,
        )
        self.cell_encoder = MLPEncoder(
            cell_dim,
            (cell_hidden_dim,),
            shared_dim,
            dropout,
        )
        self.drug_projection = nn.Sequential(
            nn.Linear(graph_output_dim + fingerprint_output_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(shared_dim * 4, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(self, batch: HybridGraphBatch) -> torch.Tensor:
        graph = self.graph_encoder(batch.node_features, batch.adjacency, batch.mask)
        fingerprint = self.fingerprint_encoder(batch.fingerprints)
        drug = self.drug_projection(torch.cat([graph, fingerprint], dim=1))
        cell = self.cell_encoder(batch.cell_features)
        interaction = drug * cell
        contrast = torch.abs(drug - cell)
        fused = torch.cat([drug, cell, interaction, contrast], dim=1)
        return self.regressor(fused).squeeze(-1)


def to_device(batch: HybridGraphBatch, device: str) -> HybridGraphBatch:
    """Move a hybrid batch to a torch device."""

    return HybridGraphBatch(
        node_features=batch.node_features.to(device),
        adjacency=batch.adjacency.to(device),
        mask=batch.mask.to(device),
        fingerprints=batch.fingerprints.to(device),
        cell_features=batch.cell_features.to(device),
        targets=batch.targets.to(device),
    )


def build_fingerprint_map(
    cohort: pd.DataFrame,
    *,
    drug_column: str = "drug_id",
    smiles_column: str = "canonical_smiles",
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[str, np.ndarray]:
    """Build one Morgan fingerprint per unique drug."""

    unique_drugs = cohort[[drug_column, smiles_column]].drop_duplicates(drug_column)
    matrix = build_fingerprint_matrix(
        unique_drugs[smiles_column],
        radius=radius,
        n_bits=n_bits,
    ).astype(np.float32)
    return {
        str(drug_id): matrix[idx]
        for idx, drug_id in enumerate(unique_drugs[drug_column].astype(str))
    }


def make_loader(
    rows: pd.DataFrame,
    graph_map: dict[str, MolecularGraph],
    fingerprint_map: dict[str, np.ndarray],
    cell_feature_map: dict[str, np.ndarray],
    *,
    cell_scaler: StandardScaler,
    target_scaler: StandardScaler,
    target: str,
    batch_size: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    """Create a hybrid graph DataLoader."""

    dataset = HybridDrugResponseDataset(
        rows,
        graph_map,
        fingerprint_map,
        cell_feature_map,
        cell_scaler=cell_scaler,
        target_scaler=target_scaler,
        target=target,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_hybrid_graph_batch,
        pin_memory=device == "cuda",
    )


def evaluate(
    model: HybridGNNRegressor,
    loader: DataLoader,
    *,
    device: str,
    target_scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict one loader and optionally inverse-transform target scale."""

    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            pred = model(batch)
            loss = loss_fn(pred, batch.targets)
            losses.append(float(loss.detach().cpu().item()))
            predictions.append(pred.detach().cpu().numpy())
            targets.append(batch.targets.detach().cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    return y_true, y_pred, float(np.mean(losses))


def train_model(
    model: HybridGNNRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: str,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    random_state: int,
    log_every: int,
) -> tuple[HybridGNNRegressor, dict[str, Any]]:
    """Train B5 with AdamW, plateau scheduling, and early stopping."""

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, patience // 3),
        min_lr=1e-6,
    )
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.targets)
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        _y_val, _pred_val, val_loss = evaluate(model, val_loader, device=device)
        scheduler.step(val_loss)
        train_loss = float(np.mean(train_losses))
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "learning_rate": current_lr,
            }
        )

        if epoch == 1 or epoch % log_every == 0:
            logger.info(
                "B5 epoch %d/%d — train loss %.4f, validation loss %.4f, lr %.2e",
                epoch,
                max_epochs,
                train_loss,
                val_loss,
                current_lr,
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_ran": epoch,
        "history": history,
    }


def metric_row(
    model_name: str,
    split_name: str,
    subset: str,
    rows: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    train_info: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Create one metrics row."""

    # Validation and test loaders are unshuffled, so these align with ``rows``.
    metrics = regression_metrics(
        y_true,
        y_pred,
        cell_keys=rows["depmap_id"].tolist(),
        drug_keys=rows["drug_id"].tolist(),
    )
    return {
        "model": model_name,
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        "device": device,
        "best_epoch": int(train_info["best_epoch"]),
        "epochs_ran": int(train_info["epochs_ran"]),
        **metrics,
    }


def run_b5_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    graph_map: dict[str, MolecularGraph],
    fingerprint_map: dict[str, np.ndarray],
    cell_feature_map: dict[str, np.ndarray],
    *,
    split_name: str,
    target: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    graph_hidden_dim: int,
    graph_layers: int,
    graph_output_dim: int,
    fingerprint_hidden_dim: int,
    fingerprint_output_dim: int,
    cell_hidden_dim: int,
    shared_dim: int,
    fusion_hidden_dim: int,
    dropout: float,
    gradient_clip_norm: float,
    device: str,
    random_state: int,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Train/evaluate B5 for one split."""

    rows = split_rows(cohort, assignments)
    cell_scaler = fit_cell_scaler(rows["train"], cell_feature_map)
    target_scaler = fit_target_scaler(rows["train"], target)
    train_loader = make_loader(
        rows["train"],
        graph_map,
        fingerprint_map,
        cell_feature_map,
        cell_scaler=cell_scaler,
        target_scaler=target_scaler,
        target=target,
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        rows["validation"],
        graph_map,
        fingerprint_map,
        cell_feature_map,
        cell_scaler=cell_scaler,
        target_scaler=target_scaler,
        target=target,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )
    test_loader = make_loader(
        rows["test"],
        graph_map,
        fingerprint_map,
        cell_feature_map,
        cell_scaler=cell_scaler,
        target_scaler=target_scaler,
        target=target,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    logger.info(
        "B5 %s loaders ready — train=%d, validation=%d, test=%d, batch_size=%d",
        split_name,
        len(rows["train"]),
        len(rows["validation"]),
        len(rows["test"]),
        batch_size,
    )

    n_cell_features = next(iter(cell_feature_map.values())).shape[0]
    n_fingerprint_bits = len(next(iter(fingerprint_map.values())))
    model = HybridGNNRegressor(
        cell_dim=n_cell_features,
        fingerprint_dim=n_fingerprint_bits,
        graph_hidden_dim=graph_hidden_dim,
        graph_layers=graph_layers,
        graph_output_dim=graph_output_dim,
        fingerprint_hidden_dim=fingerprint_hidden_dim,
        fingerprint_output_dim=fingerprint_output_dim,
        cell_hidden_dim=cell_hidden_dim,
        shared_dim=shared_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout,
    )

    t0 = time.time()
    model, train_info = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        gradient_clip_norm=gradient_clip_norm,
        random_state=random_state,
        log_every=log_every,
    )
    fit_seconds = time.time() - t0
    logger.info(
        "B5 %s fitted in %.1fs on %s — best epoch %d, val loss %.4f",
        split_name,
        fit_seconds,
        device,
        train_info["best_epoch"],
        train_info["best_validation_loss"],
    )

    results: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for subset, loader in (("validation", val_loader), ("test", test_loader)):
        y_true, y_pred, _loss = evaluate(
            model,
            loader,
            device=device,
            target_scaler=target_scaler,
        )
        row = metric_row(
            "hybrid_gnn",
            split_name,
            subset,
            rows[subset],
            y_true,
            y_pred,
            train_info=train_info,
            device=device,
        )
        row["fit_seconds"] = fit_seconds
        row["graph_hidden_dim"] = graph_hidden_dim
        row["graph_layers"] = graph_layers
        row["graph_output_dim"] = graph_output_dim
        row["fingerprint_hidden_dim"] = fingerprint_hidden_dim
        row["fingerprint_output_dim"] = fingerprint_output_dim
        row["cell_hidden_dim"] = cell_hidden_dim
        row["shared_dim"] = shared_dim
        row["fusion_hidden_dim"] = fusion_hidden_dim
        row["dropout"] = dropout
        row["learning_rate"] = learning_rate
        row["weight_decay"] = weight_decay
        row["gradient_clip_norm"] = gradient_clip_norm
        row["target_mean_train"] = float(target_scaler.mean_[0])
        row["target_scale_train"] = float(target_scaler.scale_[0])
        results.append(row)
        frames.append(
            prediction_frame(
                rows[subset],
                y_true,
                y_pred,
                stage="B5",
                model="hybrid_gnn",
                split_name=split_name,
                subset=subset,
            )
        )

    return results, frames


def run_b5(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/models/b5_hybrid_gnn_metrics.csv",
    summary: str | Path = "data/reports/b5_hybrid_gnn_summary.json",
    predictions_output: str | Path = "results/predictions/b5.csv",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    batch_size: int = 128,
    max_epochs: int = 120,
    patience: int = 16,
    learning_rate: float = 0.0005,
    weight_decay: float = 0.0003,
    graph_hidden_dim: int = 256,
    graph_layers: int = 5,
    graph_output_dim: int = 256,
    fingerprint_hidden_dim: int = 512,
    fingerprint_output_dim: int = 256,
    cell_hidden_dim: int = 256,
    shared_dim: int = 256,
    fusion_hidden_dim: int = 512,
    dropout: float = 0.15,
    gradient_clip_norm: float = 5.0,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    device: str = "auto",
    random_state: int = 42,
    log_every: int = 1,
) -> pd.DataFrame:
    """Run the heavier hybrid GNN model."""

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    torch_device = resolve_torch_device(device)
    if torch_device is None:
        raise RuntimeError("B5 requires PyTorch. Install torch before running this model.")

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)
    logger.info("Building molecular graph map for %d cohort rows ...", len(cohort))
    graph_map = build_graph_map(cohort)
    logger.info("Built molecular graph map for %d unique drugs.", len(graph_map))
    logger.info("Building Morgan fingerprint map for %d cohort rows ...", len(cohort))
    fingerprint_map = build_fingerprint_map(
        cohort,
        radius=fingerprint_radius,
        n_bits=fingerprint_bits,
    )
    logger.info("Built fingerprint map for %d unique drugs.", len(fingerprint_map))

    all_results: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    for split_name in split_names:
        logger.info("=== Processing B5 split: %s ===", split_name)
        assignments = pd.read_csv(split_dir / f"{split_name}.csv")
        merged = cohort.merge(assignments, on="pair_id", how="inner")
        train_ids = set(merged.loc[merged["split"].eq("train"), "depmap_id"].unique())
        pipeline, cell_feature_map = build_cell_features(
            str(expression_path),
            cohort,
            train_ids,
            n_components=n_components,
            random_state=random_state,
        )
        logger.info(
            "PCA explained variance: %.1f%%",
            pipeline.explained_variance_ratio_sum * 100,
        )
        split_results, split_frames = run_b5_for_split(
            cohort,
            assignments,
            graph_map,
            fingerprint_map,
            cell_feature_map,
            split_name=split_name,
            target=target,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            graph_hidden_dim=graph_hidden_dim,
            graph_layers=graph_layers,
            graph_output_dim=graph_output_dim,
            fingerprint_hidden_dim=fingerprint_hidden_dim,
            fingerprint_output_dim=fingerprint_output_dim,
            cell_hidden_dim=cell_hidden_dim,
            shared_dim=shared_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            gradient_clip_norm=gradient_clip_norm,
            device=torch_device,
            random_state=random_state,
            log_every=log_every,
        )
        all_results.extend(split_results)
        all_frames.extend(split_frames)

    metrics = pd.DataFrame(all_results)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    write_predictions(all_frames, predictions_output)

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "expression_path": str(expression_path),
        "split_dir": str(split_dir),
        "output": str(output),
        "predictions_output": str(predictions_output),
        "target": target,
        "n_pca_components": n_components,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "graph_hidden_dim": graph_hidden_dim,
        "graph_layers": graph_layers,
        "graph_output_dim": graph_output_dim,
        "fingerprint_hidden_dim": fingerprint_hidden_dim,
        "fingerprint_output_dim": fingerprint_output_dim,
        "cell_hidden_dim": cell_hidden_dim,
        "shared_dim": shared_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
        "dropout": dropout,
        "gradient_clip_norm": gradient_clip_norm,
        "fingerprint_radius": fingerprint_radius,
        "fingerprint_bits": fingerprint_bits,
        "graph_pooling": "mean_max_attention",
        "fusion": "drug_cell_interaction_and_absolute_contrast",
        "loss": "smooth_l1_beta_0.5_on_train_scaled_target",
        "optimizer": "adamw_reduce_lr_on_plateau",
        "target_scaling": "standard_scaler_fit_on_train",
        "device_requested": device,
        "device_used": torch_device,
        "random_state": random_state,
        "log_every": log_every,
        "splits": list(split_names),
        "best_by_split_subset": summarize_best(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)
    return metrics


def summarize_best(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize B5 metrics for each split/subset."""

    rows: list[dict[str, Any]] = []
    for (split_name, subset), group in metrics.groupby(["split_name", "subset"]):
        best = group.sort_values("rmse").iloc[0]
        rows.append(
            {
                "split_name": split_name,
                "subset": subset,
                "best_model": best["model"],
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "pearson": float(best["pearson"])
                if pd.notna(best["pearson"])
                else None,
                "spearman": float(best["spearman"])
                if pd.notna(best["spearman"])
                else None,
                "r2": float(best["r2"]) if pd.notna(best["r2"]) else None,
                "best_epoch": int(best["best_epoch"]),
                "device": best["device"],
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Run B5 hybrid multimodal GNN.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument("--output", default="results/models/b5_hybrid_gnn_metrics.csv")
    parser.add_argument("--summary", default="data/reports/b5_hybrid_gnn_summary.json")
    parser.add_argument("--predictions-output", default="results/predictions/b5.csv")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--graph-hidden-dim", type=int, default=256)
    parser.add_argument("--graph-layers", type=int, default=5)
    parser.add_argument("--graph-output-dim", type=int, default=256)
    parser.add_argument("--fingerprint-hidden-dim", type=int, default=512)
    parser.add_argument("--fingerprint-output-dim", type=int, default=256)
    parser.add_argument("--cell-hidden-dim", type=int, default=256)
    parser.add_argument("--shared-dim", type=int, default=256)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()
    metrics = run_b5(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        predictions_output=args.predictions_output,
        target=args.target,
        n_components=args.n_components,
        split_names=tuple(args.splits),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_layers=args.graph_layers,
        graph_output_dim=args.graph_output_dim,
        fingerprint_hidden_dim=args.fingerprint_hidden_dim,
        fingerprint_output_dim=args.fingerprint_output_dim,
        cell_hidden_dim=args.cell_hidden_dim,
        shared_dim=args.shared_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_bits=args.fingerprint_bits,
        device=args.device,
        random_state=args.random_state,
        log_every=args.log_every,
    )
    print(f"\nWrote B5 hybrid GNN metrics to {args.output}")
    print("=" * 72)
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"  {row['split_name']:>12s} / {row['subset']:>10s} / "
            f"{row['model']:>12s}: RMSE={row['rmse']:.3f}  "
            f"MAE={row['mae']:.3f}  Pearson={row['pearson']:.3f}  "
            f"R²={row['r2']:.3f}  (epoch={row['best_epoch']:.0f}, "
            f"device={row['device']})"
        )


if __name__ == "__main__":
    main()
