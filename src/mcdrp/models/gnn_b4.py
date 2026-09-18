"""B4 multimodal GNN baseline.

This model combines:

- a pure-PyTorch molecular graph encoder
- PCA-reduced cell-line expression features
- a fusion MLP for ln(IC50) regression

It is intentionally dependency-light. It does not use PyTorch Geometric yet;
that can come after this baseline proves the graph path is wired correctly.
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
        "B4 GNN requires PyTorch. Install the gpu extra or torch directly."
    ) from exc

from mcdrp.features.expression import build_cell_features
from mcdrp.features.graphs import ATOM_FEATURE_DIM, MolecularGraph, build_graph_map
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b3 import resolve_torch_device
from mcdrp.splits.make_splits import DEFAULT_SPLITS

logger = logging.getLogger(__name__)

EXPRESSION_PATH = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"


@dataclass
class GraphBatch:
    """Batched graph tensors."""

    node_features: torch.Tensor
    adjacency: torch.Tensor
    mask: torch.Tensor
    cell_features: torch.Tensor
    targets: torch.Tensor


class DrugResponseGraphDataset(Dataset[dict[str, Any]]):
    """Dataset returning one drug graph + cell features + target per pair."""

    def __init__(
        self,
        rows: pd.DataFrame,
        graph_map: dict[str, MolecularGraph],
        cell_feature_map: dict[str, np.ndarray],
        *,
        cell_scaler: StandardScaler,
        target_scaler: StandardScaler,
        target: str,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.graphs: list[MolecularGraph] = []
        self.cell_features: list[np.ndarray] = []
        self.targets: list[np.float32] = []

        n_cell_features = cell_scaler.n_features_in_
        for row in self.rows.itertuples(index=False):
            self.graphs.append(graph_map[str(row.drug_id)])
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
            "cell_features": self.cell_features[idx],
            "target": self.targets[idx],
        }


def collate_graph_batch(samples: list[dict[str, Any]]) -> GraphBatch:
    """Pad variable-size molecular graphs into one batch."""

    batch_size = len(samples)
    max_nodes = max(sample["node_features"].shape[0] for sample in samples)
    node_features = np.zeros((batch_size, max_nodes, ATOM_FEATURE_DIM), dtype=np.float32)
    adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros((batch_size, max_nodes), dtype=np.float32)
    cell_features = np.stack([sample["cell_features"] for sample in samples])
    targets = np.asarray([sample["target"] for sample in samples], dtype=np.float32)

    for idx, sample in enumerate(samples):
        n_nodes = sample["node_features"].shape[0]
        node_features[idx, :n_nodes] = sample["node_features"]
        adjacency[idx, :n_nodes, :n_nodes] = sample["adjacency"]
        mask[idx, :n_nodes] = 1.0

    return GraphBatch(
        node_features=torch.from_numpy(node_features),
        adjacency=torch.from_numpy(adjacency),
        mask=torch.from_numpy(mask),
        cell_features=torch.from_numpy(cell_features),
        targets=torch.from_numpy(targets),
    )


class GraphConvBlock(nn.Module):
    """Mean-aggregation graph convolution block with residual normalization."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim)
        self.residual = (
            nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.bmm(adjacency, x)
        output = self.self_linear(x) + self.neighbor_linear(neighbors)
        output = self.norm(output + self.residual(x))
        return self.dropout(self.activation(output))


class DrugGraphEncoder(nn.Module):
    """Encode padded molecular graphs into fixed-size vectors."""

    def __init__(
        self,
        *,
        atom_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[GraphConvBlock] = []
        input_dim = atom_dim
        for _ in range(n_layers):
            layers.append(GraphConvBlock(input_dim, hidden_dim, dropout))
            input_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

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
        pooled = masked_x.sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pool = pooled / counts
        max_pool = masked_x.masked_fill(mask.unsqueeze(-1).eq(0), -1e9).max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=1)


class MultimodalGNNRegressor(nn.Module):
    """Drug graph + cell expression regression model."""

    def __init__(
        self,
        *,
        cell_dim: int,
        graph_hidden_dim: int = 128,
        graph_layers: int = 3,
        cell_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drug_encoder = DrugGraphEncoder(
            hidden_dim=graph_hidden_dim,
            n_layers=graph_layers,
            dropout=dropout,
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, cell_hidden_dim),
            nn.LayerNorm(cell_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear((graph_hidden_dim * 2) + cell_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        drug = self.drug_encoder(batch.node_features, batch.adjacency, batch.mask)
        cell = self.cell_encoder(batch.cell_features)
        fused = torch.cat([drug, cell], dim=1)
        return self.regressor(fused).squeeze(-1)


def to_device(batch: GraphBatch, device: str) -> GraphBatch:
    """Move a graph batch to torch device."""

    return GraphBatch(
        node_features=batch.node_features.to(device),
        adjacency=batch.adjacency.to(device),
        mask=batch.mask.to(device),
        cell_features=batch.cell_features.to(device),
        targets=batch.targets.to(device),
    )


def fit_cell_scaler(
    train_rows: pd.DataFrame,
    cell_feature_map: dict[str, np.ndarray],
) -> StandardScaler:
    """Fit a cell feature scaler on training rows only."""

    values = []
    n_components = next(iter(cell_feature_map.values())).shape[0]
    for depmap_id in train_rows["depmap_id"]:
        values.append(cell_feature_map.get(str(depmap_id), np.zeros(n_components)))
    scaler = StandardScaler()
    scaler.fit(np.asarray(values, dtype=np.float32))
    return scaler


def fit_target_scaler(train_rows: pd.DataFrame, target: str) -> StandardScaler:
    """Fit the target scaler on training labels only."""

    scaler = StandardScaler()
    scaler.fit(train_rows[[target]].to_numpy(dtype=np.float32))
    return scaler


def split_rows(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Merge cohort with assignments and return train/validation/test rows."""

    data = cohort.merge(assignments, on="pair_id", how="inner", validate="one_to_one")
    return {
        subset: data.loc[data["split"].eq(subset)].copy()
        for subset in ("train", "validation", "test")
    }


def make_loader(
    rows: pd.DataFrame,
    graph_map: dict[str, MolecularGraph],
    cell_feature_map: dict[str, np.ndarray],
    *,
    cell_scaler: StandardScaler,
    target_scaler: StandardScaler,
    target: str,
    batch_size: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    """Create a graph DataLoader."""

    dataset = DrugResponseGraphDataset(
        rows,
        graph_map,
        cell_feature_map,
        cell_scaler=cell_scaler,
        target_scaler=target_scaler,
        target=target,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graph_batch,
        pin_memory=device == "cuda",
    )


def evaluate(
    model: MultimodalGNNRegressor,
    loader: DataLoader,
    *,
    device: str,
    target_scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict a loader and return y_true, y_pred, and mean scaled loss."""

    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    loss_fn = nn.MSELoss()
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
    model: MultimodalGNNRegressor,
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
) -> tuple[MultimodalGNNRegressor, dict[str, Any]]:
    """Train with validation early stopping."""

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

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
        train_loss = float(np.mean(train_losses))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": val_loss,
            }
        )

        if epoch == 1 or epoch % log_every == 0:
            logger.info(
                "B4 epoch %d/%d — train loss %.4f, validation loss %.4f",
                epoch,
                max_epochs,
                train_loss,
                val_loss,
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
        "gradient_clip_norm": gradient_clip_norm,
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


def run_b4_for_split(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    graph_map: dict[str, MolecularGraph],
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
    cell_hidden_dim: int,
    fusion_hidden_dim: int,
    dropout: float,
    gradient_clip_norm: float,
    device: str,
    random_state: int,
    log_every: int,
) -> list[dict[str, Any]]:
    """Train/evaluate B4 for one split."""

    rows = split_rows(cohort, assignments)
    cell_scaler = fit_cell_scaler(rows["train"], cell_feature_map)
    target_scaler = fit_target_scaler(rows["train"], target)
    train_loader = make_loader(
        rows["train"],
        graph_map,
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
        cell_feature_map,
        cell_scaler=cell_scaler,
        target_scaler=target_scaler,
        target=target,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    logger.info(
        "B4 %s loaders ready — train=%d, validation=%d, test=%d, batch_size=%d",
        split_name,
        len(rows["train"]),
        len(rows["validation"]),
        len(rows["test"]),
        batch_size,
    )

    n_cell_features = next(iter(cell_feature_map.values())).shape[0]
    model = MultimodalGNNRegressor(
        cell_dim=n_cell_features,
        graph_hidden_dim=graph_hidden_dim,
        graph_layers=graph_layers,
        cell_hidden_dim=cell_hidden_dim,
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
        "B4 %s fitted in %.1fs on %s — best epoch %d, val loss %.4f",
        split_name,
        fit_seconds,
        device,
        train_info["best_epoch"],
        train_info["best_validation_loss"],
    )

    results: list[dict[str, Any]] = []
    for subset, loader in (("validation", val_loader), ("test", test_loader)):
        y_true, y_pred, _loss = evaluate(
            model,
            loader,
            device=device,
            target_scaler=target_scaler,
        )
        row = metric_row(
            "gnn",
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
        row["cell_hidden_dim"] = cell_hidden_dim
        row["fusion_hidden_dim"] = fusion_hidden_dim
        row["dropout"] = dropout
        row["learning_rate"] = learning_rate
        row["weight_decay"] = weight_decay
        row["gradient_clip_norm"] = gradient_clip_norm
        row["target_mean_train"] = float(target_scaler.mean_[0])
        row["target_scale_train"] = float(target_scaler.scale_[0])
        results.append(row)

    return results


def run_b4(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/baselines/b4_gnn_metrics.csv",
    summary: str | Path = "data/reports/b4_gnn_summary.json",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    batch_size: int = 256,
    max_epochs: int = 80,
    patience: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    graph_hidden_dim: int = 128,
    graph_layers: int = 3,
    cell_hidden_dim: int = 128,
    fusion_hidden_dim: int = 256,
    dropout: float = 0.1,
    gradient_clip_norm: float = 5.0,
    device: str = "auto",
    random_state: int = 42,
    log_every: int = 1,
) -> pd.DataFrame:
    """Run the initial GNN baseline."""

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    torch_device = resolve_torch_device(device)
    if torch_device is None:
        raise RuntimeError("B4 requires PyTorch. Install torch before running this model.")

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)
    logger.info("Building molecular graph map for %d cohort rows ...", len(cohort))
    graph_map = build_graph_map(cohort)
    logger.info("Built molecular graph map for %d unique drugs.", len(graph_map))

    all_results: list[dict[str, Any]] = []
    for split_name in split_names:
        logger.info("=== Processing B4 split: %s ===", split_name)
        assignments = pd.read_csv(split_dir / f"{split_name}.csv")
        merged = cohort.merge(assignments, on="pair_id", how="inner")
        train_ids = set(merged.loc[merged["split"].eq("train"), "depmap_id"].unique())
        pipeline, cell_feature_map = build_cell_features(
            str(expression_path),
            cohort,
            train_ids,
            n_components=n_components,
        )
        logger.info(
            "PCA explained variance: %.1f%%",
            pipeline.explained_variance_ratio_sum * 100,
        )
        all_results.extend(
            run_b4_for_split(
                cohort,
                assignments,
                graph_map,
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
                cell_hidden_dim=cell_hidden_dim,
                fusion_hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                gradient_clip_norm=gradient_clip_norm,
                device=torch_device,
                random_state=random_state,
                log_every=log_every,
            )
        )

    metrics = pd.DataFrame(all_results)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    summary_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "expression_path": str(expression_path),
        "split_dir": str(split_dir),
        "output": str(output),
        "target": target,
        "n_pca_components": n_components,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "graph_hidden_dim": graph_hidden_dim,
        "graph_layers": graph_layers,
        "cell_hidden_dim": cell_hidden_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
        "dropout": dropout,
        "gradient_clip_norm": gradient_clip_norm,
        "graph_pooling": "mean_max",
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
    """Summarize B4 metrics for each split/subset."""

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

    parser = argparse.ArgumentParser(description="Run B4 multimodal GNN baseline.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument("--output", default="results/baselines/b4_gnn_metrics.csv")
    parser.add_argument("--summary", default="data/reports/b4_gnn_summary.json")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--graph-hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--cell-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
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
    metrics = run_b4(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
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
        cell_hidden_dim=args.cell_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        device=args.device,
        random_state=args.random_state,
        log_every=args.log_every,
    )
    print(f"\nWrote B4 GNN metrics to {args.output}")
    print("=" * 72)
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"  {row['split_name']:>12s} / {row['subset']:>10s} / {row['model']:>8s}: "
            f"RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}  "
            f"Pearson={row['pearson']:.3f}  R²={row['r2']:.3f}  "
            f"(epoch={row['best_epoch']:.0f}, device={row['device']})"
        )


if __name__ == "__main__":
    main()
