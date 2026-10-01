"""Small drug/cell towers with additive heads and a learned interaction.

The caller supplies train and validation only to fit. Drug/cell feature tables
are indexed on device rather than duplicated for every response pair.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if nn is not None:

    class ResidualTowers(nn.Module):
        def __init__(self, drug_dim: int, cell_dim: int, width: int, dropout: float):
            super().__init__()

            def tower(dim):
                return nn.Sequential(
                    nn.Linear(dim, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, width),
                    nn.GELU(),
                )

            self.drug = tower(drug_dim)
            self.cell = tower(cell_dim)
            self.drug_head = nn.Linear(width, 1)
            self.cell_head = nn.Linear(width, 1)
            self.interaction = nn.Sequential(
                nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)
            )

        def forward(self, drug, cell):
            d, c = self.drug(drug), self.cell(cell)
            return (
                self.drug_head(d)
                + self.cell_head(c)
                + self.interaction(torch.cat([d, c, d * c], dim=1))
            ).squeeze(1)


def feature_tables(
    rows: pd.DataFrame,
    features: dict,
    blocks: list[str],
    train: pd.DataFrame,
    scalers: dict | None = None,
) -> dict:
    """Train-only scaling of continuous blocks; binary inputs remain binary."""
    drugs = sorted(rows["drug_id"].astype(str).unique())
    cells = sorted(rows["depmap_id"].astype(str).unique())
    train_drugs = set(train["drug_id"].astype(str))
    train_cells = set(train["depmap_id"].astype(str))
    fitted = {} if scalers is None else scalers.copy()
    matrices = []
    for block in blocks:
        matrix = np.vstack([features[block][d] for d in drugs]).astype(np.float32)
        if block == "embeddings":
            if scalers is None:
                fitted[block] = StandardScaler().fit(matrix[[d in train_drugs for d in drugs]])
            matrix = fitted[block].transform(matrix)
        matrices.append(matrix)
    cell = np.vstack([features["cells"][c] for c in cells]).astype(np.float32)
    if scalers is None:
        fitted["cells"] = StandardScaler().fit(cell[[c in train_cells for c in cells]])
    return {
        "drug": np.hstack(matrices).astype(np.float32),
        "cell": fitted["cells"].transform(cell).astype(np.float32),
        "drug_index": {d: i for i, d in enumerate(drugs)},
        "cell_index": {c: i for i, c in enumerate(cells)},
        "scalers": fitted,
    }


def indices(rows: pd.DataFrame, tables: dict, device: str):
    d = [tables["drug_index"][str(x)] for x in rows["drug_id"]]
    c = [tables["cell_index"][str(x)] for x in rows["depmap_id"]]
    return torch.tensor(d, device=device), torch.tensor(c, device=device)


def predict(
    model,
    rows: pd.DataFrame,
    tables: dict,
    *,
    device: str,
    mean: float,
    scale: float,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    d, c = indices(rows, tables, device)
    drug = torch.as_tensor(tables["drug"], device=device)
    cell = torch.as_tensor(tables["cell"], device=device)
    result = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            result.append(
                model(drug[d[i : i + batch_size]], cell[c[i : i + batch_size]]).cpu().numpy()
            )
    return np.concatenate(result).astype(float) * scale + mean


def restore(artifact: dict, device: str):
    model = ResidualTowers(**artifact["dimensions"]).to(device)
    model.load_state_dict(
        {k: torch.as_tensor(v, device=device) for k, v in artifact["state"].items()}
    )
    return model


def fit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tables: dict,
    *,
    target: str,
    candidate: dict,
    config: dict,
    seed: int,
) -> tuple[dict, list[dict]]:
    if torch is None:
        raise RuntimeError("This experiment requires PyTorch; there is no backend fallback.")
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable.")
    torch.set_num_threads(8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    dimensions = {
        "drug_dim": tables["drug"].shape[1],
        "cell_dim": tables["cell"].shape[1],
        "width": candidate["width"],
        "dropout": config["dropout"],
    }
    model = ResidualTowers(**dimensions).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    mean, scale = float(train[target].mean()), float(train[target].std(ddof=0))
    scale = max(scale, 1e-8)
    y = torch.tensor((train[target].to_numpy(np.float32) - mean) / scale, device=device)
    d, c = indices(train, tables, device)
    drug = torch.as_tensor(tables["drug"], device=device)
    cell = torch.as_tensor(tables["cell"], device=device)
    history: list[dict[str, Any]] = []
    best_score, stale, best_state, best_epoch = float("inf"), 0, None, 0
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(train), device=device)
        total = 0.0
        for start in range(0, len(train), config["batch_size"]):
            batch = order[start : start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(drug[d[batch]], cell[c[batch]]), y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        val = predict(model, validation, tables, device=device, mean=mean, scale=scale)
        score = float(np.sqrt(np.mean((val - validation[target].to_numpy()) ** 2)))
        history.append(
            {
                "epoch": epoch + 1,
                "train_rmse": np.sqrt(total / len(train)) * scale,
                "validation_rmse": score,
            }
        )
        if score < best_score - 1e-6:
            best_score, stale, best_epoch = score, 0, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if best_state is None:
        raise ValueError("No finite validation checkpoint.")
    return {
        "dimensions": dimensions,
        "state": {k: v.cpu().numpy() for k, v in best_state.items()},
        "mean": mean,
        "scale": scale,
        "seed": seed,
        "blocks": candidate["blocks"],
        "scalers": tables["scalers"],
        "best_epoch": best_epoch,
        "validation_rmse": best_score,
    }, history
