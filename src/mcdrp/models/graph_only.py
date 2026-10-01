"""Drug-only graph control with pair-weighted training and no cell inputs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from mcdrp.features.graphs import build_graph_map
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b3 import resolve_torch_device
from mcdrp.models.gnn_b4 import DrugGraphEncoder, GraphBatch, collate_graph_batch, to_device
from mcdrp.results.predictions import prediction_frame, write_predictions
from mcdrp.splits.validation import validate_assignments


class GraphOnlyRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.encoder = DrugGraphEncoder(hidden_dim=hidden_dim, n_layers=layers, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        return self.head(
            self.encoder(batch.node_features, batch.adjacency, batch.mask)
        ).squeeze(-1)


def run_graph_only(
    cohort_path: str,
    split_dir: str,
    output_dir: str,
    *,
    split_name: str = "cold_drug",
    target: str = "ln_ic50",
    seed: int = 42,
    epochs: int = 200,
    patience: int = 20,
    device: str = "auto",
) -> dict[str, Any]:
    cohort = pd.read_csv(cohort_path, low_memory=False)
    assignments = pd.read_csv(Path(split_dir) / f"{split_name}.csv")
    validate_assignments(cohort, assignments)
    data = cohort.merge(assignments, on="pair_id", validate="one_to_one")
    graph_map = build_graph_map(cohort)
    if not all(graph.valid for graph in graph_map.values()):
        raise ValueError("Graph-only control requires valid molecular graphs.")
    ids = sorted(graph_map)
    # Empty cell vectors are present only to reuse padding; forward never accesses them.
    batch = collate_graph_batch(
        [
            {
                "node_features": graph_map[key].node_features,
                "adjacency": graph_map[key].adjacency,
                "cell_features": np.zeros(0, dtype=np.float32),
                "target": 0.0,
            }
            for key in ids
        ]
    )
    train = data.loc[data["split"].eq("train")]
    validation = data.loc[data["split"].eq("validation")]
    mean, scale = float(train[target].mean()), float(train[target].std(ddof=0))
    scale = scale if scale > 1e-8 else 1.0
    grouped = train.groupby("drug_id")[target].agg(["mean", "count"])
    index = {key: idx for idx, key in enumerate(ids)}
    train_indices = torch.tensor([index[str(key)] for key in grouped.index], dtype=torch.long)
    val_indices = torch.tensor(
        [index[str(key)] for key in validation["drug_id"]], dtype=torch.long
    )
    actual_device = resolve_torch_device(device)
    if actual_device is None:
        raise RuntimeError("Graph-only fitting requires PyTorch.")
    batch = to_device(batch, actual_device)
    train_indices, val_indices = train_indices.to(actual_device), val_indices.to(actual_device)
    means = torch.tensor(
        (grouped["mean"].to_numpy() - mean) / scale, dtype=torch.float32, device=actual_device
    )
    weights = torch.tensor(
        grouped["count"].to_numpy() / len(train), dtype=torch.float32, device=actual_device
    )
    val_y = torch.tensor(
        (validation[target].to_numpy() - mean) / scale,
        dtype=torch.float32,
        device=actual_device,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = GraphOnlyRegressor().to(actual_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best, best_state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(batch)
        loss = (weights * (predictions[train_indices] - means).square()).sum()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float((model(batch)[val_indices] - val_y).square().mean().item())
        if val_loss < best:
            best, best_state, stale, best_epoch = (
                val_loss,
                copy.deepcopy(model.state_dict()),
                0,
                epoch + 1,
            )
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError("Graph training produced no finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        drug_predictions = model(batch).cpu().numpy() * scale + mean
    lookup = dict(zip(ids, drug_predictions, strict=True))
    frames, metrics = [], []
    for subset in ("validation", "calibration", "test"):
        rows = data.loc[data["split"].eq(subset)]
        if rows.empty:
            continue
        pred = np.asarray([lookup[str(key)] for key in rows["drug_id"]])
        frames.append(
            prediction_frame(
                rows,
                rows[target].to_numpy(),
                pred,
                stage="B4",
                model="graph_only",
                split_name=split_name,
                subset=subset,
            )
        )
        metrics.append(
            {
                "subset": subset,
                **regression_metrics(
                    rows[target].to_numpy(),
                    pred,
                    cell_keys=rows["depmap_id"].tolist(),
                    drug_keys=rows["drug_id"].tolist(),
                ),
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_predictions(frames, output / "predictions.csv")
    report = {
        "seed": seed,
        "best_epoch": best_epoch,
        "device": actual_device,
        "objective": "Drug-mean weighted MSE equals pair MSE up to constant "
        "within-drug variance; dropout=0, full batch.",
        "metrics": json.loads(pd.DataFrame(metrics).to_json(orient="records")),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", dest="cohort_path", default="data/processed/cohort_pairs.csv"
    )
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="cold_drug")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    print(json.dumps(run_graph_only(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
