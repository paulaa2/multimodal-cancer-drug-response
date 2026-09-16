"""B3 neural baseline: MLP with Morgan FP + PCA expression features.

This baseline uses the same feature engineering as B1 and B2:

- Morgan fingerprints for the drug
- PCA-reduced gene expression for the cell line

The predictor is a small feed-forward MLP. When PyTorch is installed, training
uses PyTorch and can run on CUDA with ``--device cuda`` or ``--device auto``.
If PyTorch is not installed and CPU mode is requested, it falls back to
scikit-learn's MLPRegressor.
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised only without torch installed.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from mcdrp.features.expression import build_cell_features
from mcdrp.features.fingerprints import build_fingerprint_matrix
from mcdrp.metrics import regression_metrics
from mcdrp.models.baseline_b1 import assemble_features
from mcdrp.splits.make_splits import DEFAULT_SPLITS

logger = logging.getLogger(__name__)

EXPRESSION_PATH = "data/raw/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv"


if nn is not None:

    class TorchMLP(nn.Module):
        """Simple fully connected regressor."""

        def __init__(self, input_dim: int, hidden_layer_sizes: tuple[int, ...]) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            previous = input_dim
            for hidden in hidden_layer_sizes:
                layers.append(nn.Linear(previous, hidden))
                layers.append(nn.ReLU())
                previous = hidden
            layers.append(nn.Linear(previous, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.network(features)

else:
    TorchMLP = None


@dataclass
class MLPFit:
    """Container for either a PyTorch or sklearn MLP fit."""

    model: Any
    backend: str
    device: str
    n_iter: int
    loss: float


def prepare_split_features(
    cohort: pd.DataFrame,
    split_assignments: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Build train/validation/test feature matrices for one split."""

    data = cohort.merge(
        split_assignments, on="pair_id", how="inner", validate="one_to_one"
    )
    pair_id_to_idx = {pid: idx for idx, pid in enumerate(cohort["pair_id"])}

    subsets: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for label in ("train", "validation", "test"):
        rows = data.loc[data["split"].eq(label)].copy()
        indices = np.array([pair_id_to_idx[pid] for pid in rows["pair_id"]])
        features = assemble_features(
            rows,
            fp_matrix,
            cell_feature_map,
            row_indices=indices,
        )
        subsets[label] = (rows, features)
    return subsets


def fit_scaler(
    subsets: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Fit a StandardScaler on train only and transform all subsets."""

    scaler = StandardScaler(copy=False)
    train_rows, x_train = subsets["train"]
    scaled_train = scaler.fit_transform(x_train)

    scaled: dict[str, tuple[pd.DataFrame, np.ndarray]] = {
        "train": (train_rows, scaled_train)
    }
    for label in ("validation", "test"):
        rows, features = subsets[label]
        scaled[label] = (rows, scaler.transform(features))
    return scaled


def resolve_torch_device(device: str) -> str | None:
    """Resolve the requested device into a PyTorch device string."""

    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported MLP device: {device}")

    if torch is None:
        if device == "cuda":
            raise RuntimeError(
                "CUDA was requested, but PyTorch is not installed. "
                "Install a CUDA-enabled PyTorch build, or use --device auto/--device cpu."
            )
        return None

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        cuda_version = getattr(torch.version, "cuda", None)
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a CUDA GPU. "
            f"torch.__version__={torch.__version__!r}, "
            f"torch.version.cuda={cuda_version!r}. "
            "This usually means the installed PyTorch build is CPU-only, the NVIDIA "
            "driver is unavailable, or this machine has no CUDA-capable GPU visible. "
            "Use --device auto to fall back to CPU, or install a CUDA-enabled PyTorch build."
        )
    return device


def fit_mlp_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    batch_size: int,
    max_iter: int,
    early_stopping: bool,
    validation_fraction: float,
    random_state: int,
    device: str,
) -> MLPFit:
    """Fit the MLP using PyTorch when available, otherwise sklearn on CPU."""

    torch_device = resolve_torch_device(device)
    if torch_device is None:
        if device == "cuda":
            raise RuntimeError("PyTorch is required for CUDA training.")
        return fit_sklearn_mlp(
            x_train,
            y_train,
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
    return fit_torch_mlp(
        x_train,
        y_train,
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=random_state,
        device=torch_device,
    )


def fit_sklearn_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    batch_size: int,
    max_iter: int,
    early_stopping: bool,
    validation_fraction: float,
    random_state: int,
) -> MLPFit:
    """CPU fallback when PyTorch is not installed."""

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=12,
        random_state=random_state,
        verbose=False,
    )
    model.fit(x_train, y_train)
    return MLPFit(
        model=model,
        backend="sklearn",
        device="cpu",
        n_iter=int(model.n_iter_),
        loss=float(model.loss_),
    )


def fit_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    batch_size: int,
    max_iter: int,
    early_stopping: bool,
    validation_fraction: float,
    random_state: int,
    device: str,
) -> MLPFit:
    """Fit a PyTorch MLP on CPU or CUDA."""

    if torch is None or TorchMLP is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for torch MLP training.")

    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)

    features = np.asarray(x_train, dtype=np.float32)
    targets = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

    train_idx = np.arange(len(features))
    val_idx: np.ndarray | None = None
    if early_stopping and len(features) > 1 and validation_fraction > 0:
        shuffled = rng.permutation(len(features))
        val_size = max(1, int(len(features) * validation_fraction))
        val_idx = shuffled[:val_size]
        train_idx = shuffled[val_size:]
        if len(train_idx) == 0:
            train_idx = shuffled
            val_idx = None

    model = TorchMLP(features.shape[1], hidden_layer_sizes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_init,
        weight_decay=alpha,
    )
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(features[train_idx]),
        torch.from_numpy(targets[train_idx]),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
    )

    x_val: torch.Tensor | None = None
    y_val: torch.Tensor | None = None
    if val_idx is not None:
        x_val = torch.from_numpy(features[val_idx]).to(device)
        y_val = torch.from_numpy(targets[val_idx]).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    epochs_without_improvement = 0
    last_train_loss = float("nan")

    for epoch in range(1, max_iter + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        last_train_loss = float(np.mean(batch_losses))
        score = last_train_loss
        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                score = float(loss_fn(model(x_val), y_val).detach().cpu().item())

        if score < best_score - 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping and epochs_without_improvement >= 12:
                break

    model.load_state_dict(best_state)
    return MLPFit(
        model=model,
        backend="torch",
        device=device,
        n_iter=epoch,
        loss=last_train_loss,
    )


def predict_mlp(fit: MLPFit, features: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
    """Predict with either a PyTorch or sklearn MLP fit."""

    if fit.backend == "sklearn":
        return np.asarray(fit.model.predict(features), dtype=np.float32)

    if torch is None:
        raise RuntimeError("PyTorch is required to predict with a torch MLP.")

    model = fit.model
    model.eval()
    outputs: list[np.ndarray] = []
    values = np.asarray(features, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(fit.device)
            pred = model(batch).detach().cpu().numpy().reshape(-1)
            outputs.append(pred)
    return np.concatenate(outputs)


def metric_row(
    fit: MLPFit,
    rows: pd.DataFrame,
    features: np.ndarray,
    *,
    split_name: str,
    subset: str,
    target: str,
) -> dict[str, Any]:
    """Predict one subset and return standard regression metrics."""

    predictions = predict_mlp(fit, features)
    metrics = regression_metrics(rows[target].to_numpy(), predictions)
    return {
        "model": "mlp",
        "split_name": split_name,
        "subset": subset,
        "n_rows": int(len(rows)),
        "n_cell_lines": int(rows["depmap_id"].nunique()),
        "n_drugs": int(rows["drug_id"].nunique()),
        "backend": fit.backend,
        "device": fit.device,
        "n_iter": fit.n_iter,
        "loss": fit.loss,
        **metrics,
    }


def run_b3_for_split(
    cohort: pd.DataFrame,
    split_assignments: pd.DataFrame,
    fp_matrix: np.ndarray,
    cell_feature_map: dict[str, np.ndarray],
    *,
    split_name: str,
    target: str,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    batch_size: int,
    max_iter: int,
    early_stopping: bool,
    validation_fraction: float,
    random_state: int,
    device: str,
) -> list[dict[str, Any]]:
    """Run the MLP neural baseline for one split configuration."""

    subsets = fit_scaler(
        prepare_split_features(cohort, split_assignments, fp_matrix, cell_feature_map)
    )

    train_rows, x_train = subsets["train"]
    y_train = train_rows[target].to_numpy()

    logger.info(
        "Split %s — train shape: %s, target mean: %.3f, std: %.3f",
        split_name,
        x_train.shape,
        y_train.mean(),
        y_train.std(),
    )

    t0 = time.time()
    fit = fit_mlp_regressor(
        x_train,
        y_train,
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=random_state,
        device=device,
    )
    elapsed = time.time() - t0
    logger.info(
        "MLP fitted in %.1fs — backend: %s, device: %s, epochs: %d, final loss: %.4f",
        elapsed,
        fit.backend,
        fit.device,
        fit.n_iter,
        fit.loss,
    )

    results: list[dict[str, Any]] = []
    for subset in ("validation", "test"):
        rows, features = subsets[subset]
        row = metric_row(
            fit,
            rows,
            features,
            split_name=split_name,
            subset=subset,
            target=target,
        )
        row["fit_seconds"] = elapsed
        row["hidden_layer_sizes"] = json.dumps(list(hidden_layer_sizes))
        row["alpha"] = alpha
        row["learning_rate_init"] = learning_rate_init
        row["batch_size"] = batch_size
        row["max_iter"] = max_iter
        row["early_stopping"] = early_stopping
        results.append(row)

    return results


def run_b3(
    cohort_path: str | Path = "data/processed/cohort_pairs.csv",
    split_dir: str | Path = "data/processed/splits",
    expression_path: str | Path = EXPRESSION_PATH,
    output: str | Path = "results/baselines/b3_metrics.csv",
    summary: str | Path = "data/reports/b3_summary.json",
    *,
    target: str = "ln_ic50",
    n_components: int = 256,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    hidden_layer_sizes: tuple[int, ...] = (512, 128),
    alpha: float = 0.0001,
    learning_rate_init: float = 0.001,
    batch_size: int = 512,
    max_iter: int = 120,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    device: str = "auto",
) -> pd.DataFrame:
    """Run B3 across all requested splits."""

    cohort = pd.read_csv(cohort_path)
    split_dir = Path(split_dir)

    logger.info("Building Morgan fingerprint matrix for %d rows ...", len(cohort))
    fp_matrix = build_fingerprint_matrix(cohort["canonical_smiles"])
    logger.info("Fingerprint matrix shape: %s", fp_matrix.shape)

    all_results: list[dict[str, Any]] = []

    for split_name in split_names:
        logger.info("=== Processing split: %s ===", split_name)
        split_path = split_dir / f"{split_name}.csv"
        assignments = pd.read_csv(split_path)

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

        split_results = run_b3_for_split(
            cohort,
            assignments,
            fp_matrix,
            cell_feature_map,
            split_name=split_name,
            target=target,
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
            device=device,
        )
        all_results.extend(split_results)

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
        "fp_bits": 2048,
        "fp_radius": 2,
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "batch_size": batch_size,
        "max_iter": max_iter,
        "early_stopping": early_stopping,
        "validation_fraction": validation_fraction,
        "device_requested": device,
        "splits": list(split_names),
        "best_by_split_subset": summarize_best(metrics),
    }
    summary_path = Path(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2)

    return metrics


def summarize_best(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize the B3 result for each split/subset."""

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
                "backend": best["backend"],
                "device": best["device"],
                "n_iter": int(best["n_iter"]),
                "loss": float(best["loss"]),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Run B3 MLP neural baseline.")
    parser.add_argument("--cohort", default="data/processed/cohort_pairs.csv")
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--expression", default=EXPRESSION_PATH)
    parser.add_argument("--output", default="results/baselines/b3_metrics.csv")
    parser.add_argument("--summary", default="data/reports/b3_summary.json")
    parser.add_argument("--target", default="ln_ic50")
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[512, 128])
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable internal early stopping.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. auto uses CUDA when PyTorch can see a GPU.",
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
    metrics = run_b3(
        cohort_path=args.cohort,
        split_dir=args.split_dir,
        expression_path=args.expression,
        output=args.output,
        summary=args.summary,
        target=args.target,
        n_components=args.n_components,
        split_names=tuple(args.splits),
        hidden_layer_sizes=tuple(args.hidden_layers),
        alpha=args.alpha,
        learning_rate_init=args.learning_rate,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        early_stopping=not args.no_early_stopping,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
        device=args.device,
    )
    print(f"\nWrote B3 metrics to {args.output}")
    print("=" * 72)
    for _, row in metrics.sort_values(["split_name", "subset", "rmse"]).iterrows():
        print(
            f"  {row['split_name']:>12s} / {row['subset']:>10s} / {row['model']:>8s}: "
            f"RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}  "
            f"Pearson={row['pearson']:.3f}  R²={row['r2']:.3f}  "
            f"(backend={row['backend']}, device={row['device']}, iter={row['n_iter']:.0f})"
        )


if __name__ == "__main__":
    main()
