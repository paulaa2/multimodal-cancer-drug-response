# Project Plan

This plan translates the design document into implementation phases that can be
worked through incrementally.

## F0 - Biology and Data Audit

Goal: create a clean, frozen cohort table before modelling.

Outputs:

- source data summary with exact file names, URLs, versions, and checksums
- GDSC response table standardized to one target, initially `ln(IC50)`
- cell-line mapping table from GDSC names/IDs to DepMap IDs
- drug mapping table from GDSC compound IDs/names to canonical SMILES
- overlap report with included/excluded cell lines, drugs, and response pairs

Key rule: no model training starts until the cohort and mappings are explicit.

After the audit and SMILES lookup, build the first model-ready cohort:

```powershell
python -m mcdrp.data.build_cohort
```

This writes `data/processed/cohort_pairs.csv`, a lightweight table of valid
GDSC drug-cell-line pairs with DepMap IDs and canonical SMILES. The expression
matrix remains separate and is linked by `depmap_id`.

## F1 - Classical Baselines

Goal: measure how far simple models go.

Outputs:

- leakage-safe random-pair and cold-drug splits
- Morgan fingerprint features
- selected-gene or PCA cell-line features
- Ridge / Elastic Net baseline
- XGBoost baseline
- MLP neural baseline with the same engineered features
- first metrics table

Run the first neural baseline on one split:

```powershell
python -m mcdrp.models.baseline_b3 --splits random_pair --device cuda
```

B3 keeps the same input representation as B1/B2 and changes only the predictor
to a feed-forward MLP. If this does not beat XGBoost, the next model needs a
better representation, not just a deeper tabular network.
B2 XGBoost and B3 MLP support `--device auto|cuda|cpu`; B1 Ridge/Elastic Net is
CPU-based.

After running any baseline, create the consolidated baseline comparison:

```powershell
python -m mcdrp.results.compare_baselines
```

This writes `results/baselines/baseline_comparison.csv`,
`results/baselines/baseline_best_by_split.csv`, and
`data/reports/baseline_comparison_summary.json`.

Optional tuning scripts are available when the baseline comparison needs stronger
validation-selected hyperparameters:

```powershell
python -m mcdrp.models.tune_b1 --splits random_pair
python -m mcdrp.models.tune_b2 --splits random_pair --device cuda
python -m mcdrp.models.tune_b3 --splits random_pair --device cuda
python -m mcdrp.models.tune_b4 --splits random_pair --device cuda
python -m mcdrp.models.tune_b5 --splits random_pair --device cuda
```

`tune_b4` and `tune_b5` search a four-point grid of learning rate × dropout,
keep architecture at the published defaults, and score test only for the
validation-selected candidate. Full-split configs include `cold_tissue`.
B6 ablations overlay the validation-selected B2 XGBoost hyperparameters so the
ablation measures modalities rather than a weaker booster.

Run one split first because tuning repeats model fitting several times. The same
comparison command includes standard metrics and tuning outputs when they exist.
For tuning CSVs, it keeps only the validation-selected candidates:

```powershell
python -m mcdrp.results.compare_baselines
```

Before starting GNN work, run the baseline-stage quality gate:

```powershell
python -m mcdrp.results.validate_baseline_stage
```

This writes `data/reports/baseline_stage_validation.json` and checks cohort
schema, split leakage, metric schemas, tuning selection rows, and comparison
output consistency.

Before modelling, generate the initial split assignment files:

```powershell
python -m mcdrp.splits.make_splits
```

This writes one CSV per split under `data/processed/splits/` and a summary at
`data/reports/splits_summary.json`. The cold-cell split checks that held-out
cell lines do not appear in train. The cold-drug split checks that held-out drugs
do not appear in train.

## F2 - Multimodal GNN

Goal: train a reproducible neural model that combines transcriptomics and drug
graphs.

Outputs:

- RDKit molecular graph conversion
- cell-line MLP encoder
- GIN/GINE drug encoder
- concatenation fusion head
- training loop with early stopping
- matched comparison against fingerprint MLP baseline

Initial implementation:

```powershell
python -m mcdrp.models.gnn_b4 --splits random_pair --device cuda
python -m mcdrp.results.compare_baselines
```

The first version uses a lightweight pure-PyTorch graph encoder instead of
PyTorch Geometric. This keeps the dependency surface small while validating that
drug graphs, cell expression features, batching, training, and comparison are
wired correctly.

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b4_gnn_full.json --dry-run
python -m mcdrp.models.tune_b4 --splits random_pair --device cuda
```

The current B4 improvement keeps that dependency-light design but strengthens
the encoder with residual graph-convolution blocks, `LayerNorm`, mean+max graph
pooling, train-only target scaling, and gradient clipping.

Main hybrid model:

```powershell
python -m mcdrp.models.gnn_b5 --splits random_pair --device cuda
python -m mcdrp.models.tune_b5 --splits random_pair --device cuda
python -m mcdrp.results.compare_baselines
```

B5 is intentionally heavier than B4. It combines a deeper residual graph
encoder, attention pooling, Morgan fingerprints, expression features,
multiplicative drug-cell interactions, AdamW, learning-rate scheduling, target
scaling, and gradient clipping. This is the model to use when the project needs
a more complete multimodal architecture that is meaningfully slower to train.

## F3 - Cold-Start Evaluation

Goal: make the generalization story scientifically meaningful.

Outputs:

- random-pair split
- cold-cell split
- cold-tissue split
- cold-drug split
- scaffold split
- cold-both split
- table reporting metrics and unique drugs/cell lines/pairs per split
- plot showing random-to-cold performance degradation

`cold_tissue` holds out whole DepMap Oncotree lineages, so held-out cell lines
share no tissue of origin with training. It is the leave-tissue-out setting
relevant to drug repurposing, and it is stricter than `cold_cell`. Full
experiment and tuning configs now include it; the remaining gap is generating
the trained-model numbers.

The scaffold split is implemented as `cold_scaffold` using Bemis-Murcko
scaffolds derived from canonical SMILES. This is the key chemical
generalization benchmark because it holds out structural families rather than
only individual drug IDs.

The `cold_both` split uses only train/train, validation/validation, and
test/test cell-drug blocks. Mixed blocks are labelled `unused` and ignored by
training scripts; this avoids leaking either cell lines or drugs between train
and evaluation subsets.

Experiment configs live under `configs/experiments/` and can be launched with:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b5_hybrid_full.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/b5_hybrid_full.json
```

After model metrics exist, build the cold-start report:

```powershell
python -m mcdrp.results.compare_baselines
python -m mcdrp.results.cold_start_report --subset test
```

This writes split composition, best models, and random-to-cold RMSE drop tables
under `results/reports/`.

## F4 - Pathways and Ablations

Goal: test whether pathway-level representations improve robustness and
interpretability.

Outputs:

- pathway activity matrix
- comparison of selected genes vs PCA vs pathway scores
- ablation table under identical splits
- B6 modality ablation metrics and summary

Initial ablation implementation:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b6_ablation_random_pair.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/b6_ablation_random_pair.json
```

B6 compares `expression_pca`, `pathways`, `morgan`,
`morgan_expression_pca`, `morgan_pathways`, and
`morgan_expression_pca_pathways` under the same XGBoost family. When
`results/baselines/b2_tuning_metrics.csv` exists, B6 overlays the
validation-selected B2 hyperparameters per split so the ablation measures
modalities rather than a weaker booster. Use `--gene-sets path/to/file.gmt` to
swap the built-in compact cancer pathway panel for MSigDB Hallmark, Reactome,
or another curated gene-set collection.

After B6 metrics exist, summarize the ablation study:

```powershell
python -m mcdrp.results.ablation_report --subset test
```

To rebuild both F3 and F4 reports without retraining:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/reports_f3_f4.json
```

## F5 - Pretrained Molecular Representations

Goal: compare handcrafted molecular representations against frozen embeddings
from pretrained molecule encoders.

Implemented entry point:

```powershell
python -m mcdrp.models.pretrained_b7 --drug-embeddings data/external/drug_embeddings.csv --splits random_pair --device cuda
```

Config workflow:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b7_pretrained_random_pair.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/b7_pretrained_random_pair.json
```

B7 variants:

- `pretrained_drug`
- `pretrained_drug_expression_pca`
- `pretrained_drug_pathways`
- `pretrained_drug_expression_pca_pathways`

The embedding CSV format is documented in `docs/external_embeddings.md`.

Tuned B7 entry point:

```powershell
python -m mcdrp.experiments.run_experiment configs/experiments/b7_tuning_random_pair.json --dry-run
python -m mcdrp.experiments.run_experiment configs/experiments/b7_tuning_random_pair.json
```

The tuning stage keeps the same B7 feature variants but searches a small
XGBoost grid and keeps validation-selected test rows, matching the earlier
B1-B3 tuning protocol.

## F6 - Multi-Omics Extension

Goal: add additional cell-line modalities after B7 tuning has established
whether pretrained drug embeddings are useful.

Candidate modalities:

- copy-number alteration features
- mutation features for cancer driver genes
- methylation or proteomics, if a clean matched source is available

The key rule stays the same: every scaler, reducer, selector, and imputer must
be fit inside each split using training cell lines only.

## F7 - Interpretation and Polish

Goal: turn the implementation into a portfolio-ready scientific artifact.

Outputs:

- 2-3 biological case studies
- gene/pathway attribution analysis
- optional drug substructure explanations
- architecture and data-integration figures
- final README results section
- short report or GitHub release summary
