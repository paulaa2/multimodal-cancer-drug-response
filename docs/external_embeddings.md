# External Drug Embeddings

B7 does not download pretrained models by itself. It expects embeddings that
were generated outside the training script and saved as a CSV. This keeps the
benchmark reproducible: the modelling code consumes a frozen representation
instead of silently changing when a remote model changes.

The repository includes a generator for this file:

```powershell
pip install -e ".[embeddings]"
python -m mcdrp.features.create_drug_embeddings `
  --cohort data/processed/cohort_pairs.csv `
  --output data/external/drug_embeddings.csv `
  --model DeepChem/ChemBERTa-77M-MLM `
  --revision main `
  --batch-size 32 `
  --max-length 256 `
  --pooling mean `
  --device cuda
```

For a quick smoke test that only encodes a few drugs:

```powershell
python -m mcdrp.features.create_drug_embeddings `
  --output data/external/drug_embeddings_smoke.csv `
  --limit 5 `
  --device cuda
```

## Expected Format

Minimum CSV:

```text
drug_id,emb_0,emb_1,emb_2
1003,0.18,-0.44,0.07
1021,-0.02,0.31,0.55
```

Recommended CSV:

```text
drug_id,canonical_smiles,encoder_name,encoder_revision,pooling,emb_0,emb_1,emb_2
1003,CCOC(=O)...,DeepChem/ChemBERTa-77M-MLM,model-release-id,mean,0.18,-0.44,0.07
```

Use `--embedding-prefix emb_` when the CSV contains metadata columns in addition
to embedding dimensions.

## Key Columns

By default, B7 matches `cohort_pairs.csv` to the embedding table with `drug_id`.
If your embedding file is keyed by SMILES instead, use:

```powershell
python -m mcdrp.models.pretrained_b7 `
  --drug-embeddings data/external/drug_embeddings.csv `
  --embedding-key-column canonical_smiles `
  --cohort-key-column canonical_smiles `
  --embedding-prefix emb_
```

## Suggested Encoders

- ChemBERTa-style SMILES transformer embeddings
- MolFormer-style molecular language model embeddings
- Uni-Mol-style 3D molecular embeddings when conformers are available

Store the generated CSV under `data/external/`. That directory is ignored by git,
so keep a note of the exact encoder, checkpoint, pooling strategy, and command
used to create the file.

For final reported experiments, replace `--revision main` with a Hugging Face
commit hash so the embeddings can be regenerated exactly.
