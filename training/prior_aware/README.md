# Prior-Aware CaMCheX

This path trains a CaMCheX variant that sees the current study plus the nearest
previous study for the same patient.

## How Prior Context Rows Are Built

The prior-aware parquet has one row per current study. It does not create every
pairwise combination of a patient's history.

For a patient with four chronological studies:

```text
study_1
study_2
study_3
study_4
```

the training rows are:

```text
study_1 -> no prior
study_2 -> prior = study_1
study_3 -> prior = study_2
study_4 -> prior = study_3
```

So the meaningful prior pairs are `2<-1`, `3<-2`, and `4<-3`. The first study
still appears as a row, but its prior branch is masked.

## Build The Parquet

Default behavior stores token ids and attention masks. The text backbone remains
trainable during model training.

```bash
python src/prepare/04_build_prior_aware_dataset.py \
  --tokenizer dmis-lab/biobert-v1.1
```

For the CXR-BERT variant:

```bash
python src/prepare/04_build_prior_aware_dataset.py \
  --tokenizer microsoft/BiomedVLP-CXR-BERT-specialized
```

## Tokenized Versus Pre-Embedded

Default tokenized parquet:

```text
CSV text
-> tokenizer during parquet build
-> input_ids + attention_mask stored
-> model loads and runs BioBERT/CXR-BERT during training
```

Optional pre-embedded parquet:

```text
CSV text
-> tokenizer + frozen BioBERT/CXR-BERT during parquet build
-> CLS embeddings stored
-> model does not load BioBERT/CXR-BERT during training
```

Pre-embedding is only appropriate when the text backbone is frozen. If the text
backbone is trainable, embeddings must be recomputed every update, so cached
embeddings would be stale.

## Precompute Text Embeddings

BioBERT:

```bash
python src/prepare/04_build_prior_aware_dataset.py \
  --tokenizer dmis-lab/biobert-v1.1 \
  --precompute-text-embeddings \
  --out-prefix prior_aware_biobert_embedded_
```

CXR-BERT:

```bash
python src/prepare/04_build_prior_aware_dataset.py \
  --tokenizer microsoft/BiomedVLP-CXR-BERT-specialized \
  --precompute-text-embeddings \
  --out-prefix prior_aware_cxrbert_embedded_
```

This writes columns named:

```text
clin_embedding
obs_embedding
prior_clin_embedding
prior_obs_embedding
```

instead of:

```text
clin_input_ids
clin_attn_mask
obs_input_ids
obs_attn_mask
prior_clin_input_ids
prior_clin_attn_mask
prior_obs_input_ids
prior_obs_attn_mask
```

## Train With Precomputed Embeddings

Point the config or CLI paths at the embedded parquet files and enable the
model flag:

```bash
python training/prior_aware/prior_aware_train.py \
  --train-df-path data/data-camchex/prior_aware_biobert_embedded_train.parquet \
  --val-df-path data/data-camchex/prior_aware_biobert_embedded_development.parquet \
  --test-df-path data/data-camchex/prior_aware_biobert_embedded_test.parquet \
  --use-precomputed-text-embeddings
```

CXR-BERT version:

```bash
python training/prior_aware_cxrbert/prior_aware_train.py \
  --train-df-path data/data-camchex/prior_aware_cxrbert_embedded_train.parquet \
  --val-df-path data/data-camchex/prior_aware_cxrbert_embedded_development.parquet \
  --test-df-path data/data-camchex/prior_aware_cxrbert_embedded_test.parquet \
  --use-precomputed-text-embeddings
```

`--use-precomputed-text-embeddings` also implies frozen text behavior and the
model skips constructing the text encoder.

## Freeze Without Pre-Embedding

This mode still loads BioBERT/CXR-BERT, but keeps it frozen and runs it under
`torch.no_grad()`:

```bash
python training/prior_aware/prior_aware_train.py --freeze-text-encoder
```

This is useful for quick comparison, but if you intend to keep the text encoder
frozen for a real run, precomputed embeddings are usually faster and lighter.
