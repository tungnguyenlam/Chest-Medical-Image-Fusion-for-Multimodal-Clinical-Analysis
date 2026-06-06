# CaMCheX v3 Nano

This variant keeps the `camchex_v2nano_vitals` image backbone, CXR-BERT clinical
encoder, numeric-vitals dataset, optimizer/training defaults, and ML-Decoder
head, but removes the repeated non-image fusion blocks.

## Main Difference

With the default 512 px input, v2 used:

```text
image views:       4 views * 8 * 8 = 256 tokens
clinical text:     1 CLS  * 8 * 8 =  64 repeated CLS tokens
numeric vitals:    1 MLP  * 8 * 8 =  64 learned vital tokens
total max:                            384 tokens
```

v3 uses:

```text
image views:       4 views * 8 * 8 = 256 tokens
clinical text:     1 CLS token    =   1 token
numeric vitals:    1 MLP token    =   1 token
total max:                            258 tokens
token width:                           768
output logits:                          26
```

The clinical path still runs CXR-BERT or reads a cached CLS embedding, so the
clinical feature entering fusion is still `B x 768`. The difference is that v3
keeps it as `B x 1 x 768` instead of expanding it to `B x 64 x 768`.

The vitals path reuses `VitalsTokenProjector`, but configures it with
`grid_size=1`, so the MLP outputs `B x 1 x 768` instead of `B x 64 x 768`.

## Files

| Concern | File |
|---|---|
| Model assembly | `src/model/CaMCheXV3NanoModel.py` |
| Numeric vitals dataset | `src/dataloader/CaMCheXVitalsDataset.py` |
| Train script | `training/camchex_v3nano/camchex_v3nano_train.py` |
| Eval script | `training/camchex_v3nano/camchex_v3nano_eval.py` |
| Config | `training/camchex_v3nano/config.yaml` |

## Run

```bash
python training/camchex_v3nano/camchex_v3nano_train.py \
  --config training/camchex_v3nano/config.yaml
```

```bash
python training/camchex_v3nano/camchex_v3nano_eval.py \
  --config training/camchex_v3nano/config.yaml \
  --checkpoint-path output/camchex_v3nano/runs/<run>/checkpoints/best.pt
```

Optional frozen text embedding cache works the same way as v2:

```yaml
data:
  datamodule_cfg:
    use_text_embedding_cache: true
```

Grad-CAM uses the shared `src.interpret.run_gradcam` runner. The v3 config sets
`model.arch: camchex_v3nano` so the runner instantiates `CaMCheXV3NanoModel`
instead of the v2 model when loading checkpoints.
