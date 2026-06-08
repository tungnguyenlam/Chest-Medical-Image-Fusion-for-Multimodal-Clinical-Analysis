# Training & Eval CLI Flags

Every `training/<model>/<model>_{train,eval}.py` entry point shares a common flag set
(defined in [`add_common_args`](common.py)) and then adds a handful of **model-specific**
flags in its own `parse_args`. This file is the map of what exists and where it applies.

- **General** flags work on every script (some are train-only or eval-only — noted below).
- **Model-specific** flags exist only on the scripts listed in the [matrix](#model-specific-flags).
- CLI flags override the matching key in the model's `config.yaml`; anything left unset
  falls back to the config, then to the built-in default.

Run `python training/<model>/<model>_train.py --help` to see the live, grouped list.

---

## General flags (all scripts)

Grouped exactly as in `add_common_args`. "train-only" flags are ignored by `*_eval.py`.

### run identity & I/O
| Flag | Use |
|------|-----|
| `--config PATH` | Config YAML. Default `training/<model>/config.yaml`. |
| `--train-df-path` / `--val-df-path` / `--test-df-path` | Override the dataframe paths from config. |
| `--output-dir DIR` | Run root. Default `output/<model>/runs`. |
| `--run-name NAME` | Run folder suffix. Default `baseline`. |
| `--run-id ID` | Force a run id (else timestamp). Also selects the run for `--quick-continue`. |

### checkpointing & resume
| Flag | Use |
|------|-----|
| `--checkpoint-path PATH` | Eval: weights to load. Train: **weights-only** warm-start (fresh optimizer/scheduler/epoch). |
| `--resume-from PATH` | *Train only.* Full resume (model + optimizer + scheduler + epoch + step + early-stop state); run dir inferred from the path. |
| `--quick-continue` | *Train only.* Resume the latest checkpoint of the latest run under `--output-dir`. |
| `--seed INT` | Seed python/numpy/torch RNGs. |

### data & batching
| Flag | Use |
|------|-----|
| `--batch-size INT` | Train batch size. |
| `--val-batch-size INT` | Val/eval batch size (forward-only). Default 2× train; one-time OOM fallback to train size. |
| `--num-workers INT` | Train DataLoader workers. |
| `--val-num-workers INT` | Val/eval workers. **Default 0** so mid-epoch val doesn't fork a second pool on top of persistent train workers (host-RAM/OOM guard). |
| `--prefetch-factor INT` | DataLoader prefetch (needs `num_workers > 0`). |
| `--image-size INT` | Square resize fed to the backbone. |
| `--channel-mode MODE` | 3-channel CXR build (raw + CLAHE + third channel), or `none` for legacy ImageNet RGB. |
| `--cpu-fraction FLOAT` | Fraction of cores for the channel-cache precompute. |
| `--skip-precompute` | Skip the upfront channel-cache scan/build (channels build lazily on first access). |

### optimization
| Flag | Use |
|------|-----|
| `--loss NAME [NAME ...]` | Loss(es), overriding `model.loss` / default ASL. Several names → weighted sum. |
| `--loss-weights W [W ...]` | Positional weights for a multi-loss `--loss`. |
| `--lr FLOAT` | Learning rate. |
| `--weight-decay FLOAT` | Weight decay. |
| `--warmup-ratio FLOAT` | Warmup as a fraction of steps/epoch (default 0.05). |
| `--max-epochs INT` | Epoch cap. |
| `--accumulate-grad-batches INT` | Gradient accumulation (effective batch = batch × accumulate). |
| `--grad-clip FLOAT` | Max grad norm; `0`/negative disables. Default 1.0. |

### EMA (weight averaging) — *train only*
| Flag | Use |
|------|-----|
| `--ema` / `--no-ema` | Maintain an EMA of weights and evaluate/save it. Best with `schedule=single_cosine`. Do **not** `--resume-from` an EMA checkpoint. |
| `--ema-decay FLOAT` | EMA decay (default 0.999; higher = smoother/slower). |

### validation, early stopping & logging — *train only*
| Flag | Use |
|------|-----|
| `--val-check-interval FLOAT` | Lightning-style val cadence. |
| `--log-every-n-steps INT` | Rows written to `train_steps.csv`. |
| `--early-stop-monitor NAME` | Metric to monitor (default `val_ap`). |
| `--early-stop-mode {min,max}` | Direction (default `max`). |
| `--early-stop-patience INT` | Non-improving epochs before stop (default 3; `0` disables). |
| `--early-stop-min-delta FLOAT` | Min change counting as improvement. |
| `--quick-val-every-steps INT` | Partial val every N optimizer steps → `val_quick.csv` (doesn't affect best-checkpoint tracking). |
| `--quick-val-frac FLOAT` | Fraction of val batches for quick val (default 0.1). |
| `--full-val-fracs [F ...]` | Epoch fractions for mid-epoch full val (default 0.5; pass empty to disable). |
| `--quick-val-fracs [F ...]` | Epoch fractions for quick val (default 0.25 0.75; pass empty to disable). |

### hardware, precision & compile
| Flag | Use |
|------|-----|
| `--accelerator` / `--devices` / `--precision` | Device & precision selection. |
| `--compile-model` | *Train.* `torch.compile` (automatic dynamic) on compile-safe submodules; fusion stays eager; failures fall back to eager. RAM-heavy during compile. |

### backbone
| Flag | Use |
|------|-----|
| `--backbone-name NAME` | Override the timm backbone. |
| `--no-pretrained` | Don't load pretrained backbone weights. |

### debug
| Flag | Use |
|------|-----|
| `--fast-dev-run` | Tiny smoke run. |

### Grad-CAM panels — *train only*
| Flag | Use |
|------|-----|
| `--gradcam-epochs` | `all` (default) / `none` / comma list of epochs. Only models defining `gradcam_runner_module` emit panels. |
| `--gradcam-device` | Device for the Grad-CAM subprocess (default cpu). |

### eval-only
| Flag | Use |
|------|-----|
| `--skip-report-ablation` | *Eval.* Skip the second "current clinical indication blanked" pass (the leakage probe). |

---

## Model-specific flags

Added by each script's own `parse_args`, not by `add_common_args`.

| Flag | Where | Use |
|------|-------|-----|
| `--frontal-pretrained-path` / `--lateral-pretrained-path` | all multi-view image models¹ | Stage-1 timm backbone `state_dict`s to warm-start the frontal/lateral encoders. |
| `--text-model NAME` | all text-fusion models¹ | Override `model.text_model` (e.g. BioBERT vs CXR-BERT). |
| `--freeze-text-encoder` | text-fusion models with a cache² | Freeze the text encoder (only meaningful when token ids are encoded at train time). |
| `--text-embedding-cache-dir DIR` | text-fusion models with a cache² | Override the shared frozen text-embedding cache root. |
| `--use-precomputed-text-embeddings` | prior-aware family³ | Use the shared frozen embedding cache instead of loading the text encoder. |
| `--text-embeddings-gpu-resident` | **`prior_aware_v3nano` only** | Opt-in. Keep the precomputed embeddings as one frozen table on-device and emit row indices instead of per-sample float vectors; also precomputes indices and drops raw-text columns to keep per-worker RAM flat. Requires `--use-precomputed-text-embeddings`. |
| `--view-position {AP,PA,LATERAL,...}` | **`singleview` only** | Which view the single-view model trains/evals on. |
| `--predictions-path` / `--metrics-path` | most `*_eval.py`⁴ | Where eval writes predictions / metrics. |
| `--output-csv PATH` | prior-aware family `*_eval.py` | Eval predictions CSV path. |

¹ `camchex`, `camchex_cxrbert`, `camchex_v2nano_vitals`, `camchex_v2nano_vitals_stable`,
  `camchex_v3nano`, `prior_aware`, `prior_aware_cxrbert`, `prior_aware_v2nano`,
  `prior_aware_v3nano`. (Not `singleview`.)
² Above **minus** base `camchex` / `camchex_cxrbert` (whose train scripts only take
  `--frontal/--lateral-pretrained-path` and `--text-model`).
³ `prior_aware`, `prior_aware_cxrbert`, `prior_aware_v2nano`, `prior_aware_v3nano`.
⁴ `camchex*`, `camchex_v3nano`, vitals variants, `singleview` eval scripts.

---

## Notes that bite

- **`--compile-model` is RAM-heavy** during Inductor compilation and gives little on
  small GPUs (e.g. "Not enough SMs to use max_autotune"). Drop it first if you hit a
  host-RAM OOM (`Killed` while VRAM is idle).
- **`--use-precomputed-text-embeddings`** needs the embedding cache built once up front
  (it runs at startup). The text encoder is then not loaded, so the run can't be fine-tuned
  on text — it's frozen by construction.
- **`--text-embeddings-gpu-resident`** is the host-RAM mitigation for the prior-aware v3nano
  path; see [`prior_aware_v3nano/README.md`](prior_aware_v3nano/README.md#low-host-ram-mode---text-embeddings-gpu-resident).
- **`--resume-from` vs `--checkpoint-path`**: resume = continue the same run (full state);
  checkpoint-path = start a fresh run from those weights only.
