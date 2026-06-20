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

Grouped exactly as in `add_common_args`. Flags marked **train-only** below are not
registered on `*_eval.py` at all — `add_common_args(..., mode="eval")` drops them, so
they don't appear in `eval --help` (previously they were shown but silently ignored).
The eval parser also adds the eval-only `--skip-report-ablation` and `--skip-task2-gold`. Eval loaders pre-warm
the image channel cache for the test split up front (the same prebuild training runs for
train+val), so the first eval over a fresh split pays the channel-build cost upfront
instead of stalling mid-loop; `--skip-precompute` bypasses it.

### run identity & I/O
| Flag | Use |
|------|-----|
| `--config PATH` | Config YAML. **Eval:** if omitted and `--checkpoint-path` is given, the config is auto-resolved from the checkpoint's run (`<run_dir>/config.resolved.json`) — the exact config it was trained with — so you usually just pass the checkpoint. An explicit `--config` always wins; other flags still override individual keys. **Train / no checkpoint:** defaults to `training/<model>/config.yaml`. |
| `--train-df-path` / `--val-df-path` / `--test-df-path` | Override the dataframe paths from config. |
| `--output-dir DIR` | Run root. Default `output/<model>/runs`. |
| `--run-name NAME` | Run folder suffix. Default `baseline`. |
| `--run-id ID` | Force a run id (else timestamp). Also selects the run for `--quick-continue`. |

### checkpointing & resume
| Flag | Use |
|------|-----|
| `--checkpoint-path PATH` | Eval: weights to load (and, unless `--config` is passed, the config is auto-resolved from this checkpoint's run dir). Train: **weights-only** warm-start (fresh optimizer/scheduler/epoch). |
| `--resume-from PATH` | *Train only.* Full resume (model + optimizer + scheduler + epoch + step + early-stop state); run dir inferred from the path. Refuses an **EMA** checkpoint (its weights are the EMA snapshot but the optimizer state is for the raw weights) — use `--checkpoint-path` for a weights-only warm-start instead. |
| `--quick-continue` | *Train only.* Resume the **most recently created** run under `--output-dir` (its latest checkpoint). Pass `--run-id` to target a specific run instead. Resumes under the hood, so it inherits the EMA-checkpoint refusal above (EMA runs can't be auto-continued). |
| `--seed INT` | Seed python/numpy/torch RNGs. |

### data & batching
| Flag | Use |
|------|-----|
| `--batch-size INT` | Train batch size. |
| `--val-batch-size INT` | Val/eval batch size (forward-only). Default 2× train; one-time OOM fallback to train size. |
| `--num-workers INT` | Train DataLoader workers. |
| `--val-num-workers INT` | Val/eval workers. **Default 0** so mid-epoch val doesn't fork a second pool on top of persistent train workers (host-RAM/OOM guard). |
| `--prefetch-factor INT` | DataLoader prefetch (needs `num_workers > 0`). |
| `--malloc-arena-max INT` | Cap glibc malloc arenas (host-RAM/RSS control under `num_workers > 0`; each fork worker otherwise grows its own arena set up to ~8×ncpu, fragmenting RSS). **Default 2**; `0` leaves the glibc default. Applied via `mallopt` + `MALLOC_ARENA_MAX` before workers fork; no effect off glibc. |
| `--image-size INT` | Square resize fed to the backbone. |
| `--third-channel-mode {clahe,histeq,lbp,none}` | Third channel of the CXR build (ch0=raw, ch1=mild CLAHE are pinned). `clahe`=strong CLAHE 4.0/16x16, `histeq`=global hist-eq, `lbp`=uniform LBP, `none`=legacy ImageNet RGB. Resolved composition printed at init under `[channels]`. |
| `--cpu-fraction FLOAT` | Fraction of cores for the channel-cache precompute. |
| `--skip-precompute` | Skip the upfront channel-cache scan/build (channels build lazily on first access). |

### optimization
| Flag | Use |
|------|-----|
| `--loss NAME [NAME ...]` | Loss(es), overriding `model.loss` / default ASL. Several names → weighted sum. |
| `--loss-weights W [W ...]` | Positional weights for a multi-loss `--loss`. |
| `--label-smoothing EPS` | Positive-only (asymmetric) label smoothing for ASL: positive targets `1 → 1-EPS` (negatives stay 0). Overrides `model.loss_init_args.pos_smoothing`. Default 0.0 (off). Noise/calibration knob, not a long-tail fix; try ~0.05. |
| `--lr FLOAT` | Base learning rate (the freshly-initialised fusion + decoder train at this). |
| `--weight-decay FLOAT` | Weight decay. |
| `--backbone-lr-mult FLOAT` | Discriminative LR: train the pretrained image backbone (params under `image_encoder.`) at this multiple of `--lr`. **Default 0.3** (applied to every model); `1.0` disables (uniform LR). Overridden by `optimizer_init_args.param_group_lrs` if a config sets it. See [docs/discriminative_lr.md](../docs/discriminative_lr.md). |
| `--warmup-ratio FLOAT` | Warmup as a fraction of steps/epoch (default 0.05). |
| `--max-epochs INT` | Epoch cap. |
| `--accumulate-grad-batches INT` | Gradient accumulation (effective batch = batch × accumulate). |
| `--grad-clip FLOAT` | Max grad norm; `0`/negative disables. Default 1.0. |

### EMA (weight averaging) — *train only*
| Flag | Use |
|------|-----|
| `--ema` / `--no-ema` | Maintain an EMA of weights and evaluate/save it. Best with `schedule=single_cosine`. EMA checkpoints are eval-ready but **cannot be full-resumed** (`--resume-from` / `--quick-continue` refuse them); warm-start from them with `--checkpoint-path` instead. |
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
| `--full-val-fracs [F ...]` | Epoch fractions for mid-epoch full val (default: none — full val runs only at epoch end; pass fractions to enable). |
| `--quick-val-fracs [F ...]` | Epoch fractions for quick val (default 0.25 0.5 0.75; pass empty to disable). |

### hardware, precision & compile
| Flag | Use |
|------|-----|
| `--accelerator` / `--devices` / `--precision` | Device & precision selection. |
| `--compile-model` | *Train.* `torch.compile` (automatic dynamic) on compile-safe submodules; fusion stays eager; failures fall back to eager. RAM-heavy during compile. |
| `--channels-last` | *Train.* Run the conv backbone(s) in `torch.channels_last` (NHWC). Layout-only (numerics identical); enables cuDNN's native NHWC Tensor-Core conv kernels, typically a 10–30% throughput win on ConvNeXtV2. Pure GPU-side change, no host-RAM cost. |

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
| `--gradcam-device` | Device for the Grad-CAM subprocess. Default is the training device. |

### eval-only
| Flag | Use |
|------|-----|
| `--skip-report-ablation` | *Eval.* Skip the second "current clinical indication blanked" pass (the leakage probe). |
| `--skip-task2-gold` | *Eval.* Skip the extra CXR-LT 2024 task2 (gold) pass. By default a model whose label set is a superset of the 26 task2 labels (i.e. trained on 2024 task1/all) is also scored on the task2 gold test set: its outputs are sliced to the 26 task2 columns and written to `*_task2_gold.{csv,json}`. Needs the task2 gold file next to the eval file (`task1` → `task2` in the name); silently skipped if absent. No-op for CXR-LT 2023 models. |

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
| `--text-embeddings-gpu-resident` | **`prior_aware_v3nano`, `prior_aware_v4nano`** | Opt-in. Keep the precomputed embeddings as one frozen table on-device and emit row indices instead of per-sample float vectors; also precomputes indices and drops raw-text columns to keep per-worker RAM flat. Requires `--use-precomputed-text-embeddings`. |
| `--uint8-image-pipeline` | **`camchex_v2nano_vitals_stable`, `prior_aware_v3nano`, `prior_aware_v4nano`** | Opt-in. Ship images as uint8 [0,255] and dequantize + normalize on-device in the model instead of CPU float32 — ~4× smaller per-batch host buffer, pinned staging, and H2D copy. Requires a channel mode. Train-time augs then run on uint8, shifting value-scale aug numerics (noise/brightness), so validate with a short ablation before adopting. Eval stays on the (numerically identical) float path. |
| `--view-position {AP,PA,LATERAL,...}` | **`singleview` only** | Which view the single-view model trains/evals on. |
| `--predictions-path` / `--metrics-path` | most `*_eval.py`⁴ | Where eval writes predictions / metrics. |
| `--output-csv PATH` | prior-aware family `*_eval.py` | Eval predictions CSV path. |

¹ `camchex`, `camchex_cxrbert`, `camchex_v2nano_vitals`, `camchex_v2nano_vitals_stable`,
  `camchex_v3nano`, `prior_aware_v4nano`, `prior_aware`, `prior_aware_cxrbert`,
  `prior_aware_v2nano`, `prior_aware_v3nano`. (Not `singleview`.)
² Above **minus** base `camchex` / `camchex_cxrbert` (whose train scripts only take
  `--frontal/--lateral-pretrained-path` and `--text-model`).
³ `prior_aware`, `prior_aware_cxrbert`, `prior_aware_v2nano`, `prior_aware_v3nano`,
  `prior_aware_v4nano`.
⁴ `camchex*`, `camchex_v3nano`, vitals variants, `singleview` eval scripts.

---

## Notes that bite

- **`--compile-model` is RAM-heavy** during Inductor compilation and gives little on
  small GPUs (e.g. "Not enough SMs to use max_autotune"). Drop it first if you hit a
  host-RAM OOM (`Killed` while VRAM is idle). To curb the compile-time spike, Inductor's
  parallel compile workers are **capped to 1 by default** (`TORCHINDUCTOR_COMPILE_THREADS=1`,
  set automatically when `--compile-model` is on) — serial compilation is slower for the
  first epoch but avoids the fork-worker RAM spike. Export the var (e.g. `=4`) to override
  on a roomy box.
- **`--channels-last` is the cheap throughput lever** for the conv backbone — it changes
  only the GPU memory layout (NHWC), costs no host RAM, and is independent of
  `--compile-model`. Safe to keep on even when you've dropped `--compile-model` to dodge a
  host-RAM OOM. Numerics are identical, so A/B it purely on GPU throughput.
- **`--use-precomputed-text-embeddings`** needs the embedding cache built once up front
  (it runs at startup). The text encoder is then not loaded, so the run can't be fine-tuned
  on text — it's frozen by construction.
- **`--text-embeddings-gpu-resident`** is the host-RAM mitigation for the prior-aware v3nano
  path; see [`prior_aware_v3nano/README.md`](prior_aware_v3nano/README.md#low-host-ram-mode---text-embeddings-gpu-resident).
- **Unused raw-text columns are dropped automatically.** The prior-aware loaders prune any
  text stream the model never consumes (e.g. the `obs` streams when vitals are fed numerically)
  from the in-RAM parquet right after the embedding cache is attached — no flag needed. This
  shrinks steady-state host RAM and the copy-on-write duplication into fork workers.
- **DataLoader worker `Bus error` / "out of shared memory"** is `/dev/shm` exhaustion, *not*
  host-RAM OOM — distinct from a `Killed`. Common on WSL (small `/dev/shm` tmpfs) once
  `num_workers > 0` ships multi-view image batches between workers and the main process. The
  loaders now switch the worker IPC sharing strategy to `file_system` (disk-backed temp files)
  automatically when `num_workers > 0`, which sidesteps the `/dev/shm` size limit. Override with
  `CAMCHEX_MP_SHARING_STRATEGY=file_descriptor`. The complementary host-side fix is to enlarge
  `/dev/shm` (in `.wslconfig`/`/etc/wsl.conf`, then `wsl --shutdown`).
- **glibc arenas are capped to 2 by default** (`--malloc-arena-max`). With `num_workers > 0`
  this is often the single largest host-RAM reduction; set `--malloc-arena-max 0` to restore
  the glibc default if you need to compare.
- **`--uint8-image-pipeline`** is the largest structural per-batch host-RAM cut (images ride as
  uint8, ~4× smaller, normalized on-device). It changes train-time augmentation numerics
  (augs run on uint8 [0,255] rather than float [0,1]), so it's opt-in and wants an ablation.
  Eval/predict stay on the float path, which is numerically identical to the on-device normalize
  (verified to float32 epsilon), so a uint8-trained checkpoint evaluates correctly without the flag.
- **`--resume-from` vs `--checkpoint-path`**: resume = continue the same run (full state);
  checkpoint-path = start a fresh run from those weights only.
