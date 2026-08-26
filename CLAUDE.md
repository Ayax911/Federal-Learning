# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FedMammoBench: a federated learning framework for binary mammography classification (benign/malignant), built on Flower + PyTorch + Ray. `README.md` is stale on the specifics (see "Documentation map" below) but that one-line description of the project still holds.

## Commands

```bash
# Install (editable + dev tools)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_config_modules.py -v

# Linting / formatting
ruff check src/ tests/
ruff format src/ tests/
black src/ tests/
mypy src/

# Run a federated simulation
fedmammobench-federated --config configs/<exp>/server.yaml

# Run a centralized baseline
fedmammobench-centralized --config configs/<exp>/centralized.yaml

# Post-hoc evaluation on a checkpoint
fedmammobench-evaluate --config configs/<exp>/server.yaml --checkpoint runs/<name>/weights/global_model.pt

# Multi-device gRPC mode (manual)
python scripts/run_server.py --config configs/<exp>/server.yaml   # on the aggregation server
python scripts/run_client.py --config configs/<exp>/client.yaml \
  --server 192.168.1.10:8080 --client-id 0 \
  --manifest manifests/node0_manifest.csv --data-dir data/       # on each node

# Resume a crashed run (re-run the SAME command with --resume added)
python scripts/run_centralized.py --config configs/<exp>/centralized.yaml --resume
python scripts/run_federated.py --config configs/<exp>/server.yaml --resume
python scripts/run_server.py --config configs/<exp>/server.yaml --resume

# Docker federated deployment (automated, all in containers)
scripts/docker-deploy-federated.sh <exp>                          # Launch server + 5 clients
scripts/docker-deploy-federated.sh <exp> --monitor                # Monitor until Round 1 completes
scripts/docker-deploy-federated.sh <exp> --no-clean               # Skip cleanup of previous containers

# Docker compose wrapper (server + node0 + node1 on this or separate hosts; reads .env)
# CURRENTLY BROKEN: docker-compose.yml / docker-compose.gpu.yml were deleted from the
# repo in e2db7ec (still on this branch's history) and never restored; run.sh still
# references them via `-f docker-compose.yml [-f docker-compose.gpu.yml]` and will fail
# at the `docker compose` invocation. Use `scripts/docker-deploy-federated.sh` above
# instead until the compose files are restored.
./run.sh build --gpu
./run.sh up --gpu                 # full local experiment
./run.sh server --gpu -d          # only the aggregation server, background
SERVER_ADDRESS=192.168.1.10:8080 ./run.sh node 0 --gpu   # a node on another host
./run.sh logs node0 ; ./run.sh ps ; ./run.sh down

# Sequential multi-experiment queues (Docker, one after another; a failure does not stop the queue)
scripts/run-queue.sh exp70 exp71 exp72
scripts/run-queue.sh --file queue.txt        # one exp per line, '#' = comment
scripts/run-eval-queue.sh                    # post-hoc evaluation queue

# Plot experiment metrics
python scripts/plot_experiment.py runs/<name>/
```

Python 3.11 is required (strict `>=3.11,<3.12`). **There is no usable local interpreter on this
machine** — `.venv/` is Python 3.12 and does not have `fedmammobench` installed, and no `python3.11`
is on `PATH`. Real training/evaluation therefore runs in Docker (`Dockerfile` pins
`python:3.11-slim-bookworm` and symlinks `python3.11` to `python`; override `BASE_IMAGE` for CUDA).
Don't assume `pytest` or the `fedmammobench-*` entry points can be invoked directly on the host —
check first, or run them inside the image. Runtime deps are pinned in `requirements.txt` (not
`pyproject.toml`, which carries only loose constraints); for GPU boxes install `torch`/`torchvision`
from the CUDA wheel index *before* `pip install -r requirements.txt`.

CI (`.github/workflows/ci.yml`) runs `pytest tests/ -v --tb=short --cov=fedmammobench` on Python 3.11
plus a config-import smoke check, on every push to `main`/`feature/**` and every PR to `main`. CI covers
`src/` and `tests/` only — nothing validates `configs/**/*.ipynb`.

None of the commands above apply to the notebook experiments (`configs/exp*/*.ipynb`) — those are run
cell-by-cell in Jupyter on the lab workstation's GPU, outside Docker and outside this package. See
"Notebook experiments" below.

## Architecture

### Execution modes

The CLI exposes three entry points (`pyproject.toml [project.scripts]`):
- `fedmammobench-federated` → `cli/federated.py` → `federated/server.py:run_simulation` (Ray-based, all clients in-process)
- `fedmammobench-centralized` → `cli/centralized.py` (single-node training loop)
- `fedmammobench-evaluate` → `cli/evaluate.py` (loads checkpoint, runs `Evaluator`)

The `federated/server.py` also exposes `run_grpc_server` for real multi-device deployment via `scripts/run_server.py` + `scripts/run_client.py`. The simulation and gRPC paths share the same strategy/logging wiring; the difference is whether Flower spawns virtual clients or waits for physical ones over the network.

### Config system

All YAML configs inherit from `configs/base.yaml` via a `defaults:` key resolved in `configs/loader.py`. The loaded dict is deserialized into typed dataclasses in `configs/schema.py` (which re-exports everything from six sub-modules):

| Section | Module | Key fields |
|---------|--------|-----------|
| `data` / `partitioning` | `data_config.py` | `manifest_path`, `val_fraction`, `scheme` (iid/dirichlet/quantity_skew) |
| `model` | `model_config.py` | `name`, `weight_source`, `freeze_backbone`, `unfreeze_at_epoch`, `local_unfreeze_at_epoch` |
| `training` | `training_config.py` | `local_epochs`, `optimizer`, `scheduler`, `loss`, `save_best_checkpoint`, `best_checkpoint_metric`, `early_stopping_patience` |
| `training.augmentation` | `training_config.py` | `horizontal_flip`, `rotate_limit`, `normalize_preset`/`normalize_mean`/`normalize_std`, `resize_mode` (`squash` default / `letterbox`) |
| `federated` | `federated_config.py` | `num_clients`, `rounds`, `strategy`, `server_training`, `server_address`, `save_best_checkpoint`, `best_checkpoint_metric`, `early_stopping_patience` |
| `wandb` | `wandb_config.py` | `enabled` (default `true`), `project`, `mode` (online/offline/disabled) — one run per experiment, server/centralized side only |
| experiment + evaluation | `experiment.py` | cross-section validation (freeze reachability, preset↔channels) |

Call `cfg.validate()` after loading to catch bad combinations early. Each section has its own `validate()`; cross-section rules live in `ExperimentConfig.validate()`.

**Important:** `schema.py` re-exports all public names. `loader.py` resolves dataclass types via `vars(schema_mod)`, so any new public symbol must be re-exported there or YAML loading breaks.

### Registry pattern

Three registries follow the same decorator pattern — new entries self-register at import time via side-effect imports in the package `__init__.py`:

- **Strategies** (`federated/strategies/registry.py`): `@register_strategy("name")` on a builder function; reference impl is `fedavg.py`. Add import in `strategies/__init__.py`.
  Registered: `fedavg`, `fedprox`, `fedbn`, `scaffold`, `fedadam`, `fedyogi`.
- **Models** (`models/factory.py`): `@register_model("name")` returning an `nn.Module` with correct `in_channels`/`num_classes`/`dropout`. Add name to `Literal` in `model_config.py`. Add import in `models/__init__.py`.
  Registered: `resnet18` (default), `resnet50`, `efficientnet_b0`, `densenet121`, `inception_v3`.
- **Datasets** (`datasets/registry.py`): `@register_dataset("name")` on a builder `(cfg, train_tx, eval_tx) → dict[str, MammographyDataset]`. No `Literal` update needed (`data.name` is a plain `str`). Add import in `datasets/__init__.py`.
  Registered: `cbis_ddsm`, `vindr_mammo`, `mammo_bench`, plus the sentinel `none` (server owns no local dataset).

Shared model plumbing lives outside the per-arch modules, so a new backbone only has to produce the
feature extractor:
- `models/_head.py::build_head` — the configurable MLP head from `model.head`; always ends in
  `Linear(*, num_classes)` emitting **raw logits** (softmax/sigmoid belongs to the loss and to
  `Evaluator`, never to the model).
- `models/_adapt.py` — 1- vs 3-channel input stem. `adapt_first_conv` rewrites the module (model
  builders); `adapt_weight_tensor` rewrites the tensor (weight loaders). Both default to
  `sum_preserving` so activation magnitude survives the channel change.

### FL training loop

Each federated round (simulation or gRPC):
1. **Server** calls `on_fit_config_fn` to broadcast `{current_round, local_epochs, ...}` to selected clients.
2. **Client** (`FedMammoBenchClient.fit`): loads server parameters (strict), runs `apply_freeze_policy`, optionally applies cyclic within-round unfreeze at `local_unfreeze_at_epoch`, trains for `local_epochs`, returns updated weights + metrics.
3. **Strategy** (`aggregate_fit`) averages weights. If `server_training.enabled`, `attach_server_training` wraps `aggregate_fit` to run a server-side training step afterwards (`new_global = (1-w)*aggregated + w*server_trained`).
4. **Clients** evaluate the aggregated model on their local val split; strategy `aggregate_evaluate` weighted-averages → logged to `server_federated_metrics.csv`.
5. **`NodeMetricsRecorder`** (wraps the strategy) captures per-node fit/eval CSVs, per-round timing, and saves `global_model.pt` at the end; if `federated.save_best_checkpoint` is set, it also overwrites `weights/global_best.pt` whenever the tracked weighted-average eval metric improves round-over-round. If `federated.early_stopping_patience` is also set, it raises an internal exception (`EarlyStoppingTriggered`) once that many rounds pass with no improvement — Flower has no native "stop early" hook, so `server.py` catches it around `fl.simulation.start_simulation`/`fl.server.start_server` as a clean, expected stop rather than a crash. Same mechanism centrally via `Trainer.fit(early_stopping_patience=...)`, which just `break`s its own loop. Both require `save_best_checkpoint=True` in the same section (config-level validation enforces this).

### Resume after a crash

`--resume` on `run_centralized.py`/`run_federated.py`/`run_server.py` (v0.8.0) — no new config fields.
On a shared workstation, a hard crash (segfault, OOM-killer, CUDA driver reset) skips Python's
`finally` blocks entirely, so the existing end-of-run checkpoint saves don't help; `--resume` instead
overwrites `weights/final.pt` (centralized, + optimizer/scheduler state) or `weights/global_model.pt`
(federated) every epoch/round as a safety net, and reloads from it on the next invocation. `training.epochs`/
`federated.rounds` are the TOTAL budget across the original run + resumes. Federated resume can't ask
Flower to start its round counter above 1, so it runs only the remaining rounds and applies a
`round_offset` everywhere a round number is logged or sent to clients (never to Flower's own strategy
calls). See `configs/README.md`'s "Resume tras un crash" section for the full mechanism and limitations
(GradScaler state, mid-epoch/round crashes, `num_clients`/`partitioning` drift between resumes).

Two frozen-backbone subtleties in that loop (both fixed 2026-07-08, regression tests in `tests/test_audit_fixes.py`):
- **BatchNorm drift under freeze.** `model.train()` (called every epoch) re-enables *all* modules including frozen-backbone BN layers, which keep updating `running_mean`/`running_var` even though `requires_grad=False` stops γ/β from updating. `Trainer` re-pins BN layers with frozen affine params back to `eval()` after each `train()` call (`_freeze_bn_running_stats`) — otherwise per-client BN stats drift independently and get corrupted on aggregation.
- **Cyclic unfreeze needs a new optimizer param group.** When `local_unfreeze_at_epoch` triggers a mid-round unfreeze, the optimizer was built while those layers were still frozen and won't otherwise receive their params — `FedMammoBenchClient.fit` (`federated/client.py`) diffs newly-`requires_grad` params against the optimizer's existing param IDs and calls `optimizer.add_param_group(...)` for the new ones (using `optimizer.lr_backbone` if set).

### Outputs

**Training runs** write under `runs/<name>/`:
- `server_federated_metrics.csv` — primary federated metric (weighted avg across nodes)
- `server_metrics.csv` — centralized evaluation on server holdout (opt-in via `data.name != none`)
- `server_timing.csv` / `timing_summary.csv` — per-round and total wall times
- `weights/global_model.pt` — final aggregated checkpoint (federated, always written)
- `weights/global_best.pt` — best-round checkpoint (federated, opt-in via `federated.save_best_checkpoint`; server never reloads this itself, evaluate it post-hoc)
- `weights/final.pt` / `weights/best.pt` — last-epoch / best-epoch checkpoint (centralized, `best.pt` opt-in via `training.save_best_checkpoint`; `scripts/run_centralized.py` reloads `best.pt` before test-set evaluation when enabled)
- `clients/client_<id>/fit_metrics.csv`, `eval_metrics.csv` — per-node local view
- `tb/` — TensorBoard event files
- `wandb/` — local W&B run data, only if `wandb.enabled` (default `true`); populated even offline, sync later with `wandb sync`
- `plots/` — PNGs auto-generated at the end of every run via `fedmammobench.plotting.autoplot` (never fails the run; see `scripts/plot_experiment.py` to regenerate by hand)

**Notebook runs** (`configs/exp*/*.ipynb`, see below) share `runs/` but use a different, flatter layout
— don't expect the files above:
- `metrics.json` — test-set summary (accuracy, precision, recall, f1, roc_auc) + the run's hyperparameters
- `loss_history.csv` — per-epoch train/val loss
- `plots/loss_curve.png`, `plots/test_confusion_matrix.png`, `plots/test_metrics_bar.png`,
  `plots/test_roc_curve.png`
- `best_model.pth` / `last_model.pth` — early-stopping and final-epoch checkpoints (gitignored)
- `wandb/run-<ts>-<id>/` — the notebook's own W&B run dir (`wandb.init(dir=RUN_DIR)`), **committed**,
  binary `.wandb` file and logged media included, for every run from `exp23` on
- `predictions.csv` — Blocks 7–8 only (`exp28`–`exp41`): the evaluated manifest rows (every original
  column) plus `y_true`/`y_pred`/`y_prob` appended, one row per test-set image
- `per_dataset/metrics_per_dataset.json`, `per_dataset/test_metrics_per_dataset.png`,
  `per_dataset/test_confusion_matrix_per_dataset.png`, `per_dataset/predictions_<source_dataset>.csv`
  — Blocks 7–8 only: the same test metrics/predictions broken out by `source_dataset`, kept out of
  `plots/`/`predictions.csv` (global-only) on purpose

Unlike package runs, these notebook outputs **are committed** (`.gitignore` only excludes `*.pt`/`*.pth`),
so `runs/` doubles as the results record for the notebook series.

**Post-hoc evaluation** (via `scripts/run-eval-queue.sh` or `fedmammobench-evaluate --output-dir`) writes under `runs/<exp>/eval/<config_name>/`:
- `run.log` — evaluation logs
- `metrics.json` — summary metrics (accuracy, precision, recall, auc, etc.)
- `predictions.csv` — per-sample predictions (if `--predictions-out` was passed)

**Orchestration logs** (multi-experiment queues) live in centralized directories:
- `runs/_logs/queue/queue_<timestamp>.log` — master log for `scripts/run-queue.sh` (training queue)
- `runs/_logs/eval/eval_<timestamp>.log` — master log for `scripts/run-eval-queue.sh` (evaluation queue)

### Preprocessing reproducibility caveat (v0.9.0)

`build_transforms()` (`datasets/transforms.py`) switched its resize interpolation from Albumentations'
default bilinear to `cv2.INTER_AREA`. Every image in this benchmark is larger than `data.image_size`,
so the resize is always a downscale — but this silently changed resized pixel values for *every*
config, past and future. Historical results (exp07–exp61+) stand as a record of what was run, yet
re-running the same YAML today does not reproduce them pixel-for-pixel. Before attributing a metric
delta to a code change, check whether the comparison straddles 0.9.0.

### Transfer learning / weight sources

`model.weight_source` controls how pretrained weights are injected (in `models/weight_loaders/`):
- `auto` (default) — infers from the legacy `model.pretrained` bool: `True` → `imagenet`, `False` → `none`. Explicit `weight_source` takes precedence.
- `imagenet` — torchvision defaults
- `radimagenet` — requires `$FEDMAMMOBENCH_RADIMAGENET_DIR` env var pointing to downloaded checkpoints
- `custom` — `model.checkpoint_path` to a `.pt` file (used for warm-start from a pretrain run)
- `none` — random init (ablation)

**Checkpoint key-namespace normalization.** `save_checkpoint` serializes the **full wrapper** model, so the project's own `.pt` files (`final.pt`, `global_model.pt`) carry keys prefixed `backbone.` (320/320 for resnet50), while a bare-backbone checkpoint doesn't. The `custom` loader (`weight_loaders/custom.py::_match_state_dict_prefix`) tries the state_dict as-is and under `module.`/`backbone.` transforms, keeping whichever maximizes key overlap with the target module, and raises `RuntimeError` if 0 tensors end up matching. Until 2026-07-08 this normalization didn't exist: the checkpoint loaded straight into `model.backbone` (bare keys expected) with no fallback, so a `backbone.`-prefixed checkpoint silently matched 0/320 tensors under `strict_load: false` and the federated global model trained from random init (~0.5 AUC vs. ~0.82 centralized) — see `tests/test_audit_fixes.py::TestCustomWarmStartLoader` for the regression coverage. The `radimagenet`/`imagenet` loaders were never affected — they consume backbone-only checkpoints and remap keys directly. Post-hoc `run_evaluation.py` was never affected either: it re-loads via `load_checkpoint(--checkpoint, model)` into the full wrapper with `strict=True`.

### Experiment configs layout

`configs/exp<NN>/` holds one experiment each, but the directories come in **two unrelated flavors** —
check what's actually inside before assuming:

- **YAML experiments** (package-driven, everything above applies): `server.yaml` for the aggregation
  server (simulation or gRPC), `client.yaml` for each physical node in gRPC mode, `pretrain.yaml` for
  centralized pre-training that generates `final.pt` for warm-start, and optionally `eval/<name>.yaml`
  for post-hoc evaluation targets (see `configs/exp70/eval/mammo_bench.yaml`).
- **Notebook experiments** (see the next section): a single `.ipynb` and nothing else. These bypass
  `fedmammobench` entirely.

Not every config directory is named `exp<NN>` — one-off YAML experiments get arbitrary names
(`configs/caladam01/`), and those are usually copies of a numbered experiment whose *header comments
still name the original* (`caladam01/server.yaml` opens with `# exp34_fedadam_...`). Trust `name:` /
`output_dir:` in the YAML, not the comment block.

Legacy flat configs are in `configs/legacy/`. Older experiment directories get pruned from the repo
once superseded, so check `ls configs/` for what's currently live rather than trusting a specific
number from history or docs — **and note the numbering has been reused**: `configs/exp01`–`exp09`
today are notebooks, unrelated to the exp01–exp09 YAML experiments that `CHANGELOG.md` and `docs/`
still discuss. `configs/README.md` is the authoritative reference for experiment-specific
mechanics (per-node hyperparameter matching, best-checkpoint selection, resume, W&B, autoplot).

**Seven config generators live in `scripts/`; none of their output should be hand-edited.**
`gen_lr_backbone_grid.py` emits the exp10–exp19 notebook grid (see next section);
`gen_depth_block_grid.py` emits Block 6 (`exp24`–`exp27`, generated from the `exp23` template — see
next section); `gen_img256_block_grid.py` emits Block 7 (`exp29`–`exp32`, generated from the `exp28`
template — same idea, one resolution lower); `gen_sampler_block_grid.py` emits Block 8
(`exp38`–`exp41`, generated from the `exp37` template — same grid again, with balanced sampling);
`gen_split702010_block_grid.py` emits Block 9 (`exp43`–`exp46`, generated from the `exp42` template —
Block 8 once more on the 70/20/10 manifest); `gen_nocmmd_holdout_depth_grid.py` emits most of Block 11
(`exp52`–`exp54`, generated from the `exp51` template — a depth ladder on the CMMD-free manifest with
a warm-started head, see below; Block 10 has no generator, it's hand-authored per-cell);
`gen_federated_grid.py` emits the federated YAML grid (8 files per experiment:
`server.yaml`, `client.yaml`, `eval/{mammo_bench,node1..5_partition}.yaml`), holding everything
constant but `rounds`, `local_epochs` (= `scheduler.t_max`) and the aggregation strategy. Its module
docstring and inline comments carry calibration learnings that are cheaper to read than to rediscover
(e.g. the FedAdam `eta` value, and which manifest each node gets) — read it before writing a new
federated grid by hand.

## Notebook experiments (`configs/exp*/*.ipynb`)

The current line of work on `feature/radimagenet` (**exp01–exp19**, core series) is **standalone
Jupyter notebooks that do not import `fedmammobench` at all** — plain torchvision `resnet50` + a
hand-written `Classifier`, wrapped in a local `FullModel(backbone, classifier)`. Nothing in the
sections above (config schema, registries, strategies, weight loaders, `Trainer`) is in play. They
are all centralized, single-GPU, `Mammo-Bench`-only ablations over which ResNet50 blocks are
unfrozen (`layer4` / `layer4+layer3` / fully frozen) crossed with head type (`linear` / dropout
0.3 / 0.5).

**`configs/exp20`–`exp22` extend the same series past exp19** (same notebook structure —
parameters cell, seeding, `metrics.json` — and have committed `runs/` results) but vary image
size, batch size, and loss (`img384`/`img512`, discriminative LR + BCE) rather than freeze depth
or head type. They postdate `docs/EXPERIMENTOS_CENTRALIZADOS.md`, which as of this writing only
covers exp01–exp19 — don't assume that doc is exhaustive, check `ls configs/exp2*` and `ls runs/`
for what's actually landed.

**`configs/exp23` is both the W&B template for this notebook series and the base of Block 6.**
It started as a straight copy of `exp22` plus a hand-rolled `wandb.init()`/log/`finish()` following the
same degrade-to-offline logic as the package's `WandbWriter` (`src/fedmammobench/utils/wandb_utils.py`):
never blocks or prompts for a key, falls back to offline mode if no credentials are cached. For any
*future* one-off notebook outside a generated block, copy exp23's W&B cells (imports, `WANDB_*` params,
init, per-epoch log inside `train_model`, final test-metrics/image log + `finish()`) rather than
reinventing the pattern. The lab workstation authenticates via a shared team **service account** in
`~/.netrc` (not a personal key) — never `cat` that file or paste a key into a notebook cell (which gets
committed with its outputs); verify credentials with `grep -q "api.wandb.ai" ~/.netrc` instead.

On top of that, `exp23` diverged further from `exp22` and became the base config for **Block 6 —
depth of unfreezing** (`exp23`–`exp27`): `CrossEntropyLoss` instead of `BCEWithLogitsLoss` (the MLP
head's `linear2` outputs 2 raw logits, not 1 + sigmoid), best-checkpoint/early-stopping tracked by
**F1-macro on validation** instead of min val loss (`val_f1_history`, mirrors `val_loss_history`), the
head (`linear1`→`bn1`→ReLU→dropout→`linear2`, 2048→512→2) reinitialized `normal(mean=0, std=0.01)`
instead of `nn.Linear`'s default kaiming-uniform (`bn1` and the RadImageNet backbone are left alone),
and a single `LR=LR_BACKBONE=1e-4` for both the head and backbone param groups — **intentionally not
differential**, which deliberately re-enters the failure mode `scripts/gen_lr_backbone_grid.py`
documents from exp02/03/05/06/08/09 (undifferentiated LR destroys RadImageNet backbone weights in 1-3
epochs); this is now itself an ablation axis for the block, not an oversight. `BLOCK = 6` in all five
notebooks (was a stale `3` copied all the way from `exp20`–`exp22`, unrelated to their actual content —
fixed here). `UNFREEZE_IDX` — a list of `backbone` `nn.Sequential` indices, see the freeze cell for the
index→layer map (`0`=conv1, `1`=bn1, `4`-`7`=layer1-4) — is the only thing that varies between the five:

| Notebook | `UNFREEZE_IDX` | Depth |
|---|---|---|
| `exp24` | `[]` | backbone fully frozen |
| `exp23` | `[7]` | layer4 only |
| `exp25` | `[7, 6]` | layer4 + layer3 |
| `exp26` | `[7, 6, 5]` | layer4 + layer3 + layer2 |
| `exp27` | `[7, 6, 5, 4, 1, 0]` | entire backbone |

The freeze cell and the optimizer cell in `exp23` were generalized to read `UNFREEZE_IDX` instead of
hardcoding `backbone[7]` — the optimizer builds its backbone param group from
`[p for p in model.backbone.parameters() if p.requires_grad]` rather than a fixed index, so those two
cells are byte-identical across all five notebooks; only the parameters cell
(`EXP_ID`/`RUN_NAME`/`UNFROZEN_DESC`/`UNFREEZE_IDX`/`WANDB_TAGS`) differs. `scripts/gen_depth_block_grid.py`
substitutes exactly that cell to produce `exp24`–`exp27` from the `exp23` template — `exp23` itself is
hand-edited, not generated, so don't regenerate over it; edit the generator's `EXPERIMENTS` list and
re-run with `--force` to change `exp24`–`exp27`. All five have been executed and have committed
`runs/exp2{3..7}...` results (added 2026-08-16, `93c3a8d`).

**Stale manifest reference in `exp23`–`exp27` (as of `ee54a6b`, 2026-08-16).** Their params cell
points `MANIFEST_PATH` at `manifests/fedmammobench_tompei.csv`, which that same commit deleted from
the repo (replaced by a corrected `manifests/fedmammobench.csv`, still carrying the `source_dataset`
column). The five notebooks already ran successfully against the old file before it was removed, so
their committed `runs/` results stand, but re-running any of `exp23`–`exp27` today will fail at the
manifest-loading cell until `MANIFEST_PATH` is repointed at `fedmammobench.csv`. Block 7 (below)
already points at the live file — don't copy the stale path from `exp23`–`exp27` into new work.

**Block 7 — same depth-of-unfreezing grid as Block 6, at 256px instead of 512px, plus a per-database
test breakdown (`exp28`–`exp32`).** `exp28` is a hand-edited copy of `exp23` (same architecture/
hyperparameters/seed/W&B) with three changes: `IMAGE_SIZE` 512 → 256; `MANIFEST_PATH` repointed at
`manifests/fedmammobench.csv` (see above); and a `source_dataset`-level test breakdown appended after
the existing global test cell.

Evaluation is built around one reusable function, `evaluate_model(model, checkpoint_path, dataloader,
...)`: it loads `checkpoint_path` into `model` and runs inference over whatever `DataLoader` it's
handed, returning accuracy/AUC/precision/recall/F1 + confusion matrix (AUC/precision/recall/F1 come
back `None` with a `"warning"` key when the subset has fewer than 2 classes present — a real risk on
the smaller per-dataset subsets). It does no dataset/dataloader construction itself — the caller
decides what data and which checkpoint that means. The global test cell calls it once with the `test`
`DataLoader` already built earlier in the notebook and `BEST_MODEL_PATH`; the per-dataset cell then
builds one fresh `DataLoader` per `source_dataset` value (`CSVDataset`'s `dataframe=` constructor path,
added alongside the existing `csv_file=`+`split=` one — `CSVDataset` still filters a manifest by split
when no `dataframe` is given, so `train`/`val` construction is untouched) and calls `evaluate_model`
again for each, same `BEST_MODEL_PATH`. This means the checkpoint gets loaded from disk and the test
set gets inferred over 5 times total (once "global," then once per dataset) instead of once — a
deliberate trade for a single, stateless, order-independent evaluation function instead of one that
depends on `model` already holding the right weights from an earlier cell.

Outputs land in two places on purpose: `RUN_DIR/plots/` + `RUN_DIR/metrics.json` stay global-test-only
(unchanged from Block 6), while `RUN_DIR/per_dataset/metrics_per_dataset.json` + a grouped bar chart +
a confusion-matrix grid go in their own `RUN_DIR/per_dataset/` folder, so the two granularities' files
never mix. `wandb_run.finish()` moved from the end of the global-metrics cell to the end of the
per-dataset plotting cell, so per-dataset metrics/images still land on the same W&B run. `BLOCK = 7` in
all five notebooks; `UNFREEZE_IDX` is the only thing that varies:

`save_predictions_csv(manifest_df, result, out_path)` writes per-sample predictions alongside the
aggregate metrics: it copies whichever `DataFrame` was just evaluated (`test_processed.data` for
global, a `source_dataset` subset for per-dataset) and appends `y_true`/`y_pred`/`y_prob` from
`evaluate_model`'s returned `"labels"`/`"preds"`/`"probs"` lists — assigned by *position*, not by
pandas index, so it depends on the same row-order guarantee the rest of this section already relies
on (`shuffle=False` + `CSVDataset(dataframe=...)` preserving input row order). Global goes to
`RUN_DIR/predictions.csv`; each per-dataset subset goes to
`RUN_DIR/per_dataset/predictions_<source_dataset>.csv`, written before that iteration's `result` gets
its `labels`/`preds`/`probs` popped off (they're no longer needed once the CSV is on disk).

| Notebook | `UNFREEZE_IDX` | Depth | Block 6 counterpart |
|---|---|---|---|
| `exp29` | `[]` | backbone fully frozen | `exp24` |
| `exp28` | `[7]` | layer4 only | `exp23` |
| `exp30` | `[7, 6]` | layer4 + layer3 | `exp25` |
| `exp31` | `[7, 6, 5]` | layer4 + layer3 + layer2 | `exp26` |
| `exp32` | `[7, 6, 5, 4, 1, 0]` | entire backbone | `exp27` |

`scripts/gen_img256_block_grid.py` substitutes the parameters cell to produce `exp29`–`exp32` from the
`exp28` template, mirroring `gen_depth_block_grid.py` exactly (freeze/optimizer/per-dataset-eval cells
are already dynamic over `UNFREEZE_IDX` and don't need touching). `exp28` itself is hand-edited, not
generated — edit the generator's `EXPERIMENTS` list and re-run with `--force` to change `exp29`–`exp32`.
All five have now been executed, with committed `runs/exp2{8,9}...`/`exp3{0,1,2}...` results.

**`configs/exp33`–`exp36` extend Block 7's fully-frozen setting (`exp29`'s `UNFREEZE_IDX = []`) into a
loss-function ablation — hand-edited, not part of a generator.** All four keep `BLOCK = 7` (inherited,
unchanged from copying the exp29-lineage template — same kind of stale label the Block 6 notes flag for
`exp20`–`exp22`, not a new numbered block) and bump `DROPOUT` 0.3 → 0.5 (`mlp05` vs. Block 7's `mlp03`
in the run name) versus their Block 7 counterpart. What varies:

| Notebook | Loss | `LR`/`LR_BACKBONE` | Notes |
|---|---|---|---|
| `exp33` | `CrossEntropyLoss()`, unweighted | `1e-4` | same loss as Block 7 |
| `exp34` | `CrossEntropyLoss(weight=class_weights)` | `1e-4` | class-balanced (see below) |
| `exp35` | `CrossEntropyLoss(weight=class_weights)` | `1e-4` | repeat of `exp34` — identical params/code, re-run |
| `exp36` | `CrossEntropyLoss(weight=class_weights)` | `4e-4` | same as `exp34`/`exp35` but 4x LR (still undifferentiated head/backbone) |

`class_weights` (`exp34`–`exp36`) is inverse-class-frequency computed on `train_processed` (not
val/test) and mean-normalized so weights average to 1 — same formula as the package's
`compute_class_weights` (`src/fedmammobench/training/losses.py`), reimplemented by hand since these
notebooks don't import `fedmammobench`. It's a response to the ~66/34 Benign/Malignant train imbalance
in `manifests/fedmammobench.csv`. All four otherwise reuse Block 7's evaluation/output machinery
unchanged — `runs/exp3{3..6}...` each have `per_dataset/`, `predictions.csv`, and the rest of the
Block 7 layout described above.

**Block 8 — Block 7's depth grid again, with `(source_dataset × class)`-balanced sampling
(`exp37`–`exp41`).** Generated by `scripts/gen_sampler_block_grid.py` off the hand-edited `exp37`
template (itself derived from `exp36`). All five ran on 2026-08-19 and their results are committed
(`184a24f`, 2026-08-25) — `runs/exp3{7,8,9}...` / `runs/exp4{0,1}...`, full Block 7 layout. **Read
"What Block 8 actually showed" below before designing a follow-up: the sampler did not work.**

The motivation, measured on Block 7's committed `predictions.csv` files: `P(malignant | source_dataset)`
in train ranges from 0.046 (kau-bcmd) to 0.489 (cmmd), and identifying the source dataset from an image
is trivial, so minimizing CE rewards emitting the per-dataset prior instead of reading the lesion. That
shortcut is directly observable — restricted to **benign** test images only, the model's mean predicted
probability tracks each dataset's train prevalence at **Pearson r = 0.98 (Spearman 1.00)**. A
zero-vision classifier that only maps source dataset → its train prevalence scores **AUC 0.726** on this
test set, and every frozen-backbone run (`exp29`, `exp33`–`exp36`) sits at macro-per-dataset AUC
0.719–0.739, i.e. no better than it; only the unfrozen configs (`exp30`/`exp31`, macro 0.754) clear it.

Three changes from `exp36`, everything else held fixed:

- **`WeightedRandomSampler` on the train loader**, weights `w_i = n_cell ** (-SAMPLER_ALPHA)` over the
  8 `(source_dataset, class)` cells. At `SAMPLER_ALPHA = 1.0` all cells are equiprobable, so
  `P(malignant | dataset) = 0.5` everywhere and dataset identity carries zero information about the
  label — the shortcut loses its gradient. `num_samples = len(train_processed)` keeps 416 steps/epoch,
  same as Block 7. `val`/`test` loaders are untouched (natural distribution, `SequentialSampler`).
- **`class_weight` dropped from the loss** (`CrossEntropyLoss()` bare). With the sampler at α=1 the
  stream is already 50/50 *within* each dataset; layering `exp34`–`exp36`'s global inverse-frequency
  weights `[0.683, 1.317]` on top would double-count the balancing and over-weight malignant ~1.93x.
- **`LR = LR_BACKBONE = 1e-4`**, back to Block 7's value (`exp36` used `4e-4`, which was only safe
  because its backbone was frozen; four of these five unfreeze it). This makes Block 8 comparable
  1-to-1 against `exp28`–`exp32` with the sampler as the main variable.

`DROPOUT` stays at `exp36`'s 0.5, `BLOCK = 8`, and `UNFREEZE_IDX` is the only thing varying:

| Notebook | `UNFREEZE_IDX` | Depth | backbone trainable | Block 7 counterpart |
|---|---|---|---|---|
| `exp38` | `[]` | backbone fully frozen | 0% | `exp29` |
| `exp37` | `[7]` | layer4 only (template) | 63.7% | `exp28` |
| `exp39` | `[7, 6]` | layer4 + layer3 | 93.9% | `exp30` |
| `exp40` | `[7, 6, 5]` | layer4 + layer3 + layer2 | 99.0% | `exp31` |
| `exp41` | `[7, 6, 5, 4, 1, 0]` | entire backbone | 100% | `exp32` |

Two things to know before reading Block 8's results:

- **α=1 costs repetition.** The two rare cells (inbreast-malignant and kau-bcmd-malignant, 82 train
  images each, from only 40 and 28 patients) get sampled ~10.2x per epoch, and ~46% of unique train
  images are drawn in a given epoch (sampling with replacement at fixed `num_samples`). Watch for
  memorization of those cells. Capping the weight is *not* a neutral safety valve — it truncates
  exactly those two cells and partially restores the shortcut (a 10x cap moves `P(mal|kau)` from 0.498
  back to 0.282); lowering `SAMPLER_ALPHA` degrades all 8 cells smoothly instead and is the better knob.
- **Accuracy on kau-bcmd will drop, and that is not a regression.** Training at 50/50 calibrates the
  model to that prior, so on a 2.7%-prevalence test subset it over-predicts malignant. Compare AUC and
  macro AUC, and pick the threshold on validation per dataset, before reading F1 or accuracy.

**What Block 8 actually showed: the sampler did not remove the shortcut, and mostly cost accuracy.**
Measured from the committed artifacts (global AUC from `metrics.json:test.auc`, macro from the four
per-dataset AUCs in `per_dataset/metrics_per_dataset.json`, correlation recomputed from
`predictions.csv` the same way the motivation above was):

| Depth | Block 7 | AUC | macro | Block 8 | AUC | macro |
|---|---|---|---|---|---|---|
| frozen | `exp29` | 0.818 | 0.737 | `exp38` | **0.758** | 0.721 |
| layer4 | `exp28` | 0.805 | 0.722 | `exp37` | 0.815 | 0.729 |
| +layer3 | `exp30` | 0.839 | 0.754 | `exp39` | 0.811 | 0.713 |
| +layer2 | `exp31` | 0.848 | 0.754 | `exp40` | 0.835 | 0.753 |
| all | `exp32` | 0.846 | 0.735 | `exp41` | 0.836 | 0.750 |

- **The shortcut survives.** On benign test images only, mean predicted probability still tracks train
  prevalence at Pearson **0.87–0.91** across the five runs (Block 7: 0.89–0.99), with **Spearman 1.00 in
  all five** — the per-dataset ordering (cmmd > cdd-cesm > inbreast > kau-bcmd) is untouched even though
  training saw `P(malignant | dataset) = 0.5` everywhere. Balancing the *sampler* did not make dataset
  identity uninformative at inference. Caveat before over-reading this: it is a correlation over 4
  points, and part of that ordering may be genuine per-dataset difficulty rather than a learned prior.
- **Macro per-dataset AUC (0.713–0.753) still straddles the 0.726 zero-vision baseline**, exactly as in
  Block 7. The regularization did not buy the thing it was built to buy.
- **The frozen case got clearly worse** (`exp38`, −0.060 global AUC vs `exp29`) — with no backbone
  adaptation available, the ~10x repetition of the two rare cells is pure variance. Unfrozen configs are
  roughly a wash (±0.03).
- **Runs train longer**: best epoch lands at 12–36 vs Block 7's 9–22, so a repeat needs the same
  `PATIENCE = 20` headroom.

Follow-ups worth more than another α sweep: fixing the label noise (`docs/EXPERIMENTOS_CENTRALIZADOS.md`,
CMMD patient-level propagation) or an explicit domain-adversarial / per-dataset-normalization term —
resampling alone has now been shown insufficient here.

The sampler cell ends with a self-check that draws one epoch and prints `P(malignant | dataset)`; it
exists because the weight vector is positional. `CSVDataset.__getitem__` uses `self.data.iloc[idx]` and
`WeightedRandomSampler` indexes by that same position, so `w[i]` must line up with
`train_processed[i]` — the cell builds it with `groupby(...).transform("size")`, which preserves row
order, and a `merge`/`groupby().apply()` there would misalign the weights silently. Same row-order
guarantee `save_predictions_csv` already depends on.

**Block 9 — Block 8 re-run on a 70/20/10 split (`exp42`–`exp46`).** Generated by
`scripts/gen_split702010_block_grid.py` off the hand-edited `exp42` template (a copy of `exp37`).
In progress as of 2026-08-25 — `exp42`/`exp43`/`exp44` have run locally
(`runs/exp4{2,3,4}_...` present, notebooks show executed outputs) but are **not yet committed**
(`git status`: modified notebooks + untracked `runs/` dirs); `exp45`/`exp46` haven't run yet. Treat
any numbers from these three as provisional until the results land in a commit — don't cite them as
"COMPLETADO" the way Block 8's table below does. Everything is held fixed against
Block 8 — same architecture, seed, `SAMPLER_ALPHA = 1.0`, `DROPOUT = 0.5`, bare `CrossEntropyLoss()`,
`LR = LR_BACKBONE = 1e-4`, 256px — except `MANIFEST_PATH`, which points at
`manifests/fedmammobench_70_20_10.csv` instead of `manifests/fedmammobench.csv`. Two provenance
fields, `SPLIT_SCHEME` and `MANIFEST_PATH.name`, are recorded in the W&B config and in `metrics.json`
so a Block 9 run is never mistaken for a Block 8 one after the fact. Numbering is aligned with Block 8
(`exp42+n` ↔ `exp37+n`), so the comparison is index-by-index:

| Notebook | `UNFREEZE_IDX` | Depth | Block 8 counterpart |
|---|---|---|---|
| `exp42` | `[7]` | layer4 only (template) | `exp37` |
| `exp43` | `[]` | backbone fully frozen | `exp38` |
| `exp44` | `[7, 6]` | layer4 + layer3 | `exp39` |
| `exp45` | `[7, 6, 5]` | layer4 + layer3 + layer2 | `exp40` |
| `exp46` | `[7, 6, 5, 4, 1, 0]` | entire backbone | `exp41` |

**The 70/20/10 manifest** (`manifests/fedmammobench_70_20_10.csv`, built by
`scripts/resplit_manifest_70_20_10.py`, seed 42) is the 80/10/10 `fedmammobench.csv` with **253
whole patients moved train → val and nothing else**. Concretely:

- **Test is byte-identical** — the same 834 images / 253 patients as blocks 6–8. The script asserts
  this, plus patient-grouping and "no transition other than train → val", and refuses to write
  otherwise. So **test metrics are comparable 1-to-1 against Block 8**; val metrics are not, because
  val is a different, larger set.
- Patients land on 1774/507/253 = **70.01/20.01/9.98%**, exact per `source_dataset` up to integer
  rounding. Images follow at **70.00/20.00/10.00%** (per-dataset val share 19.99–20.04%) because the
  script picks *which* patients move by local search on image count and malignant rate — patients
  carry 1–16 images each, so image proportions can only be hit approximately, and this is the
  "closest possible" the constraint allows.
- Malignant image rate is preserved (val 34.0% vs. train 34.1%, test 33.2%).

What this block is actually testing: blocks 6–8 chose **both** early-stopping and the best checkpoint
on an 836-image val by F1-macro, and Block 8's best epoch wandered between 12 and 36 with no clean
relation to unfreeze depth — that criterion is noisy. Doubling val should stabilize model selection,
at the cost of ~12% fewer training images. **When reading the results, the two effects push in
opposite directions** — don't read a Block 9 win as "the split is better" without separating "larger
val selects better" from "smaller train fits worse". The sampler's own caveats carry over unchanged:
all 8 `(source_dataset × class)` cells stay populated, the rarest (kau-bcmd-malignant, 82 → 70 images)
is repeated ~10.4x per epoch vs. Block 8's ~10.2x, and ~46% of unique train images are drawn per epoch
in both.

**Block 10 — cross-architecture ablation: InceptionV3 and DenseNet121 at 2 depths, on the 70/20/10
split (`exp47`–`exp50`).** Extends the RadImageNet fine-tuning study to the other two architectures
RadImageNet actually publishes weights for (`weights/InceptionV3.pt`, `weights/DenseNet121.pt` — the
same files as in `weights/RadImageNet_pytorch.zip`; there is **no** RadImageNet-pretrained
Inception-ResNet-v2, and none is used here — that's a different architecture, not in torchvision, not
published by RadImageNet, and not a dependency of this project). Hand-authored, not generated by a
script — unlike the depth-only grids (Blocks 6–9), the architecture axis needs real per-cell surgery
(weight-key remapping, freeze indices, head input width), not a params-cell substitution, so
`configs/exp42/exp42.ipynb` (Block 9's `layer4`/`UNFREEZE_IDX=[7]` template) was cloned and patched
cell-by-cell instead of run through a generator. Not yet executed as of 2026-08-25 — no
`runs/exp{47,48,49,50}...`.

Everything else matches Block 9: `manifests/fedmammobench_70_20_10.csv`, `SAMPLER_ALPHA = 1.0`,
`DROPOUT = 0.5`, bare `CrossEntropyLoss()`, `LR = LR_BACKBONE = 1e-4`, the same 2-hidden-layer MLP head
(only the first `Linear`'s `in_features` changes — 2048 for InceptionV3, 1024 for DenseNet121, each
backbone's native feature width). Only two depths per architecture instead of Block 9's five — "frozen"
and "last stage unfrozen," the two extremes already used throughout Blocks 6–9 — since a 5-depth grid
doesn't map cleanly onto non-ResNet topologies (see below):

| Notebook | Architecture | Depth | Trainable backbone params |
|---|---|---|---|
| `exp47` | InceptionV3 | last stage (`Mixed_7a`/`7b`/`7c`) unfrozen | 58.8% |
| `exp48` | InceptionV3 | fully frozen | 0% |
| `exp49` | DenseNet121 | last block (`denseblock4` + `norm5`) unfrozen | 31.1% |
| `exp50` | DenseNet121 | fully frozen | 0% |

**`IMAGE_SIZE` differs by architecture, breaking with Block 9's uniform 256px.** `exp49`/`exp50`
(DenseNet121) stay at 256px. `exp47`/`exp48` (InceptionV3) run at **299px** — torchvision's
`inception_v3` has hard-coded pooling strides that produce incorrect activations below ~299px (the
same constraint the package's own `models/inception.py` docstring documents for the YAML path); running
it at 256px to match the rest of the block would make the results invalid, not just
inconsistent-resolution. `exp47`/`exp48` therefore aren't resolution-comparable to the rest of Block
6–10 the way same-architecture depth comparisons are — a deliberate deviation, not an oversight.

**Freeze-index mapping is architecture-specific and was derived from the actual checkpoint contents,
not assumed by analogy with ResNet50.** `weights/InceptionV3.pt` and `weights/DenseNet121.pt` — like
`RadImageNet-resnet50.pth` — store their backbone as an `nn.Sequential` (`backbone.N.*` keys), *not*
under the target architecture's real module names, so they need the same kind of remap
`load_radimagenet_backbone` already does for ResNet50 in `exp23`–`exp46`. This differs from the
package's own weight loader: `models/weight_loaders/_keymaps.py::_BACKBONE_SEQUENTIAL_REMAP` only has
entries for `resnet50`/`resnet18` and assumes `densenet121`/`inception_v3` need no backbone-index remap
at all — that table would silently 0-match these two `.pt` files if pointed at them as-is (the same
failure mode as the pre-2026-07-08 `backbone.`-prefix bug documented under "Checkpoint key-namespace
normalization" above), a latent gap in the package worth fixing separately from this block.
- **InceptionV3**: `backbone.N` (N = 0,1,2,4,5,7–17; N=3,6 absent — `maxpool1`/`maxpool2` have no
  params) maps 1:1 onto `list(inception_v3(aux_logits=False).children())[N]` (`Conv2d_1a_3x3` …
  `Mixed_7c`, then `avgpool` at N=18 with no checkpoint entry needed). Verified by grouping the
  checkpoint's keys by branch-name signature (`branch1x1`/`branch7x7_*`/`branch3x3dbl_*`/…) and
  matching each against torchvision's known `InceptionA`–`E` submodule shapes, not by assuming the
  published index order matches ResNet's. `load_state_dict` on the full model matches all 564/564
  backbone tensors (only `fc.weight`/`fc.bias` reported missing — expected, that's the head being
  replaced).
- **DenseNet121**: the *entire* `features` extractor (`conv0`, `norm0`, four `denseblock`/`transition`
  pairs, `norm5`) is saved under a single `backbone.0.*` prefix — one element, not six — so the "remap"
  is just stripping that one prefix before `model.features.load_state_dict(...)`. Matches 725/725
  tensors, 0 missing, 0 unexpected.
- DenseNet's own `forward()` applies `relu` + `adaptive_avg_pool2d` to `features`' output *functionally*,
  not via a submodule, so the notebook wraps `features` in a small `DenseNetBackbone` module that
  replicates that step — `backbone.features[idx]`, not `backbone[idx]`, is what the freeze cell indexes
  for `exp49`/`exp50`.

All four notebooks' forward passes (dummy `(2, 3, IMAGE_SIZE, IMAGE_SIZE)` batches, the real checkpoint
weights, CPU) were validated end-to-end to produce `(2, 2)` logits before being committed, since none of
this can be exercised through the usual `pytest`/CI path (notebooks aren't covered by CI, and there's
no local Python 3.11 interpreter — see "Commands" above).

**Finding (2026-08-25, not yet fixed in already-run notebooks): `exp35`–`exp46`'s optimizer silently
excludes `bn2` and `linear3` — the output layer — from training.** Every notebook using the
2-hidden-layer MLP head (`exp35` onward: `exp35`, `exp36`, all of Block 8 `exp37`–`exp41`, and Block 9's
already-executed `exp42`–`exp44`) builds its optimizer's head param group as
`list(model.linear1.parameters()) + list(model.bn1.parameters()) + list(model.linear2.parameters())` —
a line that never grew past the single-hidden-layer head it was originally written for. `bn2` and
`linear3` (the real 512→2 output projection) aren't in *any* `param_groups` entry, so `optimizer.step()`
never updates them; they stay at their `normal_(mean=0, std=0.01)` / zero-bias init for the whole run.
This isn't inert: with everything upstream still free to train, it behaves like a fixed
random-projection readout — representations can still rotate to work with an untrained final layer,
consistent with these blocks' AUCs (0.71–0.85) not looking obviously broken — but it means none of
`exp35`–`exp44`'s reported numbers reflect a fully-trained head, and any AUC/F1 comparison against
Block 6/7 (single-hidden-layer head, fully trained) isn't apples-to-apples on this axis alone.
`exp45`/`exp46` (generated, not yet run) carry the same bug in their saved `.ipynb` source. Fixed in
`exp47`–`exp50` by collecting head params dynamically
(`[p for name, p in model.named_parameters() if not name.startswith("backbone.")]`, mirroring how
`backbone_params` is already collected) instead of naming layers by hand. **Not yet fixed retroactively
in `exp35`–`exp46`** — that would mean re-running all of them, which hasn't been done or requested yet.

**Block 11 — 80/10/10 without CMMD, warm-started from a third-party checkpoint (`exp51`–`exp54`).**
Requested ad hoc, not a continuation of the generator progression — diverges from every earlier block
on three axes at once:

1. **A new manifest, `manifests/fedmammobench_no_cmmd.csv`** — CMMD dropped entirely from train/val/
   test (not just test), generated by `scripts/filter_manifest_exclude_cmmd.py` from the original
   80/10/10 `manifests/fedmammobench.csv`. A pure row filter, not a re-split: every `patient_id`
   belongs to exactly one `source_dataset` (asserted in the script's `validate()`), so deleting CMMD's
   4722 rows doesn't touch any other patient's split assignment, and the three remaining datasets
   (cdd-cesm, inbreast, kau-bcmd) land back at ~80.0/10.0/10.0% automatically — they were already
   independently at that ratio inside `fedmammobench.csv`. Two large side effects to know before
   reading results: images drop from 8341 to 3619 (train 6671→2895), and — because CMMD was both the
   largest dataset and the one with the highest malignant rate (~49%) — train malignant prevalence
   collapses from 34.1% to 14.9% (val 33.6%→13.5%, test 33.2%→13.0%). Meaningfully harder class
   imbalance than any earlier block.
2. **Backbone *and* head warm-started from `weights/holdout_best.pt`**, not
   `RadImageNet-resnet50.pth`. Unlike every other checkpoint in this series, `holdout_best.pt` is a
   full `backbone.* + fc.*` state dict — the head is already trained, not random. Provenance: from
   another experiment on this machine, per the user; the exact run couldn't be pinned down (no file
   under `runs/` on this disk matches it by size or hash, and nothing in the repo references it — it
   appeared in `weights/` on 2026-08-25 with no accompanying script or config). Structurally it decodes
   cleanly: `backbone.{0,1,4-7}` is the same `nn.Sequential` indexing RadImageNet's own checkpoint uses
   (0=conv1, 1=bn1, 4-7=layer1-4), and `fc.{0,1,4,5,8}` is shape-for-shape identical to this series'
   2-hidden-layer head (`Linear(2048,512)→BN→Linear(512,512)→BN→Linear(512,2)`), so `exp51`'s two
   loader functions (`load_holdout_backbone`, `load_holdout_head_state`) are plain key-remaps —
   verified to match 318/318 backbone tensors and 16/16 head tensors, zero missing/unexpected. Because
   the head is warm-started too, `FullModel.__init__`'s usual `normal_(mean=0, std=0.01)` reinit is
   skipped entirely for this block — the one deliberate way `exp51`–`exp54` break comparability with
   the rest of the series.
3. **No sampler; class-weighted CE instead.** With CMMD gone, the dominant imbalance shifted from "4
   datasets with wildly different P(malignant|dataset)" (Block 8/9's problem) back to plain within-
   dataset class imbalance (85/15) — a `WeightedRandomSampler` over `(source_dataset × class)` cells
   was judged the wrong tool for that, so the block reuses Block 7's approach (`exp33`/`exp34`):
   `class_weight` inverse to frequency, mean-normalized, computed on the (already-filtered) train split
   — `[Benign: 0.298, Malignant: 1.702]`, a much sharper ratio than `exp34`'s `[0.683, 1.317]` because
   the imbalance itself is sharper post-filter.

`exp51` is the hand-authored template; `exp52`–`exp54` are generated by
`scripts/gen_nocmmd_holdout_depth_grid.py` off it, substituting only the parameters cell (freeze/
optimizer/class_weight cells are already dynamic over `UNFREEZE_IDX`, same mechanism as every earlier
block). Deliberately not a full 5-point ladder — the `layer4+layer3` point was never requested:

| Notebook | `UNFREEZE_IDX` | Depth | Backbone trainable |
|---|---|---|---|
| `exp52` | `[]` | backbone frozen | 0% |
| `exp53` | `[7]` | layer4 only | 63.66% |
| `exp51` | `[7,6,5]` | layer4+layer3+layer2 | 99.04% |
| `exp54` | `[7,6,5,4,1,0]` | entire backbone | 100% |

All four already carry the `bn2`/`linear3` optimizer fix from Block 10 (`head_params` collected
dynamically via `named_parameters()`, not named by hand) and were validated the same way as Block 10:
dummy `(2, 3, 256, 256)` forward passes against the real `holdout_best.pt` weights, at every depth,
producing `(2, 2)` logits with the expected trainable-parameter percentages. Not yet executed as of
2026-08-25 — no `runs/exp5{1,2,3,4}...`.

**`configs/exp71`–`exp77` are an earlier, superseded draft of this same work — don't treat them as
part of the series.** Tells: the filename carries a description (`exp72_resnet50_layer4only.ipynb`)
instead of being bare `expNN.ipynb`; ~14 cells instead of ~20; **no parameters cell** (no `EXP_ID` /
`RUN_NAME`), no seeding, and no `metrics.json` write — so they produce no `runs/` directory and
there is no committed record of their results. exp01–exp09 re-do these ablations properly. Read them
for history if you like, but derive new work from exp01–exp19.

Structure of each exp01–exp19 notebook, essentially identical across the series:
- A parameters cell near the top (`EXP_ID`, `RUN_NAME`, `BLOCK`, `HEAD`, `LR`, `BATCH_SIZE`,
  `NUM_EPOCHS`, `PATIENCE`, `PROJECT_ROOT`, `DATASET_ROOT`). Deriving a new experiment means copying a
  notebook and editing this cell — that is the whole "config system" here.
- **Absolute paths hardcoded to the lab workstation** (`/media/imagenesmedicas/DATA1/...`) for
  `PROJECT_ROOT`, `DATASET_ROOT`, and the RadImageNet checkpoint. They do not resolve on other machines
  and do not honor `MAMMO_DATA` / `WEIGHTS_DIR` / `FEDMAMMOBENCH_RADIMAGENET_DIR`.
- Data comes from `manifests/fedmammobench.csv` with columns `split` / `classification` /
  `preprocessed_image_path`, read by a local `CSVDataset` (not `datasets/`).
- Comments and printed output are in **Spanish**; keep that when editing them.
- Outputs land in `runs/<RUN_NAME>/` (see Outputs above) and are committed except the checkpoint.

**Blocks 4–5 (exp10–exp19) are generated, not hand-written.** `scripts/gen_lr_backbone_grid.py` takes
`configs/exp09/exp09.ipynb` as the template and substitutes the eight cells that vary (imports, params,
seeding, DataLoaders, head class, head instance, freeze, optimizer, summary); the grid itself is the
`EXPERIMENTS` list at the top of that script. Edit the generator and re-run it with `--force` rather
than editing those ten notebooks by hand — that is the only way the block stays internally consistent.
Blocks 1–3 (exp01–exp09) predate the generator and are hand-written; don't regenerate over them.
Unlike blocks 1–3, blocks 4–5 seed everything (`SEED = 42`, `seed_everything()` after the device cell
and again before the head init) and use discriminative LRs (`LR` for the head, `LR_BACKBONE` for the
unfrozen blocks, as separate AdamW param groups) — so their numbers are **not** directly comparable to
exp01–exp09, which ran unseeded with a single LR. exp10 and exp11 exist as seeded re-runs of exp07 and
exp09 to bridge that gap. Block 6 (`exp23`–`exp27`, see above) is generated the same way, from a
different template and generator (`gen_depth_block_grid.py` off of `exp23`) — it inherits blocks 4–5's
seeding but not their discriminative-LR convention (LR is intentionally undifferentiated in Block 6).
Block 7 (`exp28`–`exp32`, see above) repeats Block 6's grid at 256px via `gen_img256_block_grid.py` off
of `exp28`, and adds the per-`source_dataset` test breakdown described above. Block 8
(`exp37`–`exp41`, see above) repeats that grid once more via `gen_sampler_block_grid.py` off of
`exp37`, adding `(source_dataset × class)`-balanced sampling and dropping the loss's `class_weight`.
Block 9 (`exp42`–`exp46`, see above) is Block 8 verbatim on the 70/20/10 manifest, via
`gen_split702010_block_grid.py` off of `exp42`. Block 10 (`exp47`–`exp50`, see above) breaks the
generator pattern — hand-authored per-cell, not generated, because the architecture axis needs real
surgery (weight-key remapping, freeze indices, head input width) rather than a params-cell swap. Block
11 (`exp51`–`exp54`, see above) returns to the generator pattern: `gen_nocmmd_holdout_depth_grid.py`
off of `exp51`, a depth ladder on the CMMD-free manifest with a warm-started head.

`runs/` is the results index for this series: one directory per `RUN_NAME`, and a notebook with no
`runs/` entry has not been executed on the workstation yet. The names encode the ablation cell
(`exp13_resnet50_radimagenet_mammobench_l4l3_linearhead_lrbb1e-4`), so `ls runs/` is the fastest way
to see which combinations already have numbers before proposing a new one.

Two subtleties the package handles centrally are re-implemented by hand in each notebook — if you fix
one, fix it in the whole series, and keep them consistent with the package versions:
- `load_radimagenet_backbone` remaps `RadImageNet-resnet50.pth`'s `backbone.<N>.*` `nn.Sequential`
  keys onto torchvision names; without it `load_state_dict(strict=False)` matches zero tensors — the
  same failure mode documented under "Checkpoint key-namespace normalization" above.
- `freeze_bn_running_stats` re-`eval()`s BN layers whose affine params are frozen, mirroring
  `Trainer._freeze_bn_running_stats`.

## Documentation map

`docs/` is large and partly historical. Trustworthiness, highest first:

- `configs/README.md` (Spanish) — **authoritative** for how experiments are actually run: env vars,
  Docker mounts, per-node hyperparameter matching, best-checkpoint selection, resume, W&B, autoplot.
- `CHANGELOG.md` — authoritative for behavior changes and *why* they were made; each entry explains
  the failure mode it addresses. Read this before assuming current behavior from old code comments.
- `docs/EXPERIMENTOS_CENTRALIZADOS.md` (Spanish) — the exp01–exp19 notebook series: the 19
  configurations, their results, and the CMMD patient-level label-propagation defect that puts a
  ~0.44 val_loss floor under all of them. Read it before interpreting any number from `runs/exp*`.
- `docs/INFORME_EXP01_22.md` (Spanish) — written report of blocks 1–3 (`exp01`–`exp22`): shared
  methodology plus the per-run results table. Complements `EXPERIMENTOS_CENTRALIZADOS.md`; stops at
  `exp22`, so nothing on blocks 6–8.
- `docs/EXTENDING.md` — adding a strategy/model/dataset.
- `docs/SRC_STRUCTURE.md` (Spanish) — one-line-per-module map of `src/fedmammobench/`.
- `docs/audit-plan.md` + `docs/audit/*.md` — the closed 2026-07-08 audit that found the
  `backbone.`-prefix checkpoint bug (see "Checkpoint key-namespace normalization"). Historical, but the
  best written account of *why* the federated results before that date were invalid.
- `docs/EXPERIMENT_AUDIT.md` — pre-publication checklist for results.
- `docs/CHECKPOINT_COMPATIBILITY.md`, `docs/RADIMAGENET_IMPLEMENTATION.md`,
  `docs/TRANSFER_LEARNING_GUIDE.md` — weight-loading specifics.
- `docs/DATA_PREPARATION.md`, `docs/METHODOLOGY.md` — manifests and experimental design.
- `docs/DOCKER.md`, `docs/FEDERATED_DEPLOYMENT_GUIDE.md`, `docs/SETUP_6NODES.md`,
  `docs/QUICK_START_6NODES.md`, `docs/NODE_CONFIGURATION_MATRIX.md` — deployment; node counts and
  IPs are snapshots of a particular setup, verify against `configs/` before relying on them.
- `README.md` — **stale**. It advertises version 0.3.0 (actual: `pyproject.toml` 0.9.0) and every
  config path it names (`configs/fedavg_cbis_ddsm.yaml`, `configs/radimagenet_*.yaml`) has since
  moved to `configs/legacy/`. Prefer this file and `configs/README.md`.

## Repo conventions

- **Language:** source code, docstrings, and this file are in English; operational docs and shell
  scripts (`configs/README.md`, `run.sh`, `scripts/run-queue.sh`, `.claude/commands/*`) are in
  Spanish, matching the user. Follow whichever the surrounding file uses.
- **Project slash commands** live in `.claude/commands/` — `/docker-run`, `/docker-queue`,
  `/new-exp`, `/eval-experiments`, `/plot`, `/compare`, `/check-manifest`, `/validate-configs`.
  Prefer these over hand-rolling equivalent shell for routine experiment work.
- **`scripts/` is append-only in practice:** one-off `run-expNN-MM.sh` / `eval-expNN-MM.sh` drivers
  accumulate with hardcoded experiment IDs and checkpoint paths. They document what was run; do not
  edit them to run something new — copy the pattern or use `scripts/run-queue.sh`.
- **Data, weights, and every `*.pt`/`*.pth` are gitignored** and live on external disks referenced
  through `MAMMO_DATA` / `WEIGHTS_DIR` / `FEDMAMMOBENCH_RADIMAGENET_DIR`. Never hardcode those paths in
  package code or YAML (the notebooks already do, and that is a known wart, not a pattern to copy).
  Non-checkpoint run artifacts under `runs/` — `metrics.json`, CSVs, `plots/*.png` — are *not* ignored
  and do get committed.
- **`$output_file` in the repo root is a tracked shell-quoting accident** (an evaluation
  `metrics.json` written to a literal `$output_file`), not an input to anything. Ignore it; don't
  build on it.
- **Notebooks are committed with their outputs**, so a re-executed notebook produces a five-figure
  line diff. Expect that; don't try to "clean up" the diff by stripping outputs unless asked — the
  stored outputs are how the experiment's results are archived.

## Extension checklist

When adding a strategy, model, or dataset, see `docs/EXTENDING.md`. The short version:
1. Write the module with `@register_*` decorator.
2. Add the side-effect import in the package `__init__.py`.
3. For models: update the `Literal` in `model_config.py`.
4. Add a test; run `pytest tests/ -v`.
5. If RNG order, defaults, `Trainer` signature, or aggregation math changed — bump version in `pyproject.toml` and add a `CHANGELOG.md` entry.
