# fedmammo

Federated learning for binary breast-cancer classification (benign vs malignant) on mammography images, built on **Flower (FLwr)** and **PyTorch**.

The project is structured as a research prototype suitable for an undergraduate thesis: clean module boundaries, YAML-driven experiments, deterministic seeding, and metrics that map to the clinical literature (sensitivity, specificity, ROC-AUC). The first federated strategy implemented end-to-end is **FedAvg**; **FedProx, SCAFFOLD, and FedBN** are present as extensible stubs so they can be filled in without touching the rest of the codebase.

---

## Status

- FedAvg: implemented
- FedProx, SCAFFOLD, FedBN: scaffolded with strategy interface and TODO markers
- Datasets: synthetic loader works out of the box; CBIS-DDSM and VinDr-Mammo loaders are implemented against a *documented assumed directory layout* and will require minor manifest tweaks once you have the data on disk
- Compute target: multi-process Flower simulation via Ray on a single machine
- Centralized training script is included as a sanity baseline

---

## Project layout

```
fedmammo/
├── configs/                       # YAML experiment configurations
│   ├── base.yaml                  # defaults
│   ├── centralized_synthetic.yaml
│   └── fedavg_synthetic.yaml
├── src/fedmammo/
│   ├── configs/                   # dataclass schema + YAML loader
│   ├── datasets/                  # base class, CBIS-DDSM, VinDr-Mammo, synthetic, transforms, partitioning
│   ├── models/                    # ResNet18, EfficientNet-B0, factory
│   ├── training/                  # Trainer, losses
│   ├── evaluation/                # Evaluator, metrics
│   ├── federated/                 # Flower client + server + strategies
│   │   └── strategies/            # fedavg (impl), fedprox/scaffold/fedbn (stubs)
│   └── utils/                     # seeding, logging, tensorboard, checkpoint, csv
├── scripts/
│   ├── run_centralized.py
│   ├── run_federated.py
│   └── run_evaluation.py
├── tests/                         # smoke tests
├── data/                          # placeholder — user-provided
├── runs/                          # TensorBoard logs, CSVs, checkpoints (gitignored)
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation

Python 3.11 is required.

```bash
# 1. create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install PyTorch first if you need a specific CUDA build (optional)
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# 3. install fedmammo
pip install -e ".[dev]"
```

### Docker

```bash
# CPU-only
docker build -t fedmammo:cpu .

# CUDA (12.1, cuDNN 8)
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
             -t fedmammo:gpu .
```

---

## Quick smoke test (no real data needed)

The synthetic dataset generates noise tensors with realistic shapes so the whole graph (data &rarr; model &rarr; Trainer &rarr; FL loop) can be exercised before you have a real download.

```bash
# Centralized baseline
python scripts/run_centralized.py --config configs/centralized_synthetic.yaml

# Federated (FedAvg, 4 simulated clients, IID partition)
python scripts/run_federated.py --config configs/fedavg_synthetic.yaml
```

TensorBoard logs and CSVs land under `runs/<experiment_name>/`.

---

## Using real data

### CBIS-DDSM

The pipeline assumes you have downloaded the CBIS-DDSM release (TCIA) and converted DICOMs to PNG (or kept DICOM &mdash; both are supported by the loader). The default loader expects a CSV manifest with at least these columns:

| column           | description                                  |
|------------------|----------------------------------------------|
| `image_path`     | path to the image file, relative or absolute |
| `pathology`      | one of `BENIGN`, `BENIGN_WITHOUT_CALLBACK`, `MALIGNANT` |
| `patient_id`     | used for patient-level (non-leaky) splits    |
| `split`          | `train` / `val` / `test` (optional; otherwise we generate splits) |

Two of the BENIGN labels are merged into a single negative class. You point the loader at the manifest via YAML:

```yaml
data:
  name: cbis_ddsm
  manifest_path: /path/to/cbis_ddsm_manifest.csv
  image_root: /path/to/cbis_ddsm_pngs
  image_format: png   # or "dicom"
```

If your manifest uses different column names, override them under `data.columns` in the YAML.

### VinDr-Mammo

VinDr-Mammo ships as DICOM with `breast-level_annotations.csv` and `finding_annotations.csv`. The loader uses **breast-level** annotations and maps the BI-RADS assessment to a binary label using:

| BI-RADS | label     |
|---------|-----------|
| 1, 2    | benign    |
| 4, 5    | malignant |
| 3       | configurable — by default **dropped** (ambiguous) |

```yaml
data:
  name: vindr_mammo
  annotations_path: /path/to/breast-level_annotations.csv
  image_root: /path/to/images           # contains study_id/series_id/image.dcm
  birads_3_policy: drop                 # or "benign" or "malignant"
```

> **Honesty note:** The exact directory layout of the public releases changes over time and across mirrors. Both loaders are written defensively (paths are resolved relative to a configurable root, missing files are logged and skipped), but you should expect to spend an hour adapting the manifest reader to the version you download. The relevant code is isolated in `src/fedmammo/datasets/cbis_ddsm.py` and `src/fedmammo/datasets/vindr_mammo.py`.

---

## Federated configuration

The simulation entrypoint uses `flwr.simulation.start_simulation` with Ray. Each "hospital" is one Flower client. Both **IID** and **non-IID** partitioning are available:

```yaml
federated:
  num_clients: 4
  rounds: 20
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  strategy:
    name: fedavg            # fedavg | fedprox | scaffold | fedbn
    server_lr: 1.0
    # strategy-specific knobs go here (mu for fedprox, etc.)

partitioning:
  scheme: dirichlet         # iid | dirichlet | quantity_skew
  alpha: 0.5                # only used by dirichlet; lower = more skewed
  min_per_client: 16
```

The Strategy factory in `src/fedmammo/federated/strategies/__init__.py` is the single switch point. Adding a new strategy is a matter of:

1. Subclass `flwr.server.strategy.Strategy` (or extend `FedAvg`).
2. Register the class with `@register_strategy("my_name")`.
3. Reference it by name in YAML.

---

## Metrics

Per round and per epoch:

- Accuracy
- Precision, Recall, F1 (binary, positive class = `malignant`)
- ROC-AUC
- Sensitivity (recall on malignant = true positive rate)
- Specificity (true negative rate on benign)

Logged to TensorBoard (`runs/<exp>/tb`) and CSV (`runs/<exp>/metrics.csv`).

---

## Reproducibility

`fedmammo.utils.seeding.set_global_seed(seed)` seeds Python, NumPy, PyTorch (CPU + CUDA), and sets `torch.backends.cudnn.deterministic = True`, `benchmark = False`. Full bit-level reproducibility across hardware is not guaranteed — cuDNN can introduce small non-determinism on some convolution kernels — but runs on the same machine with the same seed should match closely.

---

## What is intentionally *not* here

- No automatic download of CBIS-DDSM / VinDr-Mammo. Both require TCIA / PhysioNet access agreements; the user is expected to download manually.
- No SCAFFOLD / FedBN / FedProx training loop yet. The strategy classes and config plumbing exist; the algorithmic body is marked `TODO` with the relevant equations referenced.
- No real distributed deployment (gRPC across machines). The codebase uses Flower's simulation backend; lifting it to real gRPC is straightforward but out of scope for the first pass.

---

## License

MIT. See `pyproject.toml`.
