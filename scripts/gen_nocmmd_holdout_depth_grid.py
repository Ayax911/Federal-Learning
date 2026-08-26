#!/usr/bin/env python3
"""Genera el bloque 11 (sin CMMD + warm-start holdout_best.pt) a partir de exp51.

`configs/exp51/exp51.ipynb` es la base de este bloque: ResNet50, manifest
`manifests/fedmammobench_no_cmmd.csv` (80/10/10 igual que fedmammobench.csv, pero
con CMMD eliminado de train/val/test por completo — ver
scripts/filter_manifest_exclude_cmmd.py), backbone Y cabeza calientes desde
`weights/holdout_best.pt` (no solo backbone como en el resto de la serie — la
cabeza ya viene entrenada, no se reinicializa con normal_(0, 0.01)), sin
sampler (CrossEntropyLoss ponderada por clase en su lugar, igual que exp33/34),
best-checkpoint/early-stopping por F1-macro, semilla 42, img256, W&B, desglose
por source_dataset (3 bases: cdd-cesm/inbreast/kau-bcmd).

exp51 ya cubre `UNFREEZE_IDX=[7,6,5]` (layer4+layer3+layer2). Este script
completa 3 puntos más de la misma escalera de profundidad que usan los bloques
6-9 (frozen / layer4 / layer4+layer3 / layer4+layer3+layer2 / todo el
backbone) — deliberadamente sin generar el punto "layer4+layer3" (4/5 puntos
cubiertos, no los 5):

    exp51  UNFREEZE_IDX=[7,6,5]          layer4+layer3+layer2  (ya existe, template)
    exp52  UNFREEZE_IDX=[]               backbone congelado
    exp53  UNFREEZE_IDX=[7]              solo layer4
    exp54  UNFREEZE_IDX=[7,6,5,4,1,0]    todo el backbone

exp51 no se regenera (está editado a mano y es el template); este script solo
escribe exp52-exp54. La única celda que cambia entre las cuatro notebooks es la
de parámetros — las celdas de freeze, optimizer, class_weight y carga de pesos
ya son dinámicas sobre UNFREEZE_IDX / requires_grad y no necesitan tocarse.

Índices de backbone (nn.Sequential, ver backbone_remap en
load_holdout_backbone dentro del notebook): 0=conv1, 1=bn1, 4=layer1,
5=layer2, 6=layer3, 7=layer4 (2=relu/3=maxpool/8=avgpool no tienen parámetros).

Uso:
    python scripts/gen_nocmmd_holdout_depth_grid.py            # escribe configs/exp5{2,3,4}/
    python scripts/gen_nocmmd_holdout_depth_grid.py --dry-run  # solo lista qué haría
    python scripts/gen_nocmmd_holdout_depth_grid.py --force    # sobreescribe si ya existen
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "exp51" / "exp51.ipynb"
PARAMS_CELL_ID = "exp22-cell-02"

EXPERIMENTS = [
    dict(exp="exp52", unfreeze_idx=[], slug="frozen",
         desc="backbone completamente congelado",
         note="Profundidad 1/4: control con el backbone totalmente congelado."),
    dict(exp="exp53", unfreeze_idx=[7], slug="layer4",
         desc="solo la última capa (layer4) descongelada",
         note="Profundidad 2/4: solo layer4 — el punto de partida estándar del resto de la serie."),
    dict(exp="exp54", unfreeze_idx=[7, 6, 5, 4, 1, 0], slug="allunfrozen",
         desc="todo el backbone descongelado",
         note="Profundidad 4/4: todo el backbone (conv1/bn1/layer1-4) entrenable."),
]


def params_cell(spec: dict) -> str:
    run_name = (
        f"{spec['exp']}_resnet50_holdoutwarmstart_mammobench_nocmmd_{spec['slug']}"
        "_mlp05_cebalanced_lr1e-4_img256_bs16_split801010"
    )
    return f"""EXP_ID = "{spec['exp']}"
RUN_NAME = "{run_name}"
UNFROZEN_DESC = "{spec['desc']}"
BLOCK = 11  # bloque 11: split 80/10/10 SIN CMMD (manifests/fedmammobench_no_cmmd.csv,
            # generado por scripts/filter_manifest_exclude_cmmd.py) + warm-start desde
            # weights/holdout_best.pt (backbone Y cabeza, no solo backbone como en el
            # resto de la serie -- viene de otro experimento en esta máquina, origen
            # exacto sin identificar) + sin sampler, CE ponderada por clase en su lugar
            # (ver celda de class_weight). Misma escalera de profundidad que los bloques
            # 6-9, no una rejilla completa (falta el punto layer4+layer3).
# Índices de backbone (nn.Sequential) a descongelar — ver freeze cell para el mapa completo.
UNFREEZE_IDX = {spec['unfreeze_idx']!r}
HEAD = "mlp"

LR = 1e-4            # cabeza (linear1/bn1/linear2/bn2/linear3)
LR_BACKBONE = 1e-4   # backbone descongelado — igual a LR, consistente con el resto de la serie
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
PATIENCE = 20
IMAGE_SIZE = 256
ROTATION_DEGREES = 7

PROJECT_ROOT = Path(
    "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/"
    "Federal Learning/infraestructura federada/Federal-Learning"
)

DATASET_ROOT = Path(
    "/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/"
    "c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_"
    "DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench"
)

# 80/10/10, igual que fedmammobench.csv (bloques 6-8), pero con TODAS las filas de
# CMMD eliminadas de train/val/test -- no solo de test, de las tres. Generado por
# scripts/filter_manifest_exclude_cmmd.py (filtro puro, no re-split: los otros 3
# datasets ya estaban al 80/10/10 dentro de sí mismos, así que quitar CMMD entero deja
# la proporción global intacta sin tocar ningún split de ningún paciente). Efecto
# grande a tener en cuenta: CMMD era ~56.6% de las imágenes y el dataset con más tasa
# de maligno (~49%), así que sin él el train pasa de 34.1% a 14.9% maligno -- un
# desbalance bastante más fuerte que el resto de la serie (ver celda de class_weight).
SPLIT_SCHEME = "80/10/10 (sin CMMD)"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "fedmammobench_no_cmmd.csv"

# Backbone Y cabeza calientes desde otro experimento en esta máquina (origen exacto
# sin identificar), NO desde RadImageNet-resnet50.pth como el resto de la serie. Trae
# backbone.{{0,1,4-7}} (misma convención nn.Sequential que RadImageNet-resnet50.pth) +
# fc.{{0,1,4,5,8}} -- una cabeza Linear/BN/ReLU/Dropout x2 + Linear final que coincide
# EXACTO en forma con linear1/bn1/linear2/bn2/linear3 de FullModel (ver celda de carga
# de pesos, más abajo).
HOLDOUT_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "holdout_best.pt"

# W&B: la API key NUNCA se pega aquí. Se lee de la variable de entorno
# WANDB_API_KEY, o de las credenciales cacheadas en ~/.netrc tras correr
# `wandb login` una vez en este workstation (a diferencia de los
# contenedores Docker del paquete, aqui el notebook SI hereda ese
# ~/.netrc porque corre directo en el host, sin aislar el filesystem).
WANDB_PROJECT = "fedmammobench"
WANDB_ENTITY = None  # None -> entity por defecto de la API key
WANDB_MODE = "online"  # "online" | "offline" | "disabled"
WANDB_TAGS = ["notebook", "centralized", "holdout-warmstart", "mammobench", "no-cmmd", "{spec['slug']}", "img256", "class-weighted-ce", "split801010"]

RUN_DIR = PROJECT_ROOT / "runs" / RUN_NAME
PLOTS_DIR = RUN_DIR / "plots"
# Métricas/gráficas globales (todo el test set) van en PLOTS_DIR y metrics.json.
# El desglose por source_dataset va aparte, en su propia carpeta, para no
# mezclar los dos niveles de granularidad.
PER_DATASET_DIR = RUN_DIR / "per_dataset"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PER_DATASET_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = RUN_DIR / "best_model.pth"
LAST_MODEL_PATH = RUN_DIR / "last_model.pth"

print(f"RUN_DIR: {{RUN_DIR}}")"""


def build_notebook(template: dict, spec: dict) -> dict:
    nb = copy.deepcopy(template)
    found = False
    for cell in nb["cells"]:
        if cell.get("id") == PARAMS_CELL_ID:
            lines = params_cell(spec).split("\n")
            cell["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
            found = True
        if cell.get("cell_type") == "code":
            # cada notebook del bloque arranca limpio, aunque el template tenga
            # alguna celda suelta ejecutada mientras se editaba.
            cell["outputs"] = []
            cell["execution_count"] = None
    if not found:
        raise RuntimeError(f"No se encontró la celda de parámetros ({PARAMS_CELL_ID}) en el template.")
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Solo lista qué se generaría.")
    parser.add_argument("--force", action="store_true", help="Sobreescribe notebooks existentes.")
    args = parser.parse_args()

    template = json.loads(TEMPLATE.read_text())

    for spec in EXPERIMENTS:
        out_dir = REPO / "configs" / spec["exp"]
        out_path = out_dir / f"{spec['exp']}.ipynb"
        print(f"{spec['exp']}: {spec['note']}")
        print(f"  UNFREEZE_IDX={spec['unfreeze_idx']} -> {out_path.relative_to(REPO)}")

        if args.dry_run:
            continue
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} ya existe (usa --force para sobreescribir).")

        nb = build_notebook(template, spec)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

    if args.dry_run:
        print("\n(dry-run: no se escribió nada)")
    else:
        print(f"\n{len(EXPERIMENTS)} notebooks escritos.")


if __name__ == "__main__":
    main()
