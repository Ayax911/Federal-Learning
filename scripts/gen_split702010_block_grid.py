#!/usr/bin/env python3
"""Genera el bloque 9 (split 70/20/10) a partir de exp42.

`configs/exp42/exp42.ipynb` es la base de este bloque: una copia de exp37
(RadImageNet + cabeza MLP dropout 0.5, CE sin class_weight, WeightedRandomSampler
por celda (source_dataset x clase) con alpha=1, best-checkpoint por F1-macro,
semilla 42, img256, W&B, desglose por source_dataset) con un unico cambio real:

    MANIFEST_PATH -> manifests/fedmammobench_70_20_10.csv

mas dos campos de procedencia (SPLIT_SCHEME) que se registran en la config de
W&B y en metrics.json, para que un run del bloque 9 nunca se confunda con uno
del bloque 8 al leerlos despues.

Por que el split nuevo: los bloques 6-8 corrieron sobre 80/10/10, con val en
254 pacientes / 836 imagenes. Sobre esa val se decide DOS cosas — el
early-stopping y el best-checkpoint, ambas por F1-macro — y con 836 imagenes
ese criterio es ruidoso: en el bloque 8 la mejor epoca cae entre la 12 y la 36
sin patron claro respecto a la profundidad. El split 70/20/10 duplica val (507
pacientes / 1668 imagenes) a costa de train (2027 -> 1774 pacientes), asi que
la seleccion de modelo se vuelve mas estable a cambio de ~12% menos imagenes
de entrenamiento. Cual de los dos efectos gana es exactamente lo que mide este
bloque.

El TEST no cambio: son las mismas 834 imagenes / 253 pacientes que en los
bloques 6-8 (scripts/resplit_manifest_70_20_10.py solo mueve pacientes de train
a val y valida esa invariante). Por eso las metricas de test del bloque 9 son
comparables 1-a-1 contra el bloque 8, celda a celda de la rejilla. Las de val
NO lo son — val es otro conjunto, mas grande.

El bloque 9 mantiene todo lo demas fijo y solo varia la profundidad de
descongelamiento, la misma grilla de los bloques 6, 7 y 8. La numeracion esta
alineada con el bloque 8 (exp42+n corresponde a exp37+n) para que la tabla
comparativa sea directa:

    exp42  UNFREEZE_IDX=[7]            layer4               (== exp37 + split 70/20/10)
    exp43  UNFREEZE_IDX=[]             backbone congelado   (== exp38 + split 70/20/10)
    exp44  UNFREEZE_IDX=[7,6]          layer4+layer3        (== exp39 + split 70/20/10)
    exp45  UNFREEZE_IDX=[7,6,5]        layer4+layer3+layer2 (== exp40 + split 70/20/10)
    exp46  UNFREEZE_IDX=[7,6,5,4,1,0]  todo el backbone     (== exp41 + split 70/20/10)

exp42 no se regenera (esta editado a mano y es el template); este script solo
escribe exp43-exp46. La unica celda que cambia entre las 5 notebooks es la de
parametros — la celda de freeze, la del optimizador y la del sampler ya son
dinamicas sobre UNFREEZE_IDX / SAMPLER_ALPHA, y la de datos deriva todo de
MANIFEST_PATH.

Indices de backbone (nn.Sequential, ver backbone_remap en
load_radimagenet_backbone dentro del notebook): 0=conv1, 1=bn1, 4=layer1,
5=layer2, 6=layer3, 7=layer4 (2=relu/3=maxpool/8=avgpool no tienen parametros).

Al leer los resultados: el sampler sigue entrenando a prevalencia 50/50, asi
que valen las mismas advertencias del bloque 8 — la accuracy de kau-bcmd baja y
eso no es una regresion; compara AUC y AUC macro. Y ojo con la comparacion
contra el bloque 8: si el bloque 9 mejora, hay que separar "val mas grande
selecciona mejor" de "train mas chico entrena peor", que empujan en sentidos
opuestos.

Uso:
    python scripts/gen_split702010_block_grid.py            # escribe configs/exp4{3,4,5,6}/
    python scripts/gen_split702010_block_grid.py --dry-run  # solo lista lo que haria
    python scripts/gen_split702010_block_grid.py --force    # sobreescribe si ya existen
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "exp42" / "exp42.ipynb"
PARAMS_CELL_ID = "exp22-cell-02"

EXPERIMENTS = [
    dict(exp="exp43", unfreeze_idx=[], slug="frozen",
         desc="backbone completamente congelado",
         note="Profundidad 0/4: control con el backbone totalmente congelado."),
    dict(exp="exp44", unfreeze_idx=[7, 6], slug="layer4_layer3",
         desc="última y penúltima capa (layer4 + layer3) descongeladas",
         note="Profundidad 2/4: layer4 + layer3."),
    dict(exp="exp45", unfreeze_idx=[7, 6, 5], slug="layer4_layer3_layer2",
         desc="últimas tres capas (layer4 + layer3 + layer2) descongeladas",
         note="Profundidad 3/4: layer4 + layer3 + layer2."),
    dict(exp="exp46", unfreeze_idx=[7, 6, 5, 4, 1, 0], slug="all_unfrozen",
         desc="todo el backbone descongelado",
         note="Profundidad 4/4: todo el backbone (conv1/bn1/layer1-4) entrenable."),
]


def params_cell(spec: dict) -> str:
    run_name = (
        f"{spec['exp']}_resnet50_radimagenet_mammobench_{spec['slug']}"
        "_mlp05_ce_lr1e-4_img256_bs16_balsampler_split702010"
    )
    return f"""EXP_ID = "{spec['exp']}"
RUN_NAME = "{run_name}"
UNFROZEN_DESC = "{spec['desc']}"
BLOCK = 9  # bloque 9: bloque 8 (sampler balanceado) re-corrido sobre el split 70/20/10
# Índices de backbone (nn.Sequential) a descongelar — ver freeze cell para el mapa completo.
UNFREEZE_IDX = {spec['unfreeze_idx']!r}
HEAD = "mlp"

# alpha del muestreador: w_i = n_celda ** (-SAMPLER_ALPHA). Con 1.0 las 8
# celdas (4 bases x 2 clases) quedan equiprobables y P(maligno|dataset)=0.5 en
# todas; con 0.0 se recupera el muestreo natural de los bloques 6/7.
SAMPLER_ALPHA = 1.0

LR = 1e-4            # cabeza (linear1/bn1/linear2)
LR_BACKBONE = 1e-4   # backbone descongelado — igual a LR y al del bloque 8, para que la rejilla compare 1-a-1
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

# Split 70/20/10 (bloques 6-8 usaban 80/10/10 vía manifests/fedmammobench.csv).
# Generado por scripts/resplit_manifest_70_20_10.py: el TEST es bit a bit el
# mismo (mismas 834 imágenes / 253 pacientes), así que los números de test SÍ
# son comparables 1-a-1 contra el bloque 8. Lo que cambió es que val pasó de
# 254 a 507 pacientes (836 -> 1668 imágenes) a costa de train (2027 -> 1774
# pacientes), moviendo 253 pacientes enteros de train a val — ningún paciente
# cruza splits. Con val al doble, el early-stopping por F1-macro y la elección
# de best-checkpoint son bastante menos ruidosos; a cambio hay ~12% menos
# imágenes de entrenamiento.
SPLIT_SCHEME = "70/20/10"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "fedmammobench_70_20_10.csv"
RADIMAGENET_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "RadImageNet-resnet50.pth"

# W&B: la API key NUNCA se pega aquí. Se lee de la variable de entorno
# WANDB_API_KEY, o de las credenciales cacheadas en ~/.netrc tras correr
# `wandb login` una vez en este workstation (a diferencia de los
# contenedores Docker del paquete, aqui el notebook SI hereda ese
# ~/.netrc porque corre directo en el host, sin aislar el filesystem).
WANDB_PROJECT = "fedmammobench"
WANDB_ENTITY = None  # None -> entity por defecto de la API key
WANDB_MODE = "online"  # "online" | "offline" | "disabled"
WANDB_TAGS = ["notebook", "centralized", "radimagenet", "mammobench", "{spec['slug']}", "img256", "balanced-sampler", "split702010"]

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
    parser = argparse.ArgumentParser(description=__doc__)
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
