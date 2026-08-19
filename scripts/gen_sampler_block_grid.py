#!/usr/bin/env python3
"""Genera el bloque 8 (muestreo balanceado por base) a partir de exp37.

`configs/exp37/exp37.ipynb` es la base de este bloque: una copia de exp36
(RadImageNet + cabeza MLP dropout 0.5, CE, best-checkpoint por F1-macro,
semilla 42, img256, W&B, desglose por source_dataset) con tres cambios:

    - WeightedRandomSampler sobre celdas (source_dataset x clase) en el
      DataLoader de train, con w_i = n_celda ** (-SAMPLER_ALPHA)
    - fuera el class_weight de la loss (exp34-exp36 lo usaban): con el sampler
      a alpha=1 la distribucion ya es 50/50 DENTRO de cada base, asi que los
      pesos inversos a la frecuencia global contarian el balanceo dos veces y
      sobreponderarian maligno ~1.93x
    - LR = LR_BACKBONE = 1e-4 (exp36 tenia 4e-4, seguro solo porque su backbone
      estaba congelado; aqui 4 de las 5 corridas lo descongelan). 1e-4 es ademas
      el valor del bloque 7, de modo que esta rejilla compara 1-a-1 contra
      exp28-exp32 y el sampler queda como la unica variable relevante.

Por que el sampler: en train, P(maligno | source_dataset) va de 0.046
(kau-bcmd) a 0.489 (cmmd). Identificar la base de origen es trivial para la red
(escaner, preprocesado, histograma) y reduce la loss sin mirar la lesion. En los
resultados del bloque 7 eso es medible: sobre las imagenes BENIGNAS del test, la
probabilidad media que emite el modelo correlaciona r=0.98 (Spearman 1.00) con
la prevalencia de su base. Un clasificador que solo conoce la base de origen y
emite su prevalencia obtiene AUC 0.726 en este test set, y los modelos con
backbone congelado no lo superan en AUC macro (0.719-0.739). Con alpha=1 las 8
celdas quedan equiprobables, la informacion mutua entre base y etiqueta es 0, y
el atajo se queda sin gradiente.

El bloque 8 mantiene TODO eso fijo y solo varia la profundidad de
descongelamiento, la misma grilla de los bloques 6 y 7:

    exp37  UNFREEZE_IDX=[7]            layer4               (== exp28 + sampler)
    exp38  UNFREEZE_IDX=[]             backbone congelado   (== exp29 + sampler)
    exp39  UNFREEZE_IDX=[7,6]          layer4+layer3        (== exp30 + sampler)
    exp40  UNFREEZE_IDX=[7,6,5]        layer4+layer3+layer2 (== exp31 + sampler)
    exp41  UNFREEZE_IDX=[7,6,5,4,1,0]  todo el backbone     (== exp32 + sampler)

exp37 no se regenera (esta editado a mano y es el template); este script solo
escribe exp38-exp41. La unica celda que cambia entre las 5 notebooks es la de
parametros — la celda de freeze, la del optimizador y la del sampler ya son
dinamicas sobre UNFREEZE_IDX / SAMPLER_ALPHA.

Indices de backbone (nn.Sequential, ver backbone_remap en
load_radimagenet_backbone dentro del notebook): 0=conv1, 1=bn1, 4=layer1,
5=layer2, 6=layer3, 7=layer4 (2=relu/3=maxpool/8=avgpool no tienen parametros).

Al leer los resultados: el sampler entrena a prevalencia 50/50, asi que el
modelo queda calibrado para ese mundo y sobre-predice maligno en las bases de
baja prevalencia. La accuracy de kau-bcmd VA A BAJAR respecto al bloque 7 y eso
no es una regresion — compara AUC y AUC macro, y elige el umbral sobre
validacion (por nodo) antes de mirar F1.

Uso:
    python scripts/gen_sampler_block_grid.py            # escribe configs/exp3{8,9}/, exp4{0,1}/
    python scripts/gen_sampler_block_grid.py --dry-run  # solo lista lo que haria
    python scripts/gen_sampler_block_grid.py --force    # sobreescribe si ya existen
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "exp37" / "exp37.ipynb"
PARAMS_CELL_ID = "exp22-cell-02"

EXPERIMENTS = [
    dict(exp="exp38", unfreeze_idx=[], slug="frozen",
         desc="backbone completamente congelado",
         note="Profundidad 0/4: control con el backbone totalmente congelado."),
    dict(exp="exp39", unfreeze_idx=[7, 6], slug="layer4_layer3",
         desc="última y penúltima capa (layer4 + layer3) descongeladas",
         note="Profundidad 2/4: layer4 + layer3."),
    dict(exp="exp40", unfreeze_idx=[7, 6, 5], slug="layer4_layer3_layer2",
         desc="últimas tres capas (layer4 + layer3 + layer2) descongeladas",
         note="Profundidad 3/4: layer4 + layer3 + layer2."),
    dict(exp="exp41", unfreeze_idx=[7, 6, 5, 4, 1, 0], slug="all_unfrozen",
         desc="todo el backbone descongelado",
         note="Profundidad 4/4: todo el backbone (conv1/bn1/layer1-4) entrenable."),
]


def params_cell(spec: dict) -> str:
    run_name = (
        f"{spec['exp']}_resnet50_radimagenet_mammobench_{spec['slug']}"
        "_mlp05_ce_lr1e-4_img256_bs16_balsampler"
    )
    return f"""EXP_ID = "{spec['exp']}"
RUN_NAME = "{run_name}"
UNFROZEN_DESC = "{spec['desc']}"
BLOCK = 8  # bloque 8: muestreo balanceado por (source_dataset x clase) sobre la rejilla de profundidad del bloque 7
# Índices de backbone (nn.Sequential) a descongelar — ver freeze cell para el mapa completo.
UNFREEZE_IDX = {spec['unfreeze_idx']!r}
HEAD = "mlp"

# alpha del muestreador: w_i = n_celda ** (-SAMPLER_ALPHA). Con 1.0 las 8
# celdas (4 bases x 2 clases) quedan equiprobables y P(maligno|dataset)=0.5 en
# todas; con 0.0 se recupera el muestreo natural de los bloques 6/7.
SAMPLER_ALPHA = 1.0

LR = 1e-4            # cabeza (linear1/bn1/linear2)
LR_BACKBONE = 1e-4   # backbone descongelado — igual a LR y al del bloque 7, para que la rejilla compare 1-a-1
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
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "fedmammobench.csv"
RADIMAGENET_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "RadImageNet-resnet50.pth"

# W&B: la API key NUNCA se pega aquí. Se lee de la variable de entorno
# WANDB_API_KEY, o de las credenciales cacheadas en ~/.netrc tras correr
# `wandb login` una vez en este workstation (a diferencia de los
# contenedores Docker del paquete, aqui el notebook SI hereda ese
# ~/.netrc porque corre directo en el host, sin aislar el filesystem).
WANDB_PROJECT = "fedmammobench"
WANDB_ENTITY = None  # None -> entity por defecto de la API key
WANDB_MODE = "online"  # "online" | "offline" | "disabled"
WANDB_TAGS = ["notebook", "centralized", "radimagenet", "mammobench", "{spec['slug']}", "img256", "balanced-sampler"]

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
