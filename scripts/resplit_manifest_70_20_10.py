"""Re-split `manifests/fedmammobench.csv` from 80/10/10 to 70/20/10.

Hard constraints
----------------
1. **The test split is untouched.** Every image that is `split == "test"` in the input stays
   `test` in the output, and no other image becomes `test`. Test is therefore already fixed at
   253 patients (9.98% of 2534) — the 10% leg of the target is whatever test already is.
2. **Patient counts are exact.** 70/20/10 is enforced on *patients*, per `source_dataset`, by
   allocating `round(0.70 * n_patients)` to train and giving the remainder to val. Summed over
   the four datasets this lands on 1774/507/253 = 70.01/20.01/9.98%.
3. **Patients never straddle splits** (the input already satisfies this; the output preserves it
   because every decision is made on whole patients).

Soft objective
--------------
Image counts can only follow patient counts approximately — patients carry 1..16 images each.
Within the exact patient quota we pick *which* patients move so that the resulting val split is
as close as possible to 20% of that dataset's images while keeping its malignant image rate close
to the dataset's own rate. Both terms are optimized by seeded local search (swap one moved patient
for one not-moved, keep the swap if it lowers the cost) — see `_select_promotions`.

Least-churn policy: the existing val patients stay in val, and the quota is filled by promoting
patients from train. Nothing moves val -> train. So the whole diff vs. the input manifest is
"253 patients changed train -> val", which keeps the change auditable.

Usage
-----
  .venv/bin/python scripts/resplit_manifest_70_20_10.py
  .venv/bin/python scripts/resplit_manifest_70_20_10.py --in manifests/fedmammobench.csv \
      --out manifests/fedmammobench_70_20_10.csv --seed 42

Output
------
  The full input CSV, same rows in the same order, same columns — only the `split` column differs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.20
# test is not a parameter: it is whatever the input already marked as test.

# Relative weight of the malignant-rate term against the image-count term in the cost.
# 1.0 = a 1-percentage-point drift in malignant rate costs the same as 1% of the image target.
CLASS_BALANCE_WEIGHT = 1.0

SWAP_PASSES = 200


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: its dataset, current split, image count, malignant image count.

    Patients are mixed-class here (883 of 2534 carry both labels), so a patient is summarized by
    *how many* malignant images it has, not by a single class label.
    """
    tab = (
        df.assign(_mal=(df["classification"] == "Malignant").astype(int))
        .groupby("patient_id")
        .agg(
            source_dataset=("source_dataset", "first"),
            split=("split", "first"),
            n_images=("ID_image", "size"),
            n_malignant=("_mal", "sum"),
        )
        .reset_index()
    )
    assert df.groupby("patient_id")["split"].nunique().max() == 1, "patient straddles splits in input"
    return tab


def _cost(img: float, mal: float, target_img: float, target_rate: float) -> float:
    """Deviation of a candidate val split from its image-count and malignant-rate targets."""
    img_term = abs(img - target_img) / max(target_img, 1.0)
    rate = mal / img if img else 0.0
    return img_term + CLASS_BALANCE_WEIGHT * abs(rate - target_rate)


def _select_promotions(
    candidates: pd.DataFrame,
    n_promote: int,
    base_img: int,
    base_mal: int,
    target_img: float,
    target_rate: float,
    rng: np.random.Generator,
) -> list[str]:
    """Pick exactly `n_promote` of `candidates` (current train patients) to move into val.

    `base_img`/`base_mal` are what val already holds before the promotion. Seeded random start,
    then hill-climbing on single swaps: the cardinality constraint is structural (a swap never
    changes the count), so the search only ever moves along feasible states.
    """
    if n_promote <= 0:
        return []

    ids = candidates["patient_id"].to_numpy()
    imgs = candidates["n_images"].to_numpy()
    mals = candidates["n_malignant"].to_numpy()

    order = rng.permutation(len(ids))
    chosen = np.zeros(len(ids), dtype=bool)
    chosen[order[:n_promote]] = True

    def state(mask: np.ndarray) -> tuple[float, float]:
        return base_img + imgs[mask].sum(), base_mal + mals[mask].sum()

    best = _cost(*state(chosen), target_img, target_rate)

    for _ in range(SWAP_PASSES):
        improved = False
        inside = np.flatnonzero(chosen)
        outside = np.flatnonzero(~chosen)
        # Deterministic given the seed; shuffling keeps the scan from favouring low indices.
        rng.shuffle(inside)
        rng.shuffle(outside)
        for i in inside:
            for o in outside:
                if imgs[i] == imgs[o] and mals[i] == mals[o]:
                    continue  # swap is a no-op for the objective
                chosen[i], chosen[o] = False, True
                cand = _cost(*state(chosen), target_img, target_rate)
                if cand < best - 1e-12:
                    best = cand
                    improved = True
                    break
                chosen[i], chosen[o] = True, False
            if improved:
                break
        if not improved:
            break

    return ids[chosen].tolist()


def resplit(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (new manifest, per-dataset report). Only the `split` column changes."""
    patients = _patient_table(df)
    rng = np.random.default_rng(seed)

    new_split = dict(zip(patients["patient_id"], patients["split"]))
    report = []

    for dataset, grp in patients.groupby("source_dataset", sort=True):
        n_pat = len(grp)
        n_test = int((grp["split"] == "test").sum())
        # Exact patient quota. Train is rounded; val absorbs the remainder so the three legs
        # always sum back to n_pat with no patient left unassigned.
        n_train = round(TRAIN_FRACTION * n_pat)
        n_val = n_pat - n_test - n_train

        pool = grp[grp["split"] != "test"]
        cur_val = pool[pool["split"] == "val"]
        cur_train = pool[pool["split"] == "train"]
        n_promote = n_val - len(cur_val)
        if n_promote < 0:
            raise ValueError(
                f"{dataset}: target val ({n_val}) is smaller than the existing val ({len(cur_val)}); "
                "the promote-only policy cannot shrink val."
            )

        ds_images = int(grp["n_images"].sum())
        ds_malignant = int(grp["n_malignant"].sum())
        target_img = VAL_FRACTION * ds_images
        target_rate = ds_malignant / ds_images

        promoted = _select_promotions(
            cur_train,
            n_promote,
            base_img=int(cur_val["n_images"].sum()),
            base_mal=int(cur_val["n_malignant"].sum()),
            target_img=target_img,
            target_rate=target_rate,
            rng=rng,
        )
        for pid in promoted:
            new_split[pid] = "val"

        report.append(
            {
                "source_dataset": dataset,
                "patients": n_pat,
                "train_pat": n_train,
                "val_pat": n_val,
                "test_pat": n_test,
                "promoted": len(promoted),
                "target_val_img": round(target_img, 1),
            }
        )

    out = df.copy()
    out["split"] = out["patient_id"].map(new_split)
    return out, pd.DataFrame(report)


def summarize(before: pd.DataFrame, after: pd.DataFrame) -> None:
    n_img, n_pat = len(after), after["patient_id"].nunique()
    print(f"images: {n_img}   patients: {n_pat}\n")

    for label, df in (("BEFORE", before), ("AFTER", after)):
        g = df.groupby("split").agg(images=("ID_image", "size"), patients=("patient_id", "nunique"))
        g["img_%"] = (100 * g["images"] / n_img).round(2)
        g["pat_%"] = (100 * g["patients"] / n_pat).round(2)
        g["mal_%"] = (
            100 * df.assign(m=df["classification"] == "Malignant").groupby("split")["m"].mean()
        ).round(2)
        print(f"=== {label} ===")
        print(g.loc[["train", "val", "test"]])
        print()

    print("=== AFTER: patients per (source_dataset, split) ===")
    pat = after.groupby(["source_dataset", "split"])["patient_id"].nunique().unstack()[
        ["train", "val", "test"]
    ]
    print(pat.assign(**{"val_%": (100 * pat["val"] / pat.sum(axis=1)).round(2)}))
    print()
    print("=== AFTER: images per (source_dataset, split) ===")
    img = pd.crosstab(after["source_dataset"], after["split"])[["train", "val", "test"]]
    print(img.assign(**{"val_%": (100 * img["val"] / img.sum(axis=1)).round(2)}))


def validate(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Fail loudly rather than write a manifest that breaks an invariant."""
    assert len(before) == len(after), "row count changed"
    assert (before["ID_image"].to_numpy() == after["ID_image"].to_numpy()).all(), "row order changed"
    assert list(before.columns) == list(after.columns), "columns changed"

    changed = before.columns.difference(["split"])
    assert before[changed].equals(after[changed]), "a column other than `split` changed"

    b_test = set(before.loc[before["split"] == "test", "ID_image"])
    a_test = set(after.loc[after["split"] == "test", "ID_image"])
    assert b_test == a_test, "TEST SET CHANGED"

    assert after["split"].isin(["train", "val", "test"]).all(), "unknown split label"
    assert after["split"].notna().all(), "unassigned rows"
    assert after.groupby("patient_id")["split"].nunique().max() == 1, "patient straddles splits"

    # Nothing may leave val or test; the only legal transition is train -> val.
    moves = pd.crosstab(before["split"], after["split"])
    illegal = [
        (a, b) for a in moves.index for b in moves.columns if a != b and (a, b) != ("train", "val")
    ]
    for a, b in illegal:
        assert moves.loc[a, b] == 0, f"illegal transition {a} -> {b} ({moves.loc[a, b]} images)"

    print("\nOK: test identical, patients grouped, only train -> val transitions.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="manifests/fedmammobench.csv", type=Path)
    ap.add_argument("--out", dest="dst", default="manifests/fedmammobench_70_20_10.csv", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write the CSV")
    args = ap.parse_args()

    before = pd.read_csv(args.src)
    after, report = resplit(before, seed=args.seed)

    print("=== per-dataset patient quota ===")
    print(report.to_string(index=False))
    print()
    summarize(before, after)
    validate(before, after)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    after.to_csv(args.dst, index=False)
    print(f"\nWrote {args.dst}")


if __name__ == "__main__":
    main()
