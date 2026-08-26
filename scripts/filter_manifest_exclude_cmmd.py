"""Drop every CMMD row from `manifests/fedmammobench.csv`, keeping the 80/10/10 split intact.

Motivation
----------
A new experiment (`exp51`+) wants CMMD excluded entirely — not in train, not in val, not in
test — while everything else about the 80/10/10 manifest stays as-is. This is a pure row filter,
not a re-split: CMMD is 4722/8341 (56.6%) of `fedmammobench.csv`, and every one of its patients is
CMMD-only (no patient straddles CMMD and another `source_dataset`, verified below), so deleting its
rows does not touch any other patient's split assignment. Unlike `resplit_manifest_70_20_10.py`,
there is no allocation problem to solve here.

Split proportions are preserved automatically, not by construction: the three remaining datasets
(cdd-cesm, inbreast, kau-bcmd) were already put through the same per-dataset 80/10/10 target split
as CMMD was, so filtering CMMD out leaves them at ~80.0/10.0/10.0% on the nose (verified in
`validate()`, not assumed).

Usage
-----
  .venv/bin/python scripts/filter_manifest_exclude_cmmd.py
  .venv/bin/python scripts/filter_manifest_exclude_cmmd.py --in manifests/fedmammobench.csv \
      --out manifests/fedmammobench_no_cmmd.csv

Output
------
  The input CSV with every `source_dataset == "cmmd"` row removed, same columns, same row order,
  `split` values untouched for every surviving row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXCLUDED_DATASET = "cmmd"


def filter_out(df: pd.DataFrame, excluded: str = EXCLUDED_DATASET) -> pd.DataFrame:
    return df.loc[df["source_dataset"] != excluded].reset_index(drop=True)


def summarize(before: pd.DataFrame, after: pd.DataFrame) -> None:
    print(f"images: {len(before)} -> {len(after)}   "
          f"patients: {before['patient_id'].nunique()} -> {after['patient_id'].nunique()}\n")

    for label, df in (("BEFORE", before), ("AFTER", after)):
        n_img = len(df)
        g = df.groupby("split").agg(images=("ID_image", "size"), patients=("patient_id", "nunique"))
        g["img_%"] = (100 * g["images"] / n_img).round(2)
        g["mal_%"] = (
            100 * df.assign(m=df["classification"] == "Malignant").groupby("split")["m"].mean()
        ).round(2)
        print(f"=== {label} ===")
        print(g.loc[["train", "val", "test"]])
        print()

    print("=== AFTER: images per (source_dataset, split) ===")
    img = pd.crosstab(after["source_dataset"], after["split"])[["train", "val", "test"]]
    print(img)


def validate(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Fail loudly rather than write a manifest that breaks an invariant."""
    assert (before["patient_id"].groupby(before["source_dataset"]).nunique().index.nunique()
            == before["source_dataset"].nunique()), "sanity check setup failed"
    straddlers = (
        before.groupby("patient_id")["source_dataset"].nunique()
    )
    assert (straddlers == 1).all(), (
        "a patient_id spans more than one source_dataset — CMMD removal would silently drop "
        "that patient's non-CMMD images too; this script assumes 1 patient = 1 dataset"
    )

    assert (after["source_dataset"] != EXCLUDED_DATASET).all(), "CMMD rows survived the filter"
    kept = before["source_dataset"] != EXCLUDED_DATASET
    assert len(after) == int(kept.sum()), "row count doesn't match the non-CMMD subset"
    assert list(before.columns) == list(after.columns), "columns changed"

    # Every surviving row must be byte-identical to its BEFORE row (order preserved, nothing
    # touched besides deleting the CMMD rows) — merge on ID_image and diff column by column.
    merged = before.loc[kept].reset_index(drop=True)
    assert (merged["ID_image"].to_numpy() == after["ID_image"].to_numpy()).all(), "row order changed"
    assert merged.equals(after), "a surviving row was modified, not just filtered"

    assert after.groupby("patient_id")["split"].nunique().max() == 1, "patient straddles splits"

    # Split proportions should land close to 80/10/10 with no re-split performed.
    props = after["split"].value_counts(normalize=True)
    assert abs(props["train"] - 0.80) < 0.01, f"train share drifted: {props['train']:.4f}"
    assert abs(props["val"] - 0.10) < 0.01, f"val share drifted: {props['val']:.4f}"
    assert abs(props["test"] - 0.10) < 0.01, f"test share drifted: {props['test']:.4f}"

    print("\nOK: CMMD fully removed, no other row touched, patients grouped, ~80/10/10 preserved.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="manifests/fedmammobench.csv", type=Path)
    ap.add_argument("--out", dest="dst", default="manifests/fedmammobench_no_cmmd.csv", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write the CSV")
    args = ap.parse_args()

    before = pd.read_csv(args.src)
    after = filter_out(before)

    summarize(before, after)
    validate(before, after)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    after.to_csv(args.dst, index=False)
    print(f"\nWrote {args.dst}")


if __name__ == "__main__":
    main()
