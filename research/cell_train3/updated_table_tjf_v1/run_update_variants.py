#!/usr/bin/env python3
# 更新字段与旧表格教师的历史对比入口。
# 用于追溯clean_updated_for_training的生成过程，不代表最新端到端训练。
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


ROOT = Path("/data0/guanhuancong/cell/cell_train3")
HERE = ROOT / "updated_table_tjf_v1"
BASE_CLEAN = ROOT / "table_group_side_v1/saved_best_no_mosaic_20260815/clean_table.csv"
TEACHER = ROOT / "table_group_side_v1/saved_best_no_mosaic_20260815/oof_previous_before_sr_luoshi_flip.csv"
NEW_CLEAN = HERE / "clean_updated_table.csv"
DEPLOY = ROOT / "deployable_table_distill_v1"
sys.path.insert(0, str(DEPLOY))
from train_distill import (  # noqa: E402
    apply_locked_rule,
    cycle_design,
    fit_schema,
    predict_ensemble,
    train_ensemble,
)


FIELDS = [
    "内分泌检查E2",
    "内分泌检查-FSH",
    "内分泌P",
    "内分泌检查LH",
    "内分泌T",
    "内分泌检查PRL",
    "内分泌检查AMH",
]


def canonical(value) -> str:
    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".avi", ".AVI", ".mp4", ".MP4"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return str(int(text)) if text.isdigit() else text


def load_variants() -> dict[str, pd.DataFrame]:
    old = pd.read_csv(BASE_CLEAN, dtype={"sample_id": str, "group_id": str})
    new = pd.read_csv(NEW_CLEAN, dtype={"sample_key": str})
    teacher = pd.read_csv(TEACHER, dtype={"sample_id": str})[
        ["sample_id", "probability", "fold"]
    ].rename(columns={"probability": "teacher_probability"})
    old["sample_key"] = (
        old["周期号"].astype("string").str.strip().astype(str)
        + "::"
        + old["对应的video"].map(canonical)
    )
    if old.sample_key.duplicated().any() or new.sample_key.duplicated().any():
        raise RuntimeError("duplicate normalized sample key")
    lookup = new.set_index("sample_key")
    mapped = pd.DataFrame(index=old.index)
    for column in FIELDS:
        mapped[column] = pd.to_numeric(old.sample_key.map(lookup[column]), errors="coerce")
    diagnosis = old.sample_key.map(lookup.get("诊断与方案-诊断", pd.Series(dtype=object)))

    variants: dict[str, pd.DataFrame] = {"old": old.copy()}
    full = old.copy()
    fill = old.copy()
    for column in FIELDS:
        old_value = pd.to_numeric(old[column], errors="coerce")
        full[column] = mapped[column]
        fill[column] = old_value.where(old_value.notna(), mapped[column])
    variants["full_update"] = full
    variants["fill_only"] = fill

    for base_name, base in (("full_update_masks", full), ("fill_only_masks", fill)):
        frame = base.copy()
        counts = np.zeros(len(frame), dtype=float)
        for column in FIELDS:
            old_missing = pd.to_numeric(old[column], errors="coerce").isna()
            newly_filled = old_missing & mapped[column].notna()
            frame[f"backfill_mask::{column}"] = newly_filled.astype(float)
            counts += newly_filled.to_numpy(float)
        frame["backfill_count"] = counts
        frame["any_backfill"] = (counts > 0).astype(float)
        variants[base_name] = frame

    for frame in variants.values():
        frame["诊断与方案-诊断"] = diagnosis.to_numpy()
        frame.drop(columns=["teacher_probability", "fold"], errors="ignore", inplace=True)
        frame = frame.merge(teacher, on="sample_id", how="left", validate="one_to_one")
        variants_name = next((key for key, value in variants.items() if value is variants.get(key) and value is not frame), None)
        # Assignment is handled below because merge returns a new DataFrame.
    for name in list(variants):
        frame = variants[name].drop(columns=["teacher_probability", "fold"], errors="ignore")
        variants[name] = frame.merge(teacher, on="sample_id", how="left", validate="one_to_one")
        if variants[name]["teacher_probability"].isna().any():
            raise RuntimeError(f"teacher alignment failed: {name}")
    return variants


def metric(rows: pd.DataFrame, cycle_p: pd.Series) -> dict:
    p = rows.group_id.map(cycle_p).to_numpy(float)
    y = rows.label.to_numpy(int)
    pred = p >= 0.5
    return {
        "n": int(len(rows)),
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def run(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    before = pd.Series(index=rows.group_id.unique(), dtype=float)
    final = pd.Series(index=rows.group_id.unique(), dtype=float)
    fold_metrics = {}
    for fold in range(5):
        train_rows = rows[rows.fold.ne(fold)].copy()
        test_rows = rows[rows.fold.eq(fold)].copy()
        schema = fit_schema(train_rows)
        train_x, train_meta = cycle_design(train_rows, schema)
        test_x, test_meta = cycle_design(test_rows, schema)
        models = train_ensemble(
            train_x,
            train_meta.teacher_probability.to_numpy(float),
            train_meta.n.to_numpy(float),
        )
        p = predict_ensemble(models, test_x)
        pf, acted = apply_locked_rule(p, test_meta)
        before.loc[test_x.index] = p
        final.loc[test_x.index] = pf
        fold_metrics[str(fold)] = {
            "before": metric(test_rows, pd.Series(p, index=test_x.index)),
            "final": metric(test_rows, pd.Series(pf, index=test_x.index)),
            "acted_cycles": int(acted.sum()),
        }
    return before, final, fold_metrics


def paired(rows: pd.DataFrame, reference: pd.Series, candidate: pd.Series, repeats=5000) -> dict:
    y = rows.label.to_numpy(int)
    ref_ok = (rows.group_id.map(reference).to_numpy(float) >= 0.5) == y
    can_ok = (rows.group_id.map(candidate).to_numpy(float) >= 0.5) == y
    won = int((~ref_ok & can_ok).sum())
    lost = int((ref_ok & ~can_ok).sum())
    groups = rows.group_id.astype(str).unique()
    positions = {g: np.flatnonzero(rows.group_id.astype(str).eq(g).to_numpy()) for g in groups}
    rng = np.random.default_rng(260903)
    deltas = []
    for _ in range(repeats):
        draw = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([positions[g] for g in draw])
        deltas.append(float(can_ok[idx].mean() - ref_ok[idx].mean()))
    discordant = won + lost
    return {
        "won": won,
        "lost": lost,
        "net": won - lost,
        "delta_acc": float(can_ok.mean() - ref_ok.mean()),
        "cycle_bootstrap_ci95": [float(v) for v in np.quantile(deltas, [0.025, 0.975])],
        "mcnemar_exact_p": float(binomtest(min(won, lost), discordant, 0.5).pvalue) if discordant else 1.0,
    }


def main() -> None:
    variants = load_variants()
    predictions = {}
    summary = {
        "protocol": "strict paired 5-fold distillation; same samples/folds/model/seeds; isolate full corrections, missing-only fills, and backfill provenance masks",
        "variants": {},
    }
    output = variants["old"][["sample_id", "group_id", "fold", "label"]].copy()
    for name, rows in variants.items():
        before, final, folds = run(rows)
        predictions[name] = final
        summary["variants"][name] = {
            "before": metric(rows, before),
            "final": metric(rows, final),
            "folds": folds,
        }
        output[f"probability::{name}"] = output.group_id.map(final)
    for name in variants:
        if name != "old":
            summary["variants"][name]["paired_vs_old"] = paired(
                variants[name], predictions["old"], predictions[name]
            )
    output.to_csv(HERE / "update_variants_oof.csv", index=False)
    (HERE / "update_variants_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
