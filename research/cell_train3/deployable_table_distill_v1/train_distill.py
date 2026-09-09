#!/usr/bin/env python3
# 历史表格教师蒸馏与apply_locked_rule规则函数。
# 最终新增字段残差仅复用该规则，不应称最终系统使用了这里所有蒸馏模型。
"""Distil the locked table-v2 OOF teacher into a deployable cycle model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, roc_auc_score


ROOT = Path("/data0/guanhuancong/cell/cell_train3")
HERE = ROOT / "deployable_table_distill_v1"
SAVED = ROOT / "table_group_side_v1/saved_best_no_mosaic_20260815"
CLEAN = SAVED / "clean_table.csv"
TEACHER = SAVED / "oof_previous_before_sr_luoshi_flip.csv"
RAW = Path("/data0/guanhuancong/cell/merged_two_datasets_clinical_labels.xlsx")
SEEDS = [260831, 260947, 261061, 261173, 261287]

ID_COLUMNS = {
    "sample_id", "video_id", "group_id", "label", "final_class", "source", "pgt_type",
    "对应的video", "周期号", "数据集", "整倍体与否", "最终分类_来自标签表",
}
CATEGORICAL = ["pgt_type", "女方不孕类型(原发不孕继发不孕)", "用药方案类型"]


def prepare_rows() -> pd.DataFrame:
    clean = pd.read_csv(CLEAN, dtype={"sample_id": str, "group_id": str})
    teacher = pd.read_csv(TEACHER, dtype={"sample_id": str, "group_id": str})[
        ["sample_id", "probability", "fold"]
    ].rename(columns={"probability": "teacher_probability"})
    raw = pd.read_excel(RAW)
    raw = raw.drop_duplicates(["周期号", "对应的video"], keep="first")
    diagnosis = raw[["周期号", "对应的video", "诊断与方案-诊断"]]
    frame = clean.merge(teacher, on="sample_id", how="inner", validate="one_to_one")
    frame = frame.merge(diagnosis, on=["周期号", "对应的video"], how="left", validate="many_to_one")
    if len(frame) != 1265 or frame.group_id.nunique() != 323:
        raise RuntimeError("unexpected clean cohort")
    return frame


def fit_schema(train_rows: pd.DataFrame) -> dict:
    numeric = [
        column for column in train_rows.columns
        if column not in ID_COLUMNS | set(CATEGORICAL) | {"teacher_probability", "fold", "诊断与方案-诊断"}
        and pd.to_numeric(train_rows[column], errors="coerce").notna().any()
    ]
    vocab = {}
    for column in CATEGORICAL:
        counts = train_rows[column].astype("string").fillna("NA").value_counts()
        vocab[column] = counts[counts >= 3].index.astype(str).tolist()
    return {"numeric": numeric, "vocab": vocab}


def cycle_design(rows: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = rows.groupby("group_id", sort=False)
    meta = grouped[["label", "fold", "teacher_probability", "pgt_type", "MII卵数", "诊断与方案-诊断"]].first()
    meta["n"] = grouped.size().astype(float)
    meta["positive"] = grouped.label.sum().astype(float)
    x = pd.DataFrame(index=meta.index)
    for column in schema["numeric"]:
        x[column] = pd.to_numeric(grouped[column].first(), errors="coerce")
    for column, values in schema["vocab"].items():
        source = grouped[column].first().astype("string").fillna("NA").astype(str)
        for index, value in enumerate(values):
            x[f"cat::{column}::{index}"] = source.eq(value).astype(float)
    diagnosis = meta["诊断与方案-诊断"].fillna("").astype(str)
    for token in ("罗氏", "易位", "倒位", "染色体", "地贫"):
        x[f"diagnosis::{token}"] = diagnosis.str.contains(token, regex=False).astype(float)
    x["cohort_size"] = meta["n"]
    x["log1p_cohort_size"] = np.log1p(meta["n"])
    return x.astype(np.float32), meta


def make_model(seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.025,
        max_depth=2,
        min_child_weight=6,
        reg_alpha=0.8,
        reg_lambda=12.0,
        subsample=0.9,
        colsample_bytree=0.85,
        tree_method="hist",
        n_jobs=3,
        random_state=seed,
    )


def train_ensemble(x: pd.DataFrame, target: np.ndarray, weight: np.ndarray) -> list:
    models = []
    for seed in SEEDS:
        model = make_model(seed)
        model.fit(x, target, sample_weight=weight)
        models.append(model)
    return models


def predict_ensemble(models: list, x: pd.DataFrame) -> np.ndarray:
    return np.clip(np.mean([model.predict(x) for model in models], axis=0), 1e-5, 1 - 1e-5)


def apply_locked_rule(probability: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    output = probability.copy()
    mii = pd.to_numeric(meta["MII卵数"], errors="coerce")
    diagnosis = meta["诊断与方案-诊断"].fillna("").astype(str)
    mask = (
        meta.pgt_type.astype(str).eq("PGT-SR")
        & diagnosis.str.contains("罗氏", regex=False)
        & (output < 0.5)
        & (output <= 0.48)
        & mii.le(10).fillna(False).to_numpy()
    )
    output[mask] = 1 - output[mask]
    return output, np.asarray(mask, bool)


def expand_metric(rows: pd.DataFrame, cycle_probability: pd.Series) -> dict:
    probability = rows.group_id.map(cycle_probability).to_numpy(float)
    y = rows.label.to_numpy(int)
    pred = probability >= 0.5
    return {
        "n": int(len(rows)),
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, probability)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    rows = prepare_rows()
    oof_before = pd.Series(index=rows.group_id.unique(), dtype=float)
    oof_final = pd.Series(index=rows.group_id.unique(), dtype=float)
    fold_report = {}
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
        before = predict_ensemble(models, test_x)
        final, acted = apply_locked_rule(before, test_meta)
        oof_before.loc[test_x.index] = before
        oof_final.loc[test_x.index] = final
        fold_report[str(fold)] = {
            "cycles": int(len(test_x)),
            "acted_cycles": int(acted.sum()),
            "teacher_mae": float(mean_absolute_error(test_meta.teacher_probability, before)),
            "teacher_correlation": float(np.corrcoef(test_meta.teacher_probability, before)[0, 1]),
            "before": expand_metric(test_rows, pd.Series(before, index=test_x.index)),
            "final": expand_metric(test_rows, pd.Series(final, index=test_x.index)),
        }

    teacher_cycle = rows.groupby("group_id").teacher_probability.first()
    teacher_final, teacher_acted = apply_locked_rule(
        teacher_cycle.to_numpy(float),
        rows.groupby("group_id")[["pgt_type", "MII卵数", "诊断与方案-诊断"]].first().loc[teacher_cycle.index],
    )
    teacher_final = pd.Series(teacher_final, index=teacher_cycle.index)
    summary = {
        "protocol": "strict existing five cycle folds; raw clinical fields -> five-seed XGBoost distillation of locked before-rule OOF teacher; fixed Luoshi rule",
        "identity": {"n": int(len(rows)), "cycles": int(rows.group_id.nunique())},
        "teacher_before": expand_metric(rows, teacher_cycle),
        "teacher_locked_fixed": expand_metric(rows, teacher_final),
        "student_before": expand_metric(rows, oof_before),
        "student_locked_fixed": expand_metric(rows, oof_final),
        "student_teacher_mae": float(mean_absolute_error(teacher_cycle.loc[oof_before.index], oof_before)),
        "student_teacher_correlation": float(np.corrcoef(teacher_cycle.loc[oof_before.index], oof_before)[0, 1]),
        "teacher_acted_cycles": int(teacher_acted.sum()),
        "folds": fold_report,
        "deployment_gate": {
            "minimum_acc": 0.65,
            "maximum_drop_from_teacher": 0.015,
        },
    }
    student_acc = summary["student_locked_fixed"]["acc"]
    teacher_acc = summary["teacher_locked_fixed"]["acc"]
    summary["deployment_gate"]["passed"] = bool(student_acc >= 0.65 and teacher_acc - student_acc <= 0.015)

    output = rows[["sample_id", "group_id", "fold", "label"]].copy()
    output["teacher_before_probability"] = output.group_id.map(teacher_cycle)
    output["student_before_probability"] = output.group_id.map(oof_before)
    output["student_final_probability"] = output.group_id.map(oof_final)
    output.to_csv(HERE / "oof_distilled_table.csv", index=False)
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if summary["deployment_gate"]["passed"]:
        schema = fit_schema(rows)
        x, meta = cycle_design(rows, schema)
        models = train_ensemble(x, meta.teacher_probability.to_numpy(float), meta.n.to_numpy(float))
        joblib.dump({"schema": schema, "models": models, "seeds": SEEDS}, HERE / "deployable_table_model.joblib")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
