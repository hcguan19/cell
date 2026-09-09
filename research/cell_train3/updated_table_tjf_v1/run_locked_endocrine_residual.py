#!/usr/bin/env python3
# 锁定表格锚点上的新增字段小残差：Ridge/浅树拟合周期残差目标。
# 验证集选择修正头和eta后重拟合开发集，测试部分只预测。
"""Conservative endocrine residual on top of the locked 0.6704 table anchor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data0/guanhuancong/cell/cell_train3")
HERE = ROOT / "updated_table_tjf_v1"
CV = ROOT / "table_group_side_v1/group_cv5"
DEPLOY = ROOT / "deployable_table_distill_v1"
sys.path.insert(0, str(DEPLOY))
from train_distill import apply_locked_rule  # noqa: E402

FIELDS = [
    "内分泌检查E2", "内分泌检查-FSH", "内分泌P", "内分泌检查LH",
    "内分泌T", "内分泌检查PRL", "内分泌检查AMH",
]
CONTEXT = ["女方建档年龄", "男方建档年龄", "BMI", "Gn-AFC", "Gn启动剂量", "MII卵数"]
ETAS = (0.0, 0.05, 0.10, 0.20, 0.30)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    x = np.clip(np.asarray(x, float), -12, 12)
    return 1 / (1 + np.exp(-x))


def prepare():
    # 读取历史教师概率并应用锁定规则，形成当前残差学习的表格锚点。
    # 教师OOF和固定规则具有历史选择依赖，本函数不重新训练教师。
    rows = pd.read_csv(HERE / "clean_updated_for_training.csv", dtype={"sample_id": str, "group_id": str})
    grouped = rows.groupby("group_id", sort=False)
    meta = grouped[["fold", "pgt_type", "source", "MII卵数", "诊断与方案-诊断", "teacher_probability"]].first()
    meta["n"] = grouped.size().astype(float)
    meta["positive"] = grouped.label.sum().astype(float)
    base, _ = apply_locked_rule(
        meta.teacher_probability.to_numpy(float),
        meta[["pgt_type", "MII卵数", "诊断与方案-诊断"]],
    )
    meta["base_probability"] = base
    return rows, meta


def fit_schema(train_rows):
    # 只用当前训练样本确定类别词表，避免根据测试类别频率改变编码空间。
    categories = {}
    for col in ("pgt_type", "source"):
        counts = train_rows[col].astype("string").fillna("NA").value_counts()
        categories[col] = counts[counts >= 3].index.astype(str).tolist()
    return categories


def design(rows, meta, schema):
    grouped = rows.groupby("group_id", sort=False)
    groups = pd.Index(rows.group_id.unique())
    x = pd.DataFrame(index=groups)
    for col in FIELDS + CONTEXT:
        if col in rows:
            value = pd.to_numeric(grouped[col].first(), errors="coerce").reindex(groups)
            x[col] = value
            x[f"missing::{col}"] = value.isna().astype(float)
            x[f"log::{col}"] = np.log1p(value.clip(lower=0))
    ratios = {
        "E2_FSH": ("内分泌检查E2", "内分泌检查-FSH"),
        "LH_FSH": ("内分泌检查LH", "内分泌检查-FSH"),
        "AMH_FSH": ("内分泌检查AMH", "内分泌检查-FSH"),
        "P_E2": ("内分泌P", "内分泌检查E2"),
        "E2_AFC": ("内分泌检查E2", "Gn-AFC"),
    }
    for name, (a, b) in ratios.items():
        if a in x and b in x:
            x[f"ratio::{name}"] = x[a] / x[b].where(x[b].abs() > 1e-8)
    if "女方建档年龄" in x and "内分泌检查AMH" in x:
        x["interaction::age_AMH"] = x["女方建档年龄"] * x["内分泌检查AMH"]
    for col, values in schema.items():
        source = grouped[col].first().astype("string").fillna("NA").astype(str).reindex(groups)
        for j, value in enumerate(values):
            x[f"cat::{col}::{j}"] = source.eq(value).astype(float)
    base = meta.loc[groups, "base_probability"].to_numpy(float)
    x["base_logit"] = logit(base)
    x["base_uncertainty"] = 4 * base * (1 - base)
    x["cohort_size"] = meta.loc[groups, "n"].to_numpy(float)
    return x, meta.loc[groups]


def residual_targets(meta):
    # 两种目标：收缩阳性比例logit残差、平滑多数侧logit残差。
    # 残差限幅[-2,2]；比例目标按周期大小加权，多数侧目标按多数差额加权。
    prior = float(meta.positive.sum() / meta.n.sum())
    rate = (meta.positive.to_numpy(float) + 2 * prior) / (meta.n.to_numpy(float) + 2)
    rate_residual = np.clip(logit(rate) - logit(meta.base_probability), -2.0, 2.0)
    side = (meta.positive / meta.n >= 0.5).to_numpy(float)
    side_target = 0.8 * side + 0.1
    side_residual = np.clip(logit(side_target) - logit(meta.base_probability), -2.0, 2.0)
    impact = np.maximum(np.abs(2 * meta.positive - meta.n).to_numpy(float), 0.25)
    return {"rate": (rate_residual, meta.n.to_numpy(float)), "side": (side_residual, impact)}


def fit_heads(train_x, train_meta):
    # Ridge分支使用训练中位数填补+缺失指示+标准化；XGBoost分支原生处理NaN。
    # 每种目标各有Ridge与深度1树模型，均只拟合传入训练部分。
    heads = {}
    for target_name, (target, weight) in residual_targets(train_meta).items():
        ridge = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=25.0))
        ridge.fit(train_x, target, ridge__sample_weight=weight)
        heads[f"ridge_{target_name}"] = ridge
        tree = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=220, learning_rate=0.02,
            max_depth=1, min_child_weight=12, reg_alpha=2.0, reg_lambda=25.0,
            subsample=0.85, colsample_bytree=0.75, tree_method="hist", n_jobs=3,
            random_state=260906 + (0 if target_name == "rate" else 100),
        )
        tree.fit(train_x, target, sample_weight=weight)
        heads[f"xgb_{target_name}"] = tree
    return heads


def predict_heads(heads, x):
    out = {name: np.clip(model.predict(x), -2.0, 2.0) for name, model in heads.items()}
    out["mean_rate"] = 0.5 * (out["ridge_rate"] + out["xgb_rate"])
    out["mean_side"] = 0.5 * (out["ridge_side"] + out["xgb_side"])
    return out


def corrected(base, residual, eta):
    # 在logit空间加入有界小修正。4p(1-p)让不确定周期得到更大调整。
    # eta=0完全回退原表格概率，不表示视频信息参与了本步骤。
    uncertainty = 4 * base * (1 - base)
    return sigmoid(logit(base) + eta * uncertainty * residual)


def embryo_acc(rows, probability):
    return float(accuracy_score(rows.label, rows.group_id.map(probability) >= 0.5))


def metrics(rows, probability):
    p = rows.group_id.map(probability).to_numpy(float)
    y = rows.label.to_numpy(int)
    pred = p >= 0.5
    return {"n": int(len(rows)), "acc": float(accuracy_score(y, pred)), "bacc": float(balanced_accuracy_score(y, pred)), "auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0))}


def paired(rows, base, candidate, repeats=5000):
    y = rows.label.to_numpy(int)
    old_ok = (rows.group_id.map(base).to_numpy(float) >= 0.5) == y
    new_ok = (rows.group_id.map(candidate).to_numpy(float) >= 0.5) == y
    won, lost = int((~old_ok & new_ok).sum()), int((old_ok & ~new_ok).sum())
    groups = rows.group_id.astype(str).unique()
    pos = {g: np.flatnonzero(rows.group_id.astype(str).eq(g).to_numpy()) for g in groups}
    rng, delta = np.random.default_rng(260906), []
    for _ in range(repeats):
        idx = np.concatenate([pos[g] for g in rng.choice(groups, len(groups), replace=True)])
        delta.append(float(new_ok[idx].mean() - old_ok[idx].mean()))
    return {"won": won, "lost": lost, "delta_acc": float(new_ok.mean() - old_ok.mean()), "cycle_bootstrap_ci95": [float(v) for v in np.quantile(delta, [0.025, 0.975])], "mcnemar_exact_p": float(binomtest(min(won, lost), won + lost, 0.5).pvalue) if won + lost else 1.0}


def main():
    # 从既有train/val/test文件取样本，验证选头和eta，再用train+val重拟合。
    # 不能在此处依据测试成绩再次挑选头，否则破坏本层外折评估。
    rows, meta = prepare()
    base_all = meta.base_probability.copy()
    final = pd.Series(index=meta.index, dtype=float)
    fold_reports, choices = {}, {}
    for fold in range(5):
        train_ids = set(pd.read_csv(CV / f"fold_{fold}/train.csv", dtype={"sample_id": str}).sample_id)
        val_ids = set(pd.read_csv(CV / f"fold_{fold}/val.csv", dtype={"sample_id": str}).sample_id)
        test_ids = set(pd.read_csv(CV / f"fold_{fold}/test.csv", dtype={"sample_id": str}).sample_id)
        train, val, test = rows[rows.sample_id.isin(train_ids)], rows[rows.sample_id.isin(val_ids)], rows[rows.sample_id.isin(test_ids)]
        schema = fit_schema(train)
        train_x, train_meta = design(train, meta, schema)
        val_x, val_meta = design(val, meta, schema)
        val_heads = predict_heads(fit_heads(train_x, train_meta), val_x)
        candidates = []
        for head_name, residual in val_heads.items():
            for eta in ETAS:
                p = corrected(val_meta.base_probability.to_numpy(float), residual, eta)
                score = embryo_acc(val, pd.Series(p, index=val_x.index))
                candidates.append((score, -eta, head_name, eta))
        _, _, selected_head, selected_eta = max(candidates)
        dev = pd.concat([train, val], ignore_index=True)
        schema = fit_schema(dev)
        dev_x, dev_meta = design(dev, meta, schema)
        test_x, test_meta = design(test, meta, schema)
        test_heads = predict_heads(fit_heads(dev_x, dev_meta), test_x)
        test_probability = corrected(test_meta.base_probability.to_numpy(float), test_heads[selected_head], selected_eta)
        final.loc[test_x.index] = test_probability
        choices[str(fold)] = {"head": selected_head, "eta": float(selected_eta), "validation_acc": float(max(c[0] for c in candidates))}
        fold_reports[str(fold)] = {"base": metrics(test, base_all), "residual": metrics(test, pd.Series(test_probability, index=test_x.index))}
    summary = {
        "protocol": "locked 0.6704 anchor + updated endocrine residual; fixed outer train/val/test; validation may select eta=0",
        "base": metrics(rows, base_all), "residual": metrics(rows, final),
        "paired_vs_base": paired(rows, base_all, final), "choices": choices, "folds": fold_reports,
    }
    output = rows[["sample_id", "group_id", "fold", "label"]].copy()
    output["base_probability"] = output.group_id.map(base_all)
    output["residual_probability"] = output.group_id.map(final)
    output.to_csv(HERE / "locked_endocrine_residual_oof.csv", index=False)
    (HERE / "locked_endocrine_residual_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
