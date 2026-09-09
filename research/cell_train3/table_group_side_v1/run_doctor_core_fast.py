#!/usr/bin/env python3
# 核心字段表格专家：周期聚合、软目标收缩、类型专员与全局模型混合。
# 缺失由模型及既有字段处理，不在本归档中新增数据填补策略。
"""Fast doctor-core group-side table CV under the new final_class label."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
CV_ROOT = HERE / "group_cv5"
DOCTOR_CORE = [
    "女方建档年龄",
    "男方建档年龄",
    "内分泌检查AMH",
    "BMI",
    "内分泌检查-FSH",
    "内分泌检查E2",
    "Gn-AFC",
    "Gn启动剂量",
]
CONFIGS = [
    {"max_depth": 1, "min_child_weight": 3, "reg_alpha": 0.2, "reg_lambda": 4.0},
    {"max_depth": 2, "min_child_weight": 6, "reg_alpha": 0.5, "reg_lambda": 8.0},
    {"max_depth": 3, "min_child_weight": 6, "reg_alpha": 0.8, "reg_lambda": 10.0},
]
SHRINKAGE = [0.0, 2.0, 5.0]
MAJOR_TYPES = ("PGT-A", "PGT-M", "PGT-SR")


def metric(y, p, threshold=0.5):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def soft_logloss(target, probability, weight):
    target = np.asarray(target, float)
    probability = np.clip(np.asarray(probability, float), 1e-5, 1 - 1e-5)
    weight = np.asarray(weight, float)
    value = -(target * np.log(probability) + (1 - target) * np.log(1 - probability))
    return float(np.average(value, weights=weight))


def group_side_acc(targets, probability):
    pred_side = (np.asarray(probability) >= 0.5).astype(int)
    true_side = (targets["positive"] / targets["n"] >= 0.5).astype(int).to_numpy()
    return float((pred_side == true_side).mean())


def make_model(config, seed):
    return xgb.XGBRegressor(
        objective="reg:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.05,
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        subsample=0.9,
        colsample_bytree=0.85,
        **config,
    )


def encode_pgt(train, frames):
    values = train["pgt_type"].astype("string").str.strip().fillna("UNKNOWN")
    vocab = sorted(values.value_counts().loc[lambda s: s >= 5].index.tolist())
    cols = []
    for i, value in enumerate(vocab):
        name = f"pgt__{i:02d}"
        cols.append(name)
        for frame in frames.values():
            frame[name] = (
                frame["pgt_type"].astype("string").str.strip().fillna("UNKNOWN").eq(value).astype(float)
            )
    return cols


def build_groups(frame, feature_cols):
    work = frame.copy()
    grouped = work.groupby("group_id", sort=False)
    features = grouped[feature_cols].first().copy()
    targets = grouped["label"].agg(["sum", "size"]).rename(columns={"sum": "positive", "size": "n"})
    features["cohort_size"] = targets["n"].astype(float)
    features["log1p_cohort_size"] = np.log1p(targets["n"].astype(float))
    rows = work[["sample_id", "label", "group_id", "pgt_type", "最终分类_来自标签表"]].copy()
    return features, targets, rows


def fit_soft(train_x, train_t, val_x, val_t, fold, tag):
    prior = float(train_t.positive.sum() / max(train_t.n.sum(), 1))
    trials = []
    for strength in SHRINKAGE:
        target = (train_t.positive + strength * prior) / (train_t.n + strength)
        weight = train_t.n + strength
        for ci, config in enumerate(CONFIGS):
            seed = 51260815 + fold * 1000 + (sum(map(ord, tag)) % 50) * 10 + ci
            model = make_model(config, seed)
            model.fit(train_x.astype(np.float32), target, sample_weight=weight)
            p = model.predict(val_x.astype(np.float32)).clip(1e-5, 1 - 1e-5)
            trials.append(
                {
                    "config": config,
                    "strength": float(strength),
                    "seed": seed,
                    "val_group_side_acc": group_side_acc(val_t, p),
                    "val_logloss": soft_logloss(
                        (val_t.positive / val_t.n).to_numpy(float), p, val_t.n.to_numpy(float)
                    ),
                    "val_probability": p,
                }
            )
    return min(trials, key=lambda q: (-q["val_group_side_acc"], q["val_logloss"]))


def refit(dev_x, dev_t, test_x, choice):
    prior = float(dev_t.positive.sum() / max(dev_t.n.sum(), 1))
    strength = choice["strength"]
    target = (dev_t.positive + strength * prior) / (dev_t.n + strength)
    weight = dev_t.n + strength
    model = make_model(choice["config"], choice["seed"])
    model.fit(dev_x.astype(np.float32), target, sample_weight=weight)
    return model.predict(test_x.astype(np.float32)).clip(1e-5, 1 - 1e-5)


def run_fold(fold: int):
    frames = {
        split: pd.read_csv(CV_ROOT / f"fold_{fold}/{split}.csv")
        for split in ("train", "val", "test")
    }
    pgt_cols = encode_pgt(frames["train"], frames)
    base_cols = [c for c in DOCTOR_CORE if c in frames["train"].columns]
    feature_cols = [
        c
        for c in base_cols + pgt_cols
        if frames["train"][c].notna().any() and frames["train"][c].nunique(dropna=True) > 1
    ]
    grouped = {s: build_groups(frames[s], feature_cols) for s in frames}
    tx, tt, _ = grouped["train"]
    vx, vt, val_rows = grouped["val"]
    qx, qt, test_rows = grouped["test"]
    names = list(tx.columns)
    pgt_maps = {s: grouped[s][2].groupby("group_id")["pgt_type"].first() for s in grouped}

    val_prob = pd.Series(index=vt.index, dtype=float)
    test_prob = pd.Series(index=qt.index, dtype=float)

    choice = fit_soft(tx[names], tt, vx[names], vt, fold, "global")
    val_prob.loc[vt.index] = choice.pop("val_probability")
    dev_x = pd.concat([tx[names], vx[names]])
    dev_t = pd.concat([tt, vt])
    test_prob.loc[qt.index] = refit(dev_x, dev_t, qx[names], choice)
    global_val = val_prob.copy()
    global_test = test_prob.copy()
    meta = {"global": {k: v for k, v in choice.items() if k != "val_probability"}}

    for type_name in MAJOR_TYPES:
        tr = pgt_maps["train"].index[pgt_maps["train"].eq(type_name)]
        va = pgt_maps["val"].index[pgt_maps["val"].eq(type_name)]
        te = pgt_maps["test"].index[pgt_maps["test"].eq(type_name)]
        if min(len(tr), len(va), len(te)) < 4:
            continue
        choice = fit_soft(tx.loc[tr, names], tt.loc[tr], vx.loc[va, names], vt.loc[va], fold, f"t_{type_name}")
        type_val = choice.pop("val_probability")
        y_val = (vt.loc[va, "positive"] / vt.loc[va, "n"] >= 0.5).astype(int).to_numpy()
        g_val = global_val.loc[va].to_numpy(float)
        blends = []
        for alpha in np.arange(0.0, 1.01, 0.1):
            p = (1 - alpha) * g_val + alpha * type_val
            blends.append(
                {
                    "alpha": float(alpha),
                    "group_side_acc": float(accuracy_score(y_val, p >= 0.5)),
                    "logloss": float(log_loss(y_val, np.clip(p, 1e-5, 1 - 1e-5), labels=[0, 1])),
                }
            )
        blend = max(blends, key=lambda q: (q["group_side_acc"], -q["logloss"]))
        val_prob.loc[va] = (1 - blend["alpha"]) * g_val + blend["alpha"] * type_val
        dev_x = pd.concat([tx.loc[tr, names], vx.loc[va, names]])
        dev_t = pd.concat([tt.loc[tr], vt.loc[va]])
        type_test = refit(dev_x, dev_t, qx.loc[te, names], choice)
        test_prob.loc[te] = (1 - blend["alpha"]) * global_test.loc[te].to_numpy(float) + blend["alpha"] * type_test
        meta[f"type_{type_name}"] = {
            "choice": {k: v for k, v in choice.items() if k != "val_probability"},
            "blend": blend,
        }

    out = test_rows.copy()
    out["probability"] = test_rows["group_id"].map(test_prob)
    out["prediction"] = (out["probability"] >= 0.5).astype(int)
    out["fold"] = fold
    score = metric(out["label"], out["probability"])
    g = out.groupby("group_id").agg(y=("label", "mean"), p=("probability", "first"))
    score["group_side_acc"] = float(((g.y >= 0.5).astype(int) == (g.p >= 0.5).astype(int)).mean())
    return out, score, meta


def main():
    parts = []
    fold_scores = {}
    for fold in range(5):
        out, score, meta = run_fold(fold)
        parts.append(out)
        fold_scores[str(fold)] = {"metrics": score, "meta": meta}
        print(
            f"doctor_core fold={fold} acc={score['accuracy']:.4f} "
            f"auc={score['roc_auc']:.4f} group_side={score['group_side_acc']:.4f}",
            flush=True,
        )

    oof = pd.concat(parts, ignore_index=True)
    oof.to_csv(HERE / "oof_view_doctor_core.csv", index=False)
    overall = metric(oof["label"], oof["probability"])
    g = oof.groupby("group_id").agg(y=("label", "mean"), p=("probability", "first"))
    overall["group_side_acc"] = float(((g.y >= 0.5).astype(int) == (g.p >= 0.5).astype(int)).mean())
    maj = oof.groupby("group_id")["label"].transform(lambda s: int(s.mean() >= 0.5))
    overall["majority_oracle_acc"] = float((oof["label"] == maj).mean())
    overall["gap_to_majority_oracle"] = overall["majority_oracle_acc"] - overall["accuracy"]
    overall["label_rule"] = {
        "col": "最终分类_来自标签表",
        "positive": ["二倍体", "整倍体"],
        "negative": "all other non-missing",
        "drop_missing": True,
    }
    overall["folds"] = fold_scores

    summary = pd.DataFrame(
        [
            {
                "method": "view_doctor_core",
                "n": overall["n"],
                "accuracy": overall["accuracy"],
                "balanced_accuracy": overall["balanced_accuracy"],
                "f1": overall["f1"],
                "roc_auc": overall["roc_auc"],
                "group_side_acc": overall["group_side_acc"],
                "majority_oracle_acc": overall["majority_oracle_acc"],
                "gap_to_majority_oracle": overall["gap_to_majority_oracle"],
                "confusion_matrix": overall["confusion_matrix"],
            }
        ]
    )
    summary.to_csv(HERE / "summary_views.csv", index=False)
    summary.to_csv(HERE / "summary_table_best.csv", index=False)
    (HERE / "summary_views.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== BEST doctor_core ===", flush=True)
    print(json.dumps({k: overall[k] for k in [
        "n", "accuracy", "balanced_accuracy", "f1", "roc_auc",
        "group_side_acc", "majority_oracle_acc", "gap_to_majority_oracle",
        "confusion_matrix", "label_rule",
    ]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
