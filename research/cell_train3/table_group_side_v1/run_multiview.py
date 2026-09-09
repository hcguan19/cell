#!/usr/bin/env python3
# 四路周期表格专家和折外LogReg门控的历史上游实现。
# 需结合已锁定教师产物及后处理版本复现，不能仅重跑此入口保证历史最优值。
"""Multi-view group-side table stacking on the prepared no-mosaic splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
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
OLD20_LIKE = DOCTOR_CORE + [
    "内分泌T",
    "HCG日LH",
    "HCG日P",
    "HCG日E2",
    "HCG日总卵泡数",
    "hcg日17卵泡",
    "【16 22】",
    "MII卵数",
    "可利用胚胎数",
    "囊胚培养数",
    "优质囊胚总数",
    "可利用囊胚数",
    "月经_经期天数",
    "月经_周期天数",
    "月经_不规律",
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
        n_estimators=400,
        learning_rate=0.03,
        tree_method="hist",
        n_jobs=4,
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
            frame[name] = frame["pgt_type"].astype("string").str.strip().fillna("UNKNOWN").eq(value).astype(float)
    return cols


def build_groups(frame, feature_cols):
    # 将样本表折叠成周期特征；first实际为每列首个非缺失值，
    # 若同周期记录不一致，可能来自不同原始行，须先做一致性审计。
    # n/positive只用于目标与权重，临床特征不能包含标签或结局泄漏字段。
    work = frame.copy()
    grouped = work.groupby("group_id", sort=False)
    features = grouped[feature_cols].first().copy()
    targets = grouped["label"].agg(["sum", "size"]).rename(columns={"sum": "positive", "size": "n"})
    features["cohort_size"] = targets["n"].astype(float)
    features["log1p_cohort_size"] = np.log1p(targets["n"].astype(float))
    rows = work[["sample_id", "label", "group_id", "pgt_type"]].copy()
    return features, targets, rows


def fit_soft(train_x, train_t, val_x, val_t, fold, tag):
    # 软目标=(本周期阳性数+收缩强度×训练总体阳性率)/(周期大小+收缩强度)。
    # strength=0退化为周期原始阳性比例；先按验证多数方向Acc选，再比logloss。
    prior = float(train_t.positive.sum() / max(train_t.n.sum(), 1))
    trials = []
    for strength in SHRINKAGE:
        target = (train_t.positive + strength * prior) / (train_t.n + strength)
        weight = train_t.n + strength
        for ci, config in enumerate(CONFIGS):
            seeds = [41260815 + fold * 1000 + sum(map(ord, tag)) % 50 * 10 + ci * 3 + k for k in range(3)]
            preds = []
            for seed in seeds:
                model = make_model(config, seed)
                model.fit(train_x.astype(np.float32), target, sample_weight=weight)
                preds.append(model.predict(val_x.astype(np.float32)))
            p = np.mean(preds, axis=0).clip(1e-5, 1 - 1e-5)
            trials.append(
                {
                    "config": config,
                    "strength": float(strength),
                    "seeds": seeds,
                    "val_group_side_acc": group_side_acc(val_t, p),
                    "val_logloss": soft_logloss((val_t.positive / val_t.n).to_numpy(float), p, val_t.n.to_numpy(float)),
                    "val_probability": p,
                }
            )
    return min(trials, key=lambda q: (-q["val_group_side_acc"], q["val_logloss"]))


def refit(dev_x, dev_t, test_x, choice):
    # 参数由验证阶段固定，训练+验证合并重拟合后，仅预测外折测试。
    prior = float(dev_t.positive.sum() / max(dev_t.n.sum(), 1))
    strength = choice["strength"]
    target = (dev_t.positive + strength * prior) / (dev_t.n + strength)
    weight = dev_t.n + strength
    preds = []
    for seed in choice["seeds"]:
        model = make_model(choice["config"], seed)
        model.fit(dev_x.astype(np.float32), target, sample_weight=weight)
        preds.append(model.predict(test_x.astype(np.float32)))
    return np.mean(preds, axis=0).clip(1e-5, 1 - 1e-5)


def run_view(fold, frames, base_cols, pgt_cols, hier=True):
    feature_cols = [c for c in base_cols + pgt_cols if c in frames["train"].columns]
    feature_cols = [
        c for c in feature_cols if frames["train"][c].notna().any() and frames["train"][c].nunique(dropna=True) > 1
    ]
    grouped = {s: build_groups(frames[s], feature_cols) for s in frames}
    tx, tt, _ = grouped["train"]
    vx, vt, val_rows = grouped["val"]
    qx, qt, test_rows = grouped["test"]
    names = list(tx.columns)
    pgt_maps = {s: grouped[s][2].groupby("group_id")["pgt_type"].first() for s in grouped}

    val_prob = pd.Series(index=vt.index, dtype=float)
    test_prob = pd.Series(index=qt.index, dtype=float)

    def fit_groups(train_groups, val_groups, test_groups, tag):
        choice = fit_soft(
            tx.loc[train_groups, names],
            tt.loc[train_groups],
            vx.loc[val_groups, names],
            vt.loc[val_groups],
            fold,
            tag,
        )
        val_p = choice.pop("val_probability")
        val_prob.loc[val_groups] = val_p
        dev_x = pd.concat([tx.loc[train_groups, names], vx.loc[val_groups, names]])
        dev_t = pd.concat([tt.loc[train_groups], vt.loc[val_groups]])
        test_prob.loc[test_groups] = refit(dev_x, dev_t, qx.loc[test_groups, names], choice)

    fit_groups(tt.index, vt.index, qt.index, "global")
    if hier:
        global_val = val_prob.copy()
        global_test = test_prob.copy()
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

    if val_prob.isna().any() or test_prob.isna().any():
        raise ValueError("missing probabilities")
    val_out = val_rows.copy()
    val_out["probability"] = val_rows["group_id"].map(val_prob)
    test_out = test_rows.copy()
    test_out["probability"] = test_rows["group_id"].map(test_prob)
    test_out["fold"] = fold
    return val_out, test_out


def main():
    views = {
        "doctor_core": DOCTOR_CORE,
        "old20_like": OLD20_LIKE,
        "full_numeric": None,  # filled per fold from available numeric columns
    }
    view_oofs = {name: [] for name in views}
    view_vals = {name: [] for name in views}

    for fold in range(5):
        frames = {
            split: pd.read_csv(CV_ROOT / f"fold_{fold}/{split}.csv")
            for split in ("train", "val", "test")
        }
        pgt_cols = encode_pgt(frames["train"], frames)
        skip = {
            "sample_id",
            "video_id",
            "group_id",
            "label",
            "source",
            "pgt_type",
            "对应的video",
            "周期号",
            "数据集",
            "整倍体与否",
            "最终分类_来自标签表",
            "final_class",
            "IVF/ICSI/PGT1号",
            "女方不孕类型(原发不孕继发不孕)",
            "用药方案类型",
            *pgt_cols,
        }
        full_numeric = [
            c
            for c in frames["train"].columns
            if c not in skip and pd.api.types.is_numeric_dtype(frames["train"][c])
        ]
        view_defs = {
            "doctor_core": [c for c in DOCTOR_CORE if c in frames["train"].columns],
            "old20_like": [c for c in OLD20_LIKE if c in frames["train"].columns],
            "full_numeric": full_numeric,
        }
        for name, cols in view_defs.items():
            val_out, test_out = run_view(fold, frames, cols, pgt_cols, hier=True)
            view_vals[name].append(val_out.assign(fold=fold, view=name))
            view_oofs[name].append(test_out.assign(view=name))
            print(
                f"{name} fold={fold} acc={accuracy_score(test_out.label, test_out.probability >= 0.5):.4f}",
                flush=True,
            )

    # Save single-view OOFs and build strict gate/stack.
    single_scores = {}
    merged_test = None
    for name, parts in view_oofs.items():
        oof = pd.concat(parts, ignore_index=True)
        oof.to_csv(HERE / f"oof_view_{name}.csv", index=False)
        single_scores[name] = metric(oof["label"], oof["probability"])
        piece = oof[["sample_id", "label", "group_id", "pgt_type", "fold", "probability"]].rename(
            columns={"probability": name}
        )
        merged_test = piece if merged_test is None else merged_test.merge(
            piece[["sample_id", name]], on="sample_id", how="inner"
        )

    # Strict OOF logreg gate over the three views using other folds.
    gate_parts = []
    gate_meta = {}
    for fold in range(5):
        train = merged_test[merged_test["fold"] != fold]
        test = merged_test[merged_test["fold"] == fold].copy()
        x_cols = ["doctor_core", "old20_like", "full_numeric"]
        clf = LogisticRegression(max_iter=2000, class_weight=None)
        clf.fit(train[x_cols].to_numpy(float), train["label"].to_numpy(int))
        test["probability"] = clf.predict_proba(test[x_cols].to_numpy(float))[:, 1]
        test["prediction"] = (test["probability"] >= 0.5).astype(int)
        gate_parts.append(test)
        gate_meta[str(fold)] = {
            "coef": dict(zip(x_cols, map(float, clf.coef_.ravel()))),
            "intercept": float(clf.intercept_[0]),
        }
    gate = pd.concat(gate_parts, ignore_index=True)
    gate.to_csv(HERE / "oof_view_logreg_gate.csv", index=False)

    # Pair rule gate: use full_numeric unless another view is much more confident.
    rule_parts = []
    for fold in range(5):
        train = merged_test[merged_test["fold"] != fold]
        test = merged_test[merged_test["fold"] == fold].copy()
        best = None
        for margin in np.linspace(0.02, 0.20, 10):
            for base in ("full_numeric", "old20_like", "doctor_core"):
                others = [c for c in ("full_numeric", "old20_like", "doctor_core") if c != base]
                p = train[base].to_numpy(float).copy()
                base_conf = np.abs(train[base] - 0.5).to_numpy(float)
                for other in others:
                    other_p = train[other].to_numpy(float)
                    other_conf = np.abs(other_p - 0.5)
                    take = (other_conf >= base_conf + margin) & (np.sign(other_p - 0.5) != 0)
                    p = np.where(take, other_p, p)
                acc = accuracy_score(train["label"], p >= 0.5)
                key = (acc, -margin)
                if best is None or key > best[0]:
                    best = (key, float(margin), base)
        margin, base = best[1], best[2]
        others = [c for c in ("full_numeric", "old20_like", "doctor_core") if c != base]
        p = test[base].to_numpy(float).copy()
        base_conf = np.abs(test[base] - 0.5).to_numpy(float)
        for other in others:
            other_p = test[other].to_numpy(float)
            other_conf = np.abs(other_p - 0.5)
            take = other_conf >= base_conf + margin
            p = np.where(take, other_p, p)
        test["probability"] = p
        test["prediction"] = (p >= 0.5).astype(int)
        rule_parts.append(test)
        gate_meta[f"rule_fold_{fold}"] = {"margin": margin, "base": base}
    rule = pd.concat(rule_parts, ignore_index=True)
    rule.to_csv(HERE / "oof_view_confidence_gate.csv", index=False)

    def group_side(frame, prob_col="probability"):
        g = frame.groupby("group_id").agg(y=("label", "mean"), p=(prob_col, "first"))
        return float(((g.y >= 0.5).astype(int) == (g.p >= 0.5).astype(int)).mean())

    maj = merged_test.groupby("group_id")["label"].transform(lambda s: int(s.mean() >= 0.5))
    oracle = float((merged_test["label"] == maj).mean())

    rows = []
    for name, score in single_scores.items():
        oof = pd.concat(view_oofs[name], ignore_index=True)
        rows.append(
            {
                "method": f"view_{name}",
                **score,
                "group_side_acc": group_side(oof),
                "majority_oracle_acc": oracle,
                "gap_to_majority_oracle": oracle - score["accuracy"],
            }
        )
    for name, frame in (
        ("view_logreg_gate", gate),
        ("view_confidence_gate", rule),
    ):
        score = metric(frame["label"], frame["probability"])
        rows.append(
            {
                "method": name,
                **score,
                "group_side_acc": group_side(frame),
                "majority_oracle_acc": oracle,
                "gap_to_majority_oracle": oracle - score["accuracy"],
            }
        )

    # Include previous best soft_pgt_hier if present.
    prev = HERE / "oof_soft_pgt_hier.csv"
    if prev.exists():
        prev_oof = pd.read_csv(prev)
        score = metric(prev_oof["label"], prev_oof["probability"])
        rows.append(
            {
                "method": "soft_pgt_hier",
                **score,
                "group_side_acc": group_side(prev_oof),
                "majority_oracle_acc": oracle,
                "gap_to_majority_oracle": oracle - score["accuracy"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    summary.to_csv(HERE / "summary_views.csv", index=False)
    (HERE / "summary_views.json").write_text(
        json.dumps({"results": rows, "gate_meta": gate_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("BEST", summary.iloc[0].to_dict())


if __name__ == "__main__":
    main()
