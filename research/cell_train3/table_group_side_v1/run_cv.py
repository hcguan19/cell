#!/usr/bin/env python3
# 周期表格XGBoost早期候选，包含软比例目标与多数侧目标。
# 周期临床特征取first，同一周期的预测展开到全部样本。
"""Group-side focused pure-table CV on no-mosaic clinical data.

Goal: approach the within-cycle majority-vote oracle by correctly predicting each
cycle's majority euploidy side, then expand that cycle probability to embryos.
"""

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
DATA_CSV = HERE / "data" / "clean_table.csv"
MAJOR_TYPES = ("PGT-A", "PGT-M", "PGT-SR")

RATIOS = {
    "ratio_MII_per_follicle": ("MII卵数", "HCG日总卵泡数"),
    "ratio_usable_embryo_per_MII": ("可利用胚胎数", "MII卵数"),
    "ratio_blast_culture_per_usable": ("囊胚培养数", "可利用胚胎数"),
    "ratio_usable_blast_per_culture": ("可利用囊胚数", "囊胚培养数"),
    "ratio_good_blast_per_culture": ("优质囊胚总数", "囊胚培养数"),
}

CONFIGS = [
    {"max_depth": 1, "min_child_weight": 3, "reg_alpha": 0.2, "reg_lambda": 4.0},
    {"max_depth": 2, "min_child_weight": 3, "reg_alpha": 0.2, "reg_lambda": 5.0},
    {"max_depth": 2, "min_child_weight": 6, "reg_alpha": 0.5, "reg_lambda": 8.0},
    {"max_depth": 3, "min_child_weight": 6, "reg_alpha": 0.8, "reg_lambda": 10.0},
    {"max_depth": 3, "min_child_weight": 12, "reg_alpha": 1.5, "reg_lambda": 18.0},
]
SHRINKAGE = [0.0, 2.0, 5.0, 10.0]
SEEDS_PER_TRIAL = 3


def stable_tag_code(tag: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(tag)) % 97


def metric(y, p, threshold=0.5):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
    else:
        out["roc_auc"] = float("nan")
    return out


def soft_logloss(target, probability, weight):
    target = np.asarray(target, float)
    probability = np.clip(np.asarray(probability, float), 1e-5, 1 - 1e-5)
    weight = np.asarray(weight, float)
    value = -(target * np.log(probability) + (1 - target) * np.log(1 - probability))
    return float(np.average(value, weights=weight))


def group_side_acc(targets: pd.DataFrame, probability: np.ndarray) -> float:
    pred_side = (np.asarray(probability) >= 0.5).astype(int)
    true_side = (targets["positive"] / targets["n"] >= 0.5).astype(int).to_numpy()
    return float((pred_side == true_side).mean())


def make_regressor(config, seed):
    return xgb.XGBRegressor(
        objective="reg:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.025,
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
        subsample=0.9,
        colsample_bytree=0.85,
        **config,
    )


def add_ratios(frame: pd.DataFrame) -> list[str]:
    added = []
    for name, (num, den) in RATIOS.items():
        if num in frame.columns and den in frame.columns:
            den_v = pd.to_numeric(frame[den], errors="coerce")
            num_v = pd.to_numeric(frame[num], errors="coerce")
            frame[name] = num_v.div(den_v.where(den_v > 0))
            added.append(name)
    return added


def encode_categoricals(train: pd.DataFrame, frames: dict[str, pd.DataFrame], cols: list[str]):
    encoded = []
    for col in cols:
        values = train[col].astype("string").str.strip().fillna("__MISSING__")
        vocab = sorted(values.value_counts().loc[lambda s: s >= 5].index.tolist())
        for i, value in enumerate(vocab):
            name = f"{col}__{i:02d}"
            encoded.append(name)
            for frame in frames.values():
                cur = frame[col].astype("string").str.strip().fillna("__MISSING__")
                frame[name] = cur.eq(value).astype(float)
    return encoded


def build_group_matrix(frame: pd.DataFrame, feature_cols: list[str]):
    work = frame.copy()
    grouped = work.groupby("group_id", sort=False)
    features = grouped[feature_cols].first()
    targets = grouped["label"].agg(["sum", "size"]).rename(columns={"sum": "positive", "size": "n"})
    features = features.copy()
    features["cohort_size"] = targets["n"].astype(float)
    features["log1p_cohort_size"] = np.log1p(targets["n"].astype(float))
    rows = work[["sample_id", "label", "group_id", "pgt_type", "source"]].copy()
    return features, targets, rows


def fit_soft_model(train_x, train_t, val_x, val_t, fold, tag):
    prior = float(train_t.positive.sum() / max(train_t.n.sum(), 1))
    trials = []
    for strength in SHRINKAGE:
        target = (train_t.positive + strength * prior) / (train_t.n + strength)
        weight = train_t.n + strength
        for ci, config in enumerate(CONFIGS):
            seeds = [
                20260815 + fold * 1000 + stable_tag_code(tag) * 10 + ci * 3 + k
                for k in range(SEEDS_PER_TRIAL)
            ]
            probs = []
            for seed in seeds:
                model = make_regressor(config, seed)
                model.fit(train_x.astype(np.float32), target, sample_weight=weight)
                probs.append(model.predict(val_x.astype(np.float32)))
            p = np.mean(probs, axis=0).clip(1e-5, 1 - 1e-5)
            rate = (val_t.positive / val_t.n).to_numpy(float)
            trials.append(
                {
                    "config_index": ci,
                    "config": config,
                    "strength": float(strength),
                    "seeds": seeds,
                    "val_logloss": soft_logloss(rate, p, val_t.n.to_numpy(float)),
                    "val_group_side_acc": group_side_acc(val_t, p),
                    "val_probability": p,
                }
            )
    # Prefer group-side Acc, then logloss.
    return min(
        trials,
        key=lambda q: (-q["val_group_side_acc"], q["val_logloss"], q["config_index"], q["strength"]),
    ), trials


def refit_soft(dev_x, dev_t, test_x, choice):
    prior = float(dev_t.positive.sum() / max(dev_t.n.sum(), 1))
    strength = choice["strength"]
    target = (dev_t.positive + strength * prior) / (dev_t.n + strength)
    weight = dev_t.n + strength
    preds = []
    for seed in choice["seeds"]:
        model = make_regressor(choice["config"], seed)
        model.fit(dev_x.astype(np.float32), target, sample_weight=weight)
        preds.append(model.predict(test_x.astype(np.float32)))
    return np.mean(preds, axis=0).clip(1e-5, 1 - 1e-5)


def fit_majority_model(train_x, train_t, val_x, val_t, fold, tag):
    y_train = (train_t.positive / train_t.n >= 0.5).astype(int)
    y_val = (val_t.positive / val_t.n >= 0.5).astype(int)
    # Emphasize clearer majorities and larger cohorts.
    clarity = (2.0 * (train_t.positive / train_t.n) - 1.0).abs().clip(0.05, 1.0)
    weight = (train_t.n.astype(float) * clarity).to_numpy(float)
    trials = []
    for ci, config in enumerate(CONFIGS):
        seeds = [
            30260815 + fold * 1000 + stable_tag_code(tag) * 10 + ci * 3 + k
            for k in range(SEEDS_PER_TRIAL)
        ]
        probs = []
        for seed in seeds:
            model = make_regressor(config, seed)
            model.fit(train_x.astype(np.float32), y_train.to_numpy(float), sample_weight=weight)
            probs.append(model.predict(val_x.astype(np.float32)))
        p = np.mean(probs, axis=0).clip(1e-5, 1 - 1e-5)
        trials.append(
            {
                "config_index": ci,
                "config": config,
                "strength": 0.0,
                "seeds": seeds,
                "val_logloss": float(log_loss(y_val, p, labels=[0, 1])),
                "val_group_side_acc": float(accuracy_score(y_val, p >= 0.5)),
                "val_probability": p,
            }
        )
    return min(
        trials,
        key=lambda q: (-q["val_group_side_acc"], q["val_logloss"], q["config_index"]),
    ), trials


def refit_majority(dev_x, dev_t, test_x, choice):
    y = (dev_t.positive / dev_t.n >= 0.5).astype(int)
    clarity = (2.0 * (dev_t.positive / dev_t.n) - 1.0).abs().clip(0.05, 1.0)
    weight = (dev_t.n.astype(float) * clarity).to_numpy(float)
    preds = []
    for seed in choice["seeds"]:
        model = make_regressor(choice["config"], seed)
        model.fit(dev_x.astype(np.float32), y.to_numpy(float), sample_weight=weight)
        preds.append(model.predict(test_x.astype(np.float32)))
    return np.mean(preds, axis=0).clip(1e-5, 1 - 1e-5)


def expand(rows: pd.DataFrame, group_probability: pd.Series) -> np.ndarray:
    return rows["group_id"].map(group_probability).to_numpy(float)


def load_fold(fold: int):
    fold_dir = CV_ROOT / f"fold_{fold}"
    frames = {
        split: pd.read_csv(fold_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    return frames


def prepare_features(frames: dict[str, pd.DataFrame]):
    cat_cols = [c for c in ("IVF/ICSI/PGT1号", "女方不孕类型(原发不孕继发不孕)", "用药方案类型") if c in frames["train"].columns]
    for frame in frames.values():
        add_ratios(frame)
    encoded = encode_categoricals(frames["train"], frames, cat_cols)
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
        *cat_cols,
        *encoded,
    }
    numeric = [
        c
        for c in frames["train"].columns
        if c not in skip and pd.api.types.is_numeric_dtype(frames["train"][c])
    ]
    # Keep only columns with train signal; preserve order and uniqueness.
    feature_cols = []
    seen = set()
    for c in numeric + encoded:
        if c in seen:
            continue
        if frames["train"][c].notna().any() and frames["train"][c].nunique(dropna=True) > 1:
            feature_cols.append(c)
            seen.add(c)
    return feature_cols


def run_variant(fold, frames, feature_cols, mode: str, pgt_hier: bool):
    grouped = {split: build_group_matrix(frames[split], feature_cols) for split in frames}
    pgt_maps = {}
    for split, (_, _, rows) in grouped.items():
        pgt_maps[split] = rows.groupby("group_id")["pgt_type"].first()

    tx, tt, _ = grouped["train"]
    vx, vt, val_rows = grouped["val"]
    qx, qt, test_rows = grouped["test"]
    feature_names = list(tx.columns)

    val_prob = pd.Series(index=vt.index, dtype=float)
    test_prob = pd.Series(index=qt.index, dtype=float)
    choices = {}

    def fit_one(train_groups, val_groups, test_groups, tag):
        if mode == "majority":
            choice, _ = fit_majority_model(
                tx.loc[train_groups, feature_names],
                tt.loc[train_groups],
                vx.loc[val_groups, feature_names],
                vt.loc[val_groups],
                fold,
                tag,
            )
            val_p = choice.pop("val_probability")
            val_prob.loc[val_groups] = val_p
            dev_x = pd.concat(
                [tx.loc[train_groups, feature_names], vx.loc[val_groups, feature_names]]
            )
            dev_t = pd.concat([tt.loc[train_groups], vt.loc[val_groups]])
            test_prob.loc[test_groups] = refit_majority(
                dev_x, dev_t, qx.loc[test_groups, feature_names], choice
            )
        else:
            choice, _ = fit_soft_model(
                tx.loc[train_groups, feature_names],
                tt.loc[train_groups],
                vx.loc[val_groups, feature_names],
                vt.loc[val_groups],
                fold,
                tag,
            )
            val_p = choice.pop("val_probability")
            val_prob.loc[val_groups] = val_p
            dev_x = pd.concat(
                [tx.loc[train_groups, feature_names], vx.loc[val_groups, feature_names]]
            )
            dev_t = pd.concat([tt.loc[train_groups], vt.loc[val_groups]])
            test_prob.loc[test_groups] = refit_soft(
                dev_x, dev_t, qx.loc[test_groups, feature_names], choice
            )
        choices[tag] = {
            "n_train_groups": int(len(train_groups)),
            "n_val_groups": int(len(val_groups)),
            "n_test_groups": int(len(test_groups)),
            "choice": {k: v for k, v in choice.items() if k != "val_probability"},
        }

    if not pgt_hier:
        fit_one(tt.index, vt.index, qt.index, "global")
    else:
        # Start from a global fit, then overwrite major PGT types.
        fit_one(tt.index, vt.index, qt.index, "global")
        global_val = val_prob.copy()
        global_test = test_prob.copy()
        for ti, type_name in enumerate(MAJOR_TYPES):
            train_groups = pgt_maps["train"].index[pgt_maps["train"].eq(type_name)]
            val_groups = pgt_maps["val"].index[pgt_maps["val"].eq(type_name)]
            test_groups = pgt_maps["test"].index[pgt_maps["test"].eq(type_name)]
            if min(len(train_groups), len(val_groups), len(test_groups)) < 4:
                continue
            # Fit type-specific model.
            if mode == "majority":
                choice, _ = fit_majority_model(
                    tx.loc[train_groups, feature_names],
                    tt.loc[train_groups],
                    vx.loc[val_groups, feature_names],
                    vt.loc[val_groups],
                    fold,
                    f"type_{type_name}",
                )
                type_val = choice.pop("val_probability")
            else:
                choice, _ = fit_soft_model(
                    tx.loc[train_groups, feature_names],
                    tt.loc[train_groups],
                    vx.loc[val_groups, feature_names],
                    vt.loc[val_groups],
                    fold,
                    f"type_{type_name}",
                )
                type_val = choice.pop("val_probability")

            # Blend type vs global on validation only.
            y_val = (vt.loc[val_groups, "positive"] / vt.loc[val_groups, "n"] >= 0.5).astype(int).to_numpy()
            g_val = global_val.loc[val_groups].to_numpy(float)
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
            blend = max(blends, key=lambda q: (q["group_side_acc"], -q["logloss"], -q["alpha"]))
            val_prob.loc[val_groups] = (1 - blend["alpha"]) * g_val + blend["alpha"] * type_val

            dev_x = pd.concat(
                [tx.loc[train_groups, feature_names], vx.loc[val_groups, feature_names]]
            )
            dev_t = pd.concat([tt.loc[train_groups], vt.loc[val_groups]])
            if mode == "majority":
                type_test = refit_majority(dev_x, dev_t, qx.loc[test_groups, feature_names], choice)
            else:
                type_test = refit_soft(dev_x, dev_t, qx.loc[test_groups, feature_names], choice)
            test_prob.loc[test_groups] = (
                (1 - blend["alpha"]) * global_test.loc[test_groups].to_numpy(float)
                + blend["alpha"] * type_test
            )
            choices[f"type_{type_name}"] = {
                "n_train_groups": int(len(train_groups)),
                "n_val_groups": int(len(val_groups)),
                "n_test_groups": int(len(test_groups)),
                "choice": {k: v for k, v in choice.items() if k != "val_probability"},
                "blend": blend,
            }

    missing_val = val_prob.isna().sum()
    missing_test = test_prob.isna().sum()
    if missing_val or missing_test:
        raise ValueError(f"missing group probabilities fold={fold} val={missing_val} test={missing_test}")

    test_sample_p = expand(test_rows, test_prob)
    val_sample_p = expand(val_rows, val_prob)
    test_out = test_rows.copy()
    test_out["probability"] = test_sample_p
    test_out["prediction"] = (test_sample_p >= 0.5).astype(int)
    test_out["fold"] = fold
    test_out["group_probability"] = test_out["group_id"].map(test_prob)

    # Group-side metrics on test groups.
    true_side = (qt["positive"] / qt["n"] >= 0.5).astype(int)
    pred_side = (test_prob >= 0.5).astype(int)
    fold_metrics = {
        "sample": metric(test_out["label"], test_out["probability"]),
        "group_side_acc": float((true_side.to_numpy() == pred_side.to_numpy()).mean()),
        "val_sample_acc": float(accuracy_score(val_rows["label"], val_sample_p >= 0.5)),
        "val_group_side_acc": float(
            (
                (vt["positive"] / vt["n"] >= 0.5).astype(int).to_numpy()
                == (val_prob >= 0.5).astype(int).to_numpy()
            ).mean()
        ),
        "choices": choices,
        "n_features": int(len(feature_names)),
    }
    return test_out, fold_metrics


def score_frame(frame: pd.DataFrame, name: str):
    m = metric(frame["label"], frame["probability"])
    # group side
    g = (
        frame.groupby("group_id")
        .agg(label_sum=("label", "sum"), n=("label", "size"), p=("probability", "first"))
        .reset_index()
    )
    true_side = (g["label_sum"] / g["n"] >= 0.5).astype(int)
    pred_side = (g["p"] >= 0.5).astype(int)
    m["group_side_acc"] = float((true_side == pred_side).mean())
    m["name"] = name
    m["folds"] = []
    for fold, part in frame.groupby("fold"):
        fm = metric(part["label"], part["probability"])
        m["folds"].append({"fold": int(fold), **fm})
    return m


def majority_oracle_from_frame(frame: pd.DataFrame):
    maj = frame.groupby("group_id")["label"].transform(lambda s: int(s.mean() >= 0.5))
    return float((frame["label"] == maj).mean())


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"missing {DATA_CSV}; run prepare_data.py first")

    variants = [
        ("soft_global", {"mode": "soft", "pgt_hier": False}),
        ("majority_global", {"mode": "majority", "pgt_hier": False}),
        ("soft_pgt_hier", {"mode": "soft", "pgt_hier": True}),
        ("majority_pgt_hier", {"mode": "majority", "pgt_hier": True}),
    ]

    all_results = {}
    oofs = {}
    for name, cfg in variants:
        parts = []
        fold_infos = {}
        for fold in range(5):
            frames = load_fold(fold)
            feature_cols = prepare_features(frames)
            test_out, fold_metrics = run_variant(
                fold, frames, feature_cols, mode=cfg["mode"], pgt_hier=cfg["pgt_hier"]
            )
            parts.append(test_out)
            fold_infos[str(fold)] = fold_metrics
            print(
                f"{name} fold={fold} acc={fold_metrics['sample']['accuracy']:.4f} "
                f"group_side={fold_metrics['group_side_acc']:.4f}",
                flush=True,
            )
        oof = pd.concat(parts, ignore_index=True)
        oof_path = HERE / f"oof_{name}.csv"
        oof.to_csv(oof_path, index=False)
        scored = score_frame(oof, name)
        scored["majority_oracle_acc"] = majority_oracle_from_frame(oof)
        scored["gap_to_majority_oracle"] = scored["majority_oracle_acc"] - scored["accuracy"]
        scored["fold_details"] = fold_infos
        all_results[name] = scored
        oofs[name] = oof
        print(
            f"[OOF] {name}: Acc={scored['accuracy']:.4f} "
            f"AUC={scored['roc_auc']:.4f} group_side={scored['group_side_acc']:.4f} "
            f"oracle={scored['majority_oracle_acc']:.4f}",
            flush=True,
        )

    # Optional: validation-selected blend between soft_pgt_hier and majority_pgt_hier.
    soft = oofs["soft_pgt_hier"]
    maj = oofs["majority_pgt_hier"]
    blend_parts = []
    blend_meta = {}
    for fold in range(5):
        # Use other folds to choose alpha? Strict: use each fold's val from fold_details is hard
        # because we only saved test. Instead choose alpha on the 4 other test folds (nested OOF).
        train_folds = [f for f in range(5) if f != fold]
        best = None
        for alpha in np.arange(0.0, 1.01, 0.1):
            sub_s = soft[soft["fold"].isin(train_folds)]
            sub_m = maj[maj["fold"].isin(train_folds)]
            p = (1 - alpha) * sub_s["probability"].to_numpy() + alpha * sub_m["probability"].to_numpy()
            acc = accuracy_score(sub_s["label"], p >= 0.5)
            g = sub_s.copy()
            g["probability"] = p
            gacc = score_frame(g, "tmp")["group_side_acc"]
            key = (gacc, acc, -abs(alpha - 0.5))
            if best is None or key > best[0]:
                best = (key, float(alpha))
        alpha = best[1]
        part = soft[soft["fold"] == fold].copy()
        part_m = maj[maj["fold"] == fold]
        part["probability"] = (1 - alpha) * part["probability"].to_numpy() + alpha * part_m["probability"].to_numpy()
        part["prediction"] = (part["probability"] >= 0.5).astype(int)
        blend_parts.append(part)
        blend_meta[str(fold)] = {"alpha_majority": alpha}
    blend = pd.concat(blend_parts, ignore_index=True)
    blend.to_csv(HERE / "oof_blend_soft_majority_pgt_hier.csv", index=False)
    blend_score = score_frame(blend, "blend_soft_majority_pgt_hier")
    blend_score["majority_oracle_acc"] = majority_oracle_from_frame(blend)
    blend_score["gap_to_majority_oracle"] = (
        blend_score["majority_oracle_acc"] - blend_score["accuracy"]
    )
    blend_score["fold_alphas"] = blend_meta
    all_results["blend_soft_majority_pgt_hier"] = blend_score

    summary_rows = []
    for name, scored in all_results.items():
        summary_rows.append(
            {
                "method": name,
                "n": scored["n"],
                "accuracy": scored["accuracy"],
                "balanced_accuracy": scored["balanced_accuracy"],
                "f1": scored["f1"],
                "roc_auc": scored["roc_auc"],
                "group_side_acc": scored["group_side_acc"],
                "majority_oracle_acc": scored["majority_oracle_acc"],
                "gap_to_majority_oracle": scored["gap_to_majority_oracle"],
                "confusion_matrix": scored["confusion_matrix"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False)
    summary_df.to_csv(HERE / "summary_table.csv", index=False)
    (HERE / "summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best = summary_df.iloc[0].to_dict()
    print("\n=== BEST ===", flush=True)
    print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
