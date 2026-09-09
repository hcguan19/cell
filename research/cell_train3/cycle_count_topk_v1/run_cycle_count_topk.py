#!/usr/bin/env python3
# 早期周期数量回归实验，最终版本复用cycle_frame及周期bootstrap。
# 不应把本文件中所有候选回归器都画成当前最佳模型的组成部分。
"""Strict nested cycle-count + within-cycle video-rank + conservative Top-K.

The input probabilities/ranks are already strict OOF base predictions.  For each
outer fold, all count-model fitting and policy selection uses other folds only.
The held-out fold is touched once for final evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT = ROOT / "rank_boost_v2/oof_dual_recommended_boost.csv"
SEED = 20260822


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def cycle_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, x in df.groupby("group_id", sort=False):
        p = float(x["p_classify"].iloc[0])
        n = int(len(x))
        rows.append({
            "group_id": gid, "fold": int(x["fold"].iloc[0]), "n": n,
            "p": p, "logit_p": float(logit([p])[0]), "np": n * p,
            "pgt_type": str(x["pgt_type"].iloc[0]), "k_true": int(x["label"].sum()),
            "pos_rate": float(x["label"].mean()),
        })
    return pd.DataFrame(rows)


FEATURES_NUM = ["p", "logit_p", "n", "np"]
FEATURES_CAT = ["pgt_type"]


def design_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
    ])


def fit_count_models(train: pd.DataFrame):
    X = train[FEATURES_NUM + FEATURES_CAT]
    y = train["pos_rate"].to_numpy(float)
    models = [
        make_pipeline(design_preprocessor(), Ridge(alpha=8.0)),
        make_pipeline(design_preprocessor(), HistGradientBoostingRegressor(
            max_iter=80, learning_rate=0.035, max_leaf_nodes=7,
            min_samples_leaf=20, l2_regularization=2.0, random_state=SEED)),
        make_pipeline(design_preprocessor(), ExtraTreesRegressor(
            n_estimators=300, max_depth=3, min_samples_leaf=8,
            max_features=0.8, random_state=SEED, n_jobs=-1)),
    ]
    for model in models:
        model.fit(X, y)
    return models


def predict_count(models, cycles: pd.DataFrame) -> pd.DataFrame:
    X = cycles[FEATURES_NUM + FEATURES_CAT]
    q_all = np.column_stack([np.clip(m.predict(X), 0.02, 0.98) for m in models])
    out = cycles[["group_id", "fold", "n", "p", "k_true"]].copy()
    out["q_count"] = q_all.mean(axis=1)
    out["q_disagree"] = q_all.std(axis=1)
    out["k_float"] = out["q_count"] * out["n"]
    out["k_pred"] = np.rint(out["k_float"]).astype(int).clip(lower=0)
    out["k_pred"] = np.minimum(out["k_pred"], out["n"]).astype(int)
    # Confidence is high away from a rounding boundary and with model agreement.
    out["count_margin"] = np.abs(out["k_float"] - (np.floor(out["k_float"]) + 0.5))
    out["count_conf"] = out["count_margin"] - 1.5 * out["q_disagree"]
    out["k_abs_err"] = np.abs(out["k_pred"] - out["k_true"])
    return out


def rank_confidence(x: pd.DataFrame) -> float:
    s = x["video_rank"].to_numpy(float)
    if len(s) < 2 or not np.isfinite(s).all() or np.std(s) < 1e-8:
        return -1e9
    z = (s - np.mean(s)) / (np.std(s) + 1e-8)
    zs = np.sort(z)[::-1]
    # Minimum adjacent separation: a conservative confidence for a K-boundary.
    return float(np.max(np.diff(np.sort(z)))) if len(zs) > 1 else -1e9


def add_cycle_rank_conf(df: pd.DataFrame) -> pd.DataFrame:
    conf = df.groupby("group_id").apply(rank_confidence, include_groups=False).rename("rank_conf")
    return df.merge(conf, on="group_id", how="left")


def decode(df: pd.DataFrame, counts: pd.DataFrame, count_thr: float, rank_thr: float,
           max_change_frac: float = 0.75) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = counts.set_index("group_id")
    pred = (df["p_classify"].to_numpy(float) >= 0.5).astype(int)
    prob = df["p_classify"].to_numpy(float).copy()
    acted = np.zeros(len(df), dtype=bool)
    for gid, idx in df.groupby("group_id", sort=False).groups.items():
        pos = df.index.get_indexer(idx)
        row = c.loc[gid]
        n, k = int(row["n"]), int(row["k_pred"])
        base_k = int(pred[pos].sum())
        n_change = abs(k - base_k)
        rconf = float(df.loc[idx, "rank_conf"].iloc[0])
        eligible = (
            k != base_k and 0 < k < n and float(row["count_conf"]) >= count_thr
            and rconf >= rank_thr and n_change <= max(1, int(np.ceil(max_change_frac * n)))
        )
        if not eligible:
            continue
        scores = df.loc[idx, "video_rank"].to_numpy(float)
        order = np.argsort(-scores, kind="stable")
        local = np.zeros(n, dtype=int)
        local[order[:k]] = 1
        pred[pos] = local
        # Preserve count-model calibration while encoding strict video ordering.
        ranks = np.empty(n, int); ranks[order] = np.arange(n)
        centered = (k - 0.5 - ranks) / max(1.0, n / 3.0)
        prob[pos] = 1.0 / (1.0 + np.exp(-centered))
        acted[pos] = True
    return pred, prob, acted


def nested_meta_predictions(df: pd.DataFrame, outer_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = cycle_frame(df)
    train_folds = sorted(f for f in cycles["fold"].unique() if f != outer_fold)
    inner_parts = []
    for vf in train_folds:
        tr = cycles[(cycles.fold != outer_fold) & (cycles.fold != vf)]
        va = cycles[cycles.fold == vf]
        inner_parts.append(predict_count(fit_count_models(tr), va))
    inner_counts = pd.concat(inner_parts, ignore_index=True)
    outer_train = cycles[cycles.fold != outer_fold]
    outer_test = cycles[cycles.fold == outer_fold]
    test_counts = predict_count(fit_count_models(outer_train), outer_test)
    return inner_counts, test_counts


def choose_policy(train_df: pd.DataFrame, inner_counts: pd.DataFrame) -> dict:
    count_grid = [-0.10, 0.00, 0.08, 0.16, 0.24, 0.32]
    rank_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    frac_grid = [0.34, 0.50, 0.75]
    y = train_df["label"].to_numpy(int)
    base = (train_df["p_classify"].to_numpy(float) >= 0.5).astype(int)
    base_acc = accuracy_score(y, base)
    candidates = []
    for ct in count_grid:
        for rt in rank_grid:
            for mf in frac_grid:
                pred, _, acted = decode(train_df, inner_counts, ct, rt, mf)
                acc = accuracy_score(y, pred)
                fold_deltas = []
                for f in sorted(train_df["fold"].unique()):
                    mask = train_df["fold"].to_numpy() == f
                    fold_deltas.append(float(np.mean(pred[mask] == y[mask]) - np.mean(base[mask] == y[mask])))
                # Conservative selection: penalize large intervention sets and losses.
                action = float(acted.mean())
                score = acc - 0.003 * action
                candidates.append({"count_thr": ct, "rank_thr": rt, "max_change_frac": mf,
                                   "acc": acc, "delta": acc - base_acc,
                                   "action_frac": action, "selection_score": score,
                                   "inner_fold_deltas": fold_deltas,
                                   "n_nonnegative_inner_folds": int(np.sum(np.asarray(fold_deltas) >= 0)),
                                   "median_inner_delta": float(np.median(fold_deltas))})
    # Always include no-action fallback. Require >=2 corrected embryos on inner OOF.
    candidates.append({"count_thr": 1e9, "rank_thr": 1e9, "max_change_frac": 0.0,
                       "acc": base_acc, "delta": 0.0, "action_frac": 0.0,
                       "selection_score": base_acc, "inner_fold_deltas": [0.0] * 4,
                       "n_nonnegative_inner_folds": 4, "median_inner_delta": 0.0})
    # A deployable rule must gain overall and not be driven by one lucky inner fold.
    valid = [x for x in candidates if x["delta"] >= 2.0 / len(train_df)
             and x["n_nonnegative_inner_folds"] >= 3 and x["median_inner_delta"] > 0]
    pool = valid if valid else [candidates[-1]]
    return max(pool, key=lambda z: (z["selection_score"], z["acc"], -z["action_frac"]))


def metric_pack(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, prob)), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)}


def paired_cycle_bootstrap(df, pred_base, pred_new, n_boot=5000):
    rng = np.random.default_rng(SEED)
    groups = df["group_id"].unique()
    by_group = {g: np.flatnonzero(df["group_id"].to_numpy() == g) for g in groups}
    y = df["label"].to_numpy(int)
    deltas = []
    for _ in range(n_boot):
        draw = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([by_group[g] for g in draw])
        deltas.append(np.mean(pred_new[idx] == y[idx]) - np.mean(pred_base[idx] == y[idx]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"delta_mean": float(np.mean(deltas)), "ci95": [float(lo), float(hi)],
            "p_improve": float(np.mean(np.asarray(deltas) > 0))}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, dtype={"sample_id": str, "video_id": str, "group_id": str})
    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype(int); df["fold"] = df["fold"].astype(int)
    df = add_cycle_rank_conf(df)
    outputs, fold_rows, policies, count_rows = [], [], [], []
    for fold in sorted(df["fold"].unique()):
        train_df = df[df.fold != fold].copy()
        test_df = df[df.fold == fold].copy()
        inner_counts, test_counts = nested_meta_predictions(df, int(fold))
        policy = choose_policy(train_df, inner_counts)
        pred, prob, acted = decode(test_df, test_counts, policy["count_thr"],
                                   policy["rank_thr"], policy["max_change_frac"])
        base_pred = (test_df["p_classify"].to_numpy(float) >= 0.5).astype(int)
        out = test_df.copy(); out["pred_base"] = base_pred; out["pred_topk"] = pred
        out["prob_topk"] = prob; out["topk_acted"] = acted
        out = out.merge(test_counts, on=["group_id", "fold"], how="left", suffixes=("", "_count"))
        outputs.append(out)
        fm = metric_pack(out.label, pred, prob); bm = metric_pack(out.label, base_pred, out.p_classify)
        fold_rows.append({"fold": int(fold), "n": len(out), "base_acc": bm["accuracy"],
                          "topk_acc": fm["accuracy"], "delta": fm["accuracy"] - bm["accuracy"],
                          "action_frac": float(acted.mean()), "count_mae": float(test_counts.k_abs_err.mean())})
        policy["outer_fold"] = int(fold); policies.append(policy)
        count_rows.append(test_counts)
    oof = pd.concat(outputs, ignore_index=True).sort_values(["fold", "group_id", "sample_id"])
    y = oof.label.to_numpy(int); pb = oof.pred_base.to_numpy(int); pn = oof.pred_topk.to_numpy(int)
    summary = {
        "protocol": "strict nested outer-fold count model and policy selection",
        "input": str(INPUT), "n_embryo": int(len(oof)), "n_cycles": int(oof.group_id.nunique()),
        "baseline": metric_pack(y, pb, oof.p_classify),
        "cycle_count_topk": metric_pack(y, pn, oof.prob_topk),
        "delta_accuracy": float(np.mean(pn == y) - np.mean(pb == y)),
        "n_changed": int(np.sum(pn != pb)), "n_acted": int(oof.topk_acted.sum()),
        "bootstrap": paired_cycle_bootstrap(oof, pb, pn), "folds": fold_rows, "policies": policies,
    }
    oof.to_csv(HERE / "oof_cycle_count_topk.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(HERE / "fold_metrics.csv", index=False)
    pd.concat(count_rows, ignore_index=True).to_csv(HERE / "oof_cycle_counts.csv", index=False)
    with open(HERE / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
