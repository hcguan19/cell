#!/usr/bin/env python3
# 早期V-JEPA嵌套排序实验与共享fit_predict训练函数。
# 最终阶段版本只复用排序原语，不是运行本文件main所得的全部历史候选。
"""Strict nested evaluation of frozen V-JEPA2.1 features and conservative Top-K fusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data0/guanhuancong/cell/cell_train3")
FEATURE_ROOT = ROOT / "vjepa21_vitb_video_v1/features"
FEATURE_META = ROOT / "cycle_count_topk_bestvideo_v1/oof_discrete_topk.csv"
TABLE = ROOT / "qwen_pls2_video_topk_v1/oof_pls2_topk.csv"
OUT = ROOT / "vjepa21_vitb_video_v1"
SEED = 20260905

sys.path.insert(0, str(ROOT / "raw_video_sibling_ssl_v1"))
from common import pair_auc

sys.path.insert(0, str(ROOT))
from evaluate_updated_table_video_topk import crossfit_dist, decode, select_policy

sys.path.insert(0, str(ROOT / "cycle_count_topk_bestvideo_v1"))
from run_best_video_topk import metrics


def load_features():
    meta = pd.read_csv(FEATURE_META, dtype={"sample_id": str})
    result = np.empty((len(meta), 8, 9, 768), dtype=np.float32)
    valid = np.empty((len(meta), 4), dtype=bool)
    seen = np.zeros(len(meta), dtype=bool)
    for shard in range(6):
        rows = np.load(FEATURE_ROOT / f"rows_{shard}.npy")
        result[rows] = np.load(FEATURE_ROOT / f"features_{shard}.npy").astype(np.float32)
        valid[rows] = np.load(FEATURE_ROOT / f"phase_valid_{shard}.npy")
        seen[rows] = True
    if not seen.all():
        raise ValueError(f"missing {int((~seen).sum())} feature rows")
    return meta[["sample_id"]], result, valid


def representations(grid, valid):
    temporal = grid.mean(axis=2)  # N, 8, 768
    stages = temporal.reshape(len(grid), 4, 2, 768).mean(axis=2)
    within = temporal.reshape(len(grid), 4, 2, 768)
    within = within[:, :, 1] - within[:, :, 0]
    across = stages[:, 1:] - stages[:, :-1]
    spatial = grid.mean(axis=1)
    validity = valid.astype(np.float32)
    return {
        "global": np.concatenate([grid.mean(axis=(1, 2)), validity], axis=1),
        "stages": np.concatenate([stages.reshape(len(grid), -1), validity], axis=1),
        "dynamics": np.concatenate(
            [across.reshape(len(grid), -1), within.reshape(len(grid), -1), validity], axis=1
        ),
        "spatial": np.concatenate([spatial.reshape(len(grid), -1), validity], axis=1),
    }


def paired_examples(z, labels, groups):
    differences = []
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        pos, neg = idx[labels[idx] == 1], idx[labels[idx] == 0]
        if len(pos) and len(neg):
            delta = (z[pos][:, None, :] - z[neg][None, :, :]).reshape(-1, z.shape[1])
            differences.append(delta)
    if not differences:
        raise ValueError("no mixed-label cycle in training data")
    positive = np.concatenate(differences)
    x = np.concatenate([positive, -positive])
    y = np.concatenate([np.ones(len(positive), int), np.zeros(len(positive), int)])
    return x, y


def fit_predict(x, labels, groups, train_idx, test_idx, components, c_value):
    # 共享排序训练原语。x为每样本特征矩阵，train/test是全局整数行号。
    # scaler与PCA只拟合train；成对样本只在train内的同周期正负样本间构造。
    scaler = StandardScaler().fit(x[train_idx])
    train_scaled = scaler.transform(x[train_idx])
    n_components = min(components, len(train_idx) - 1, x.shape[1])
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=SEED)
    z_train = pca.fit_transform(train_scaled)
    z_test = pca.transform(scaler.transform(x[test_idx]))
    # 正差分x_pos-x_neg标1，反向差分标0，学习无截距线性排序方向。
    # 没有混合标签周期时无法构造监督对，原函数会明确报错。
    pair_x, pair_y = paired_examples(z_train, labels[train_idx], groups[train_idx])
    ranker = LogisticRegression(
        C=c_value, fit_intercept=False, solver="liblinear", max_iter=2000, random_state=SEED
    ).fit(pair_x, pair_y)
    train_score = ranker.decision_function(z_train)
    # 这里的分数标准化使用训练统计量，区别于最终融合入口的按折整批标准化。
    center, scale = float(train_score.mean()), float(train_score.std() + 1e-8)
    return (ranker.decision_function(z_test) - center) / scale


def candidate_grid():
    return [
        (representation, components, c_value)
        for representation in ("global", "stages", "dynamics", "spatial")
        for components in (16, 32, 64)
        for c_value in (0.03, 0.3)
    ]


def threshold_for_accuracy(labels, scores):
    # 只应传入开发集标签来选阈值。此函数是历史实验工具，
    # 当前phasewise固定排序入口只导入fit_predict，并不调用该阈值搜索。
    values = np.unique(scores)
    candidates = np.r_[values[0] - 1e-6, (values[:-1] + values[1:]) / 2, values[-1] + 1e-6]
    accuracies = np.asarray([np.mean((scores >= threshold) == labels) for threshold in candidates])
    best = np.flatnonzero(accuracies == accuracies.max())
    selected = best[np.argmin(np.abs(candidates[best]))]
    return float(candidates[selected]), float(accuracies[selected])


def main():
    feature_ids, grid, valid = load_features()
    rep_by_name = representations(grid, valid)
    table = pd.read_csv(TABLE, dtype={"sample_id": str, "group_id": str, "video_id": str})
    table = table.merge(feature_ids.assign(feature_row=np.arange(len(feature_ids))), on="sample_id", validate="one_to_one")
    order = table.feature_row.to_numpy(int)
    reps = {name: value[order] for name, value in rep_by_name.items()}
    labels = table.label.to_numpy(int)
    groups = table.group_id.astype(str).to_numpy()
    folds = table.fold.to_numpy(int)
    table["p_classify"] = table.p_classify.to_numpy(float)
    current_rank = table.video_rank.to_numpy(float)
    nested_score = np.zeros(len(table), float)
    nested_pred = np.zeros(len(table), int)
    topk_pred = np.zeros(len(table), int)
    topk_prob = table.p_classify.to_numpy(float).copy()
    selected = []
    policies = []

    for outer in sorted(np.unique(folds)):
        development_idx = np.flatnonzero(folds != outer)
        test_idx = np.flatnonzero(folds == outer)
        inner_predictions = {}
        rows = []
        for candidate in candidate_grid():
            representation, components, c_value = candidate
            prediction = np.zeros(len(development_idx), float)
            for inner in sorted(np.unique(folds[development_idx])):
                inner_test_global = np.flatnonzero((folds != outer) & (folds == inner))
                inner_train_global = np.flatnonzero((folds != outer) & (folds != inner))
                positions = np.searchsorted(development_idx, inner_test_global)
                prediction[positions] = fit_predict(
                    reps[representation], labels, groups,
                    inner_train_global, inner_test_global, components, c_value,
                )
            fold_pair = []
            for inner in sorted(np.unique(folds[development_idx])):
                mask = folds[development_idx] == inner
                fold_pair.append(pair_auc(labels[development_idx][mask], prediction[mask], groups[development_idx][mask]))
            pooled_pair = pair_auc(labels[development_idx], prediction, groups[development_idx])
            pooled_auc = roc_auc_score(labels[development_idx], prediction)
            objective = pooled_pair - 0.10 * float(np.std(fold_pair))
            rows.append({
                "representation": representation, "components": components, "C": c_value,
                "pair_auc": pooled_pair, "auc": pooled_auc,
                "fold_pair_auc": fold_pair, "objective": objective,
            })
            inner_predictions[candidate] = prediction
        best = max(rows, key=lambda row: (row["objective"], row["pair_auc"], row["auc"]))
        key = (best["representation"], best["components"], best["C"])
        development_score = inner_predictions[key]
        test_score = fit_predict(
            reps[key[0]], labels, groups, development_idx, test_idx, key[1], key[2]
        )
        threshold, inner_accuracy = threshold_for_accuracy(labels[development_idx], development_score)
        nested_score[test_idx] = test_score
        nested_pred[test_idx] = (test_score >= threshold).astype(int)

        development = table.iloc[development_idx].copy()
        test = table.iloc[test_idx].copy()
        development["video_rank"] = development_score
        test["video_rank"] = test_score
        inner_by_c, test_by_c = {}, {}
        for count_c in (0.05, 0.2, 1.0):
            inner_by_c[count_c], test_by_c[count_c] = crossfit_dist(table, int(outer), count_c)
        policy = select_policy(development, inner_by_c)
        distribution = test_by_c[policy["C"]]
        fold_pred, fold_prob, acted = decode(
            test, distribution, policy["post_thr"], policy["margin_thr"],
            policy["entropy_thr"], policy["rank_gap_thr"], policy["max_changes"],
        )
        topk_pred[test_idx] = fold_pred
        topk_prob[test_idx] = fold_prob
        policy["outer"] = int(outer)
        policy["test_acted"] = int(acted.sum())
        policies.append(policy)
        best["outer"] = int(outer)
        best["threshold"] = threshold
        best["inner_accuracy"] = inner_accuracy
        best["test_pair_auc"] = pair_auc(labels[test_idx], test_score, groups[test_idx])
        best["test_auc"] = roc_auc_score(labels[test_idx], test_score)
        selected.append(best)
        print(json.dumps(best), flush=True)

    current_pred = table.pred_pls2_topk.to_numpy(int)
    current_prob = table.prob_pls2_topk.to_numpy(float)
    table_pred = (table.p_classify.to_numpy(float) >= 0.5).astype(int)
    summary = {
        "protocol": "strict outer/inner fold V-JEPA2.1 frozen pairwise linear ranking + strict nested Top-K",
        "n": int(len(table)), "cycles": int(table.group_id.nunique()),
        "phase_valid_rate": valid.mean(axis=0).tolist(),
        "video": {
            "vjepa21": metrics(labels, nested_pred, 1 / (1 + np.exp(-nested_score))),
            "vjepa21_pair_auc": pair_auc(labels, nested_score, groups),
            "current_pair_auc": pair_auc(labels, current_rank, groups),
            "current_auc": float(roc_auc_score(labels, current_rank)),
        },
        "fusion": {
            "table": metrics(labels, table_pred, table.p_classify.to_numpy(float)),
            "current_pls2_topk": metrics(labels, current_pred, current_prob),
            "vjepa21_topk": metrics(labels, topk_pred, topk_prob),
            "delta_vjepa_vs_table": float(np.mean(topk_pred == labels) - np.mean(table_pred == labels)),
            "delta_vjepa_vs_current": float(np.mean(topk_pred == labels) - np.mean(current_pred == labels)),
            "won_vs_current": int(np.sum((topk_pred == labels) & (current_pred != labels))),
            "lost_vs_current": int(np.sum((topk_pred != labels) & (current_pred == labels))),
        },
        "selected_video_models": selected,
        "topk_policies": policies,
    }
    result = table.copy()
    result["vjepa21_rank"] = nested_score
    result["vjepa21_pred"] = nested_pred
    result["vjepa21_topk_pred"] = topk_pred
    result["vjepa21_topk_prob"] = topk_prob
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "oof_nested_rank_topk.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
