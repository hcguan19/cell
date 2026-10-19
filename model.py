"""周期阳性数量模型与保守Top-K决策。

该模块只负责可解释的低容量决策层：
1. 将每个周期展开为 I(K>=j) 有序二分类任务；
2. 预测离散数量后验 P(K)；
3. 在后验和视频排序都足够可信时执行Top-K，否则回退表格。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import COUNT_MODEL_C_VALUES, RANDOM_SEED


COUNT_NUMERIC_FEATURES = ["p", "logit_p", "n", "np", "j", "j_frac", "p_minus_jfrac"]
COUNT_CATEGORICAL_FEATURES = ["pgt_type"]


def logit(probability):
    probability = np.clip(np.asarray(probability, float), 1e-5, 1 - 1e-5)
    return np.log(probability / (1 - probability))


def cycle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in frame.groupby("group_id", sort=False):
        probability = float(group.p_classify.iloc[0])
        size = int(len(group))
        rows.append({
            "group_id": group_id,
            "fold": int(group.fold.iloc[0]),
            "n": size,
            "p": probability,
            "logit_p": float(logit([probability])[0]),
            "np": size * probability,
            "pgt_type": str(group.pgt_type.iloc[0]),
            "k_true": int(group.label.sum()),
        })
    return pd.DataFrame(rows)


def expand_ordinal(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle in cycles.itertuples(index=False):
        for j in range(1, int(cycle.n) + 1):
            rows.append({
                "group_id": cycle.group_id,
                "p": cycle.p,
                "logit_p": cycle.logit_p,
                "n": cycle.n,
                "np": cycle.np,
                "pgt_type": cycle.pgt_type,
                "j": j,
                "j_frac": j / cycle.n,
                "p_minus_jfrac": cycle.p - j / cycle.n,
                "target": int(cycle.k_true >= j),
            })
    return pd.DataFrame(rows)


def fit_count_model(cycles: pd.DataFrame, c_value: float):
    expanded = expand_ordinal(cycles)
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), COUNT_NUMERIC_FEATURES),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            COUNT_CATEGORICAL_FEATURES,
        ),
    ])
    model = make_pipeline(
        preprocessor,
        LogisticRegression(
            C=c_value,
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )
    model.fit(
        expanded[COUNT_NUMERIC_FEATURES + COUNT_CATEGORICAL_FEATURES],
        expanded.target,
    )
    return model


def predict_count_distribution(model, cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle in cycles.itertuples(index=False):
        j = np.arange(1, int(cycle.n) + 1)
        features = pd.DataFrame({
            "p": cycle.p,
            "logit_p": cycle.logit_p,
            "n": cycle.n,
            "np": cycle.np,
            "pgt_type": cycle.pgt_type,
            "j": j,
            "j_frac": j / cycle.n,
            "p_minus_jfrac": cycle.p - j / cycle.n,
        })
        survival = model.predict_proba(
            features[COUNT_NUMERIC_FEATURES + COUNT_CATEGORICAL_FEATURES]
        )[:, 1]
        survival = np.minimum.accumulate(survival)
        posterior = np.empty(int(cycle.n) + 1, float)
        posterior[0] = 1 - survival[0]
        if cycle.n > 1:
            posterior[1:-1] = survival[:-1] - survival[1:]
        posterior[-1] = survival[-1]
        posterior = np.clip(posterior, 0, None)
        posterior /= posterior.sum()
        order = np.argsort(-posterior)
        entropy = -np.sum(posterior * np.log(posterior + 1e-12)) / np.log(len(posterior))
        rows.append({
            "group_id": cycle.group_id,
            "fold": int(cycle.fold),
            "n": int(cycle.n),
            "k_true": int(cycle.k_true),
            "k_mode": int(order[0]),
            "k_second": int(order[1]),
            "post_max": float(posterior[order[0]]),
            "post_margin": float(posterior[order[0]] - posterior[order[1]]),
            "post_entropy": float(entropy),
            "posterior": json.dumps(posterior.tolist(), separators=(",", ":")),
        })
    return pd.DataFrame(rows)


def crossfit_count_distribution(frame: pd.DataFrame, outer_fold: int, c_value: float):
    """开发折产生内层OOF后验，外折产生一次最终后验。"""
    cycles = cycle_frame(frame)
    development_folds = sorted(set(cycles.fold) - {outer_fold})
    inner = []
    for validation_fold in development_folds:
        train = cycles[(cycles.fold != outer_fold) & (cycles.fold != validation_fold)]
        validation = cycles[cycles.fold == validation_fold]
        inner.append(predict_count_distribution(fit_count_model(train, c_value), validation))
    train = cycles[cycles.fold != outer_fold]
    test = cycles[cycles.fold == outer_fold]
    outer_prediction = predict_count_distribution(fit_count_model(train, c_value), test)
    return pd.concat(inner, ignore_index=True), outer_prediction


def boundary_rank_gap(group: pd.DataFrame, k: int) -> float:
    n = len(group)
    if k <= 0 or k >= n:
        return -1e9
    scores = np.sort(group.video_rank.to_numpy(float))[::-1]
    return float((scores[k - 1] - scores[k]) / (np.std(scores) + 1e-8))


def decode_topk(
    frame: pd.DataFrame,
    distribution: pd.DataFrame,
    post_threshold: float,
    margin_threshold: float,
    entropy_threshold: float,
    rank_gap_threshold: float,
    max_count_change: int,
):
    lookup = distribution.set_index("group_id")
    base_prediction = (frame.p_classify.to_numpy(float) >= 0.5).astype(int)
    prediction = base_prediction.copy()
    probability = frame.p_classify.to_numpy(float).copy()
    acted = np.zeros(len(frame), bool)

    for group_id, labels in frame.groupby("group_id", sort=False).groups.items():
        positions = frame.index.get_indexer(labels)
        row = lookup.loc[group_id]
        n, k = int(row.n), int(row.k_mode)
        base_k = int(base_prediction[positions].sum())
        gap = boundary_rank_gap(frame.loc[labels], k)
        eligible = (
            0 < k < n
            and k != base_k
            and abs(k - base_k) <= max_count_change
            and row.post_max >= post_threshold
            and row.post_margin >= margin_threshold
            and row.post_entropy <= entropy_threshold
            and gap >= rank_gap_threshold
        )
        if not eligible:
            continue

        scores = frame.loc[labels, "video_rank"].to_numpy(float)
        order = np.argsort(-scores, kind="stable")
        local = np.zeros(n, int)
        local[order[:k]] = 1
        prediction[positions] = local

        ranks = np.empty(n, int)
        ranks[order] = np.arange(n)
        centered = (k - 0.5 - ranks) / max(1.0, n / 3.0)
        probability[positions] = 1 / (1 + np.exp(-centered))
        acted[positions] = True
    return prediction, probability, acted


def select_topk_policy(development: pd.DataFrame, candidates: dict[float, pd.DataFrame]) -> dict:
    labels = development.label.to_numpy(int)
    base = (development.p_classify.to_numpy(float) >= 0.5).astype(int)
    base_accuracy = accuracy_score(labels, base)
    rows = []
    for c_value, distribution in candidates.items():
        for post_threshold in (0.20, 0.25, 0.30, 0.35, 0.40):
            for margin_threshold in (0.02, 0.05, 0.10, 0.15):
                for entropy_threshold in (0.65, 0.75, 0.85, 0.95):
                    for rank_gap_threshold in (0.0, 0.25, 0.5, 0.75):
                        for max_count_change in (1, 2):
                            prediction, _, acted = decode_topk(
                                development,
                                distribution,
                                post_threshold,
                                margin_threshold,
                                entropy_threshold,
                                rank_gap_threshold,
                                max_count_change,
                            )
                            accuracy = accuracy_score(labels, prediction)
                            fold_deltas = []
                            for fold in sorted(development.fold.unique()):
                                mask = development.fold.to_numpy() == fold
                                fold_deltas.append(float(
                                    np.mean(prediction[mask] == labels[mask])
                                    - np.mean(base[mask] == labels[mask])
                                ))
                            rows.append({
                                "C": c_value,
                                "post_thr": post_threshold,
                                "margin_thr": margin_threshold,
                                "entropy_thr": entropy_threshold,
                                "rank_gap_thr": rank_gap_threshold,
                                "max_changes": max_count_change,
                                "acc": float(accuracy),
                                "delta": float(accuracy - base_accuracy),
                                "action_frac": float(acted.mean()),
                                "fold_deltas": fold_deltas,
                                "positive_folds": int(np.sum(np.asarray(fold_deltas) > 0)),
                                "nonnegative_folds": int(np.sum(np.asarray(fold_deltas) >= 0)),
                            })
    valid = [
        row for row in rows
        if row["delta"] >= 2 / len(development)
        and row["positive_folds"] >= 2
        and row["nonnegative_folds"] == len(row["fold_deltas"])
    ]
    if not valid:
        return {
            "fallback": True,
            "C": 0.2,
            "post_thr": 2.0,
            "margin_thr": 2.0,
            "entropy_thr": 0.0,
            "rank_gap_thr": 1e9,
            "max_changes": 0,
            "acc": float(base_accuracy),
            "delta": 0.0,
            "action_frac": 0.0,
            "fold_deltas": [0.0] * 4,
            "positive_folds": 0,
            "nonnegative_folds": 4,
        }
    best = max(valid, key=lambda row: (row["acc"] - 0.004 * row["action_frac"], -row["action_frac"]))
    best["fallback"] = False
    return best


def classification_metrics(labels, prediction, probability) -> dict:
    labels = np.asarray(labels, int)
    prediction = np.asarray(prediction, int)
    probability = np.asarray(probability, float)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


class ConservativeTopKModel:
    """对一个外折训练数量模型、选择策略并预测该外折。"""

    def fit_predict_outer(self, frame: pd.DataFrame, outer_fold: int):
        development = frame[frame.fold.ne(outer_fold)].copy()
        test = frame[frame.fold.eq(outer_fold)].copy()
        inner_by_c, test_by_c = {}, {}
        for c_value in COUNT_MODEL_C_VALUES:
            inner_by_c[c_value], test_by_c[c_value] = crossfit_count_distribution(
                frame, outer_fold, c_value
            )
        policy = select_topk_policy(development, inner_by_c)
        distribution = test_by_c[policy["C"]]
        prediction, probability, acted = decode_topk(
            test,
            distribution,
            policy["post_thr"],
            policy["margin_thr"],
            policy["entropy_thr"],
            policy["rank_gap_thr"],
            policy["max_changes"],
        )
        policy = dict(policy)
        policy["outer"] = int(outer_fold)
        policy["test_acted"] = int(acted.sum())
        return test.index.to_numpy(), prediction, probability, policy

