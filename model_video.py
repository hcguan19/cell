"""阶段视频排序分数的标准化、组合与外折候选选择。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import VIDEO_CANDIDATES


PHASE_KEYS = sorted({key for weights in VIDEO_CANDIDATES.values() for key in weights})
PHASE_REQUIRED_COLUMNS = {"sample_id", "group_id", "fold", "label", *PHASE_KEYS}


def load_phase_input(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str, "group_id": str}).reset_index(drop=True)
    missing = sorted(PHASE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"phase input missing columns: {missing}")
    if frame.sample_id.duplicated().any():
        raise ValueError("phase input has duplicated sample_id")
    return frame


def merge_inputs(table: pd.DataFrame, phase: pd.DataFrame) -> pd.DataFrame:
    """显式检查覆盖率后按样本、周期、折号和标签一对一连接。"""
    keys = ["sample_id", "group_id", "fold", "label"]
    table_keys = set(map(tuple, table[keys].to_numpy()))
    phase_keys = set(map(tuple, phase[keys].to_numpy()))
    if table_keys != phase_keys:
        raise ValueError(
            f"table/phase key mismatch: table_only={len(table_keys-phase_keys)}, "
            f"phase_only={len(phase_keys-table_keys)}"
        )
    return table.merge(phase, on=keys, how="inner", validate="one_to_one").reset_index(drop=True)


def pair_auc(labels, scores, groups) -> float:
    """计算所有同周期正负样本对的排序正确率。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    values = []
    for group in sorted(set(groups)):
        idx = np.flatnonzero(groups == group)
        pos = scores[idx][labels[idx] == 1]
        neg = scores[idx][labels[idx] == 0]
        if len(pos) and len(neg):
            diff = (pos[:, None] - neg[None, :]).ravel()
            values.extend((diff > 0) + 0.5 * (diff == 0))
    return float(np.mean(values)) if values else 0.5


def standardize_by_fold(values, folds) -> np.ndarray:
    """保持历史算法：使用各评价折整批无标签分数做标准化。"""
    values = np.asarray(values, float)
    folds = np.asarray(folds, int)
    result = np.zeros(len(values), float)
    for fold in np.unique(folds):
        mask = folds == fold
        result[mask] = (values[mask] - values[mask].mean()) / (values[mask].std() + 1e-8)
    return result


class VideoRankBranch:
    """管理当前排序分数和固定的小残差候选。"""

    def __init__(self, frame: pd.DataFrame):
        folds = frame.fold.to_numpy(int)
        self.values = {"current": standardize_by_fold(frame.video_rank.to_numpy(float), folds)}
        for feature in PHASE_KEYS:
            self.values[feature] = standardize_by_fold(frame[feature].to_numpy(float), folds)

    def score(self, weights: dict[str, float]) -> np.ndarray:
        result = self.values["current"].copy()
        for feature, weight in weights.items():
            result += float(weight) * self.values[feature]
        return result

    def select_for_outer_fold(self, labels, groups, folds, outer: int) -> dict:
        labels = np.asarray(labels, int)
        groups = np.asarray(groups, str)
        folds = np.asarray(folds, int)
        development = folds != outer
        development_folds = sorted(np.unique(folds[development]))
        baseline = np.asarray([
            pair_auc(labels[folds == fold], self.values["current"][folds == fold], groups[folds == fold])
            for fold in development_folds
        ])
        candidates = []
        for name, weights in VIDEO_CANDIDATES.items():
            score = self.score(weights)
            fold_pair = np.asarray([
                pair_auc(labels[folds == fold], score[folds == fold], groups[folds == fold])
                for fold in development_folds
            ])
            l1 = sum(abs(weight) for weight in weights.values())
            pooled = pair_auc(labels[development], score[development], groups[development])
            candidates.append({
                "name": name,
                "weights": weights,
                "pair_auc": pooled,
                "fold_pair_auc": fold_pair.tolist(),
                "positive_folds": int(np.sum(fold_pair > baseline)),
                "nonnegative_folds": int(np.sum(fold_pair >= baseline)),
                "worst_delta": float(np.min(fold_pair - baseline)),
                "objective": float(pooled - 0.05 * np.std(fold_pair) - 0.002 * l1),
            })
        valid = [
            row for row in candidates
            if row["positive_folds"] >= 2
            and row["nonnegative_folds"] >= 3
            and row["worst_delta"] >= -0.015
        ]
        chosen = max(valid, key=lambda row: row["objective"]) if valid else candidates[0]
        return dict(chosen)

