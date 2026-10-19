"""表格分支的数据契约与基线预测。

当前清晰版不从原始Excel重训历史四路教师，而是读取已经严格对齐的
表格OOF输入。这样不会把无法完整追溯的早期教师误写成新的端到端训练。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TABLE_REQUIRED_COLUMNS = {
    "sample_id",
    "group_id",
    "fold",
    "label",
    "p_classify",
    "video_rank",
    "pgt_type",
    "pred_pls2_topk",
    "prob_pls2_topk",
}


def load_table_input(path: str | Path) -> pd.DataFrame:
    """读取最终表格输入并执行不会读取测试标签统计量的结构检查。"""
    frame = pd.read_csv(
        path,
        dtype={"sample_id": str, "group_id": str, "video_id": str},
    ).reset_index(drop=True)
    missing = sorted(TABLE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"table input missing columns: {missing}")
    if frame.sample_id.duplicated().any():
        duplicated = frame.loc[frame.sample_id.duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"duplicated sample_id: {duplicated}")

    frame["label"] = frame.label.astype(int)
    frame["fold"] = frame.fold.astype(int)
    probability = frame.p_classify.to_numpy(float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("p_classify must be finite and within [0, 1]")
    if not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError("label must be binary")
    if sorted(frame.fold.unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("fold must contain exactly 0,1,2,3,4")
    if (frame.groupby("group_id").fold.nunique() != 1).any():
        raise ValueError("one group_id appears in multiple folds")
    if (frame.groupby("group_id").p_classify.nunique(dropna=False) != 1).any():
        raise ValueError("p_classify is not constant inside a cycle")
    return frame


class TableBranch:
    """提供表格概率、表格分类和历史PLS2-TopK对照。"""

    @staticmethod
    def probability(frame: pd.DataFrame) -> np.ndarray:
        return frame.p_classify.to_numpy(float)

    @staticmethod
    def prediction(frame: pd.DataFrame) -> np.ndarray:
        return (TableBranch.probability(frame) >= 0.5).astype(int)

    @staticmethod
    def historical_topk_prediction(frame: pd.DataFrame) -> np.ndarray:
        return frame.pred_pls2_topk.to_numpy(int)

    @staticmethod
    def historical_topk_probability(frame: pd.DataFrame) -> np.ndarray:
        return frame.prob_pls2_topk.to_numpy(float)

