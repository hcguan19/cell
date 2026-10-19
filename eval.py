#!/usr/bin/env python3
"""评价train.py生成的OOF预测，不执行任何训练或策略选择。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DEFAULT_OUTPUT_DIR
from model import classification_metrics
from model_table import TableBranch
from model_video import pair_auc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "oof_predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "metrics.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.predictions, dtype={"sample_id": str, "group_id": str})
    required = {
        "label", "group_id", "p_classify", "video_rank",
        "pred_pls2_topk", "prob_pls2_topk",
        "phasewise_fused_rank", "phasewise_topk_pred", "phasewise_topk_prob",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction file missing columns: {missing}")

    labels = frame.label.to_numpy(int)
    groups = frame.group_id.astype(str).to_numpy()
    table_prediction = TableBranch.prediction(frame)
    table_probability = TableBranch.probability(frame)
    historical_prediction = TableBranch.historical_topk_prediction(frame)
    historical_probability = TableBranch.historical_topk_probability(frame)
    final_prediction = frame.phasewise_topk_pred.to_numpy(int)
    final_probability = frame.phasewise_topk_prob.to_numpy(float)

    summary = {
        "n_samples": int(len(frame)),
        "n_cycles": int(frame.group_id.nunique()),
        "classification": {
            "table": classification_metrics(labels, table_prediction, table_probability),
            "historical_pls2_topk": classification_metrics(
                labels, historical_prediction, historical_probability
            ),
            "phasewise_topk": classification_metrics(
                labels, final_prediction, final_probability
            ),
        },
        "ranking": {
            "current_pair_auc": pair_auc(labels, frame.video_rank, groups),
            "phasewise_pair_auc": pair_auc(labels, frame.phasewise_fused_rank, groups),
            "current_auc": float(roc_auc_score(labels, frame.video_rank)),
            "phasewise_auc": float(roc_auc_score(labels, frame.phasewise_fused_rank)),
        },
        "comparison": {
            "won_vs_historical_topk": int(
                ((final_prediction == labels) & (historical_prediction != labels)).sum()
            ),
            "lost_vs_historical_topk": int(
                ((final_prediction != labels) & (historical_prediction == labels)).sum()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

