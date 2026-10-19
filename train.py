#!/usr/bin/env python3
"""训练/选择当前最佳组合的五折决策层并生成OOF预测。

输入是已经生成的表格OOF和阶段视频OOF。本入口不重新训练早期四路表格教师，
也不重新提取V-JEPA特征；对应原归档最终融合层的可读实现。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import DEFAULT_OUTPUT_DIR, DEFAULT_PHASE_INPUT, DEFAULT_TABLE_INPUT
from model import ConservativeTopKModel
from model_table import load_table_input
from model_video import VideoRankBranch, load_phase_input, merge_inputs, pair_auc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-input", type=Path, default=DEFAULT_TABLE_INPUT)
    parser.add_argument("--phase-input", type=Path, default=DEFAULT_PHASE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = load_table_input(args.table_input)
    phase = load_phase_input(args.phase_input)
    data = merge_inputs(table, phase)

    labels = data.label.to_numpy(int)
    groups = data.group_id.astype(str).to_numpy()
    folds = data.fold.to_numpy(int)
    video = VideoRankBranch(data)

    fused_rank = np.zeros(len(data), float)
    final_prediction = np.zeros(len(data), int)
    final_probability = data.p_classify.to_numpy(float).copy()
    video_selections = []
    topk_policies = []

    for outer_fold in sorted(np.unique(folds)):
        test_mask = folds == outer_fold
        selection = video.select_for_outer_fold(labels, groups, folds, int(outer_fold))
        score = video.score(selection["weights"])
        fused_rank[test_mask] = score[test_mask]
        selection["outer"] = int(outer_fold)
        selection["test_pair_auc"] = pair_auc(
            labels[test_mask], score[test_mask], groups[test_mask]
        )
        video_selections.append(selection)

        # 数量模型不读取video_rank；Top-K解码读取当前外折选中的排序分数。
        fold_frame = data.copy()
        fold_frame["video_rank"] = score
        positions, prediction, probability, policy = ConservativeTopKModel().fit_predict_outer(
            fold_frame, int(outer_fold)
        )
        final_prediction[positions] = prediction
        final_probability[positions] = probability
        topk_policies.append(policy)
        print(json.dumps({"outer_fold": int(outer_fold), "video": selection, "topk": policy}), flush=True)

    output = data.copy()
    output["phasewise_fused_rank"] = fused_rank
    output["phasewise_topk_pred"] = final_prediction
    output["phasewise_topk_prob"] = final_probability
    prediction_path = args.output_dir / "oof_predictions.csv"
    output.to_csv(prediction_path, index=False)

    artifact = {
        "protocol": "five-fold phase-rank selection + nested ordinal count + conservative Top-K",
        "table_input": str(args.table_input),
        "phase_input": str(args.phase_input),
        "n_samples": int(len(data)),
        "n_cycles": int(data.group_id.nunique()),
        "video_selections": video_selections,
        "topk_policies": topk_policies,
    }
    artifact_path = args.output_dir / "training_artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"predictions": str(prediction_path), "artifact": str(artifact_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

