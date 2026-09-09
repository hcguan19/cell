#!/usr/bin/env python3
# 阶段视觉表征候选训练：读取冻结特征，构造全局/时差/空间表示。
# 每个候选固定PCA32 + 成对LogReg(C=0.03)，保存五折排序分数。
"""Fixed five-fold screen of phase-wise continuous V-JEPA2.1 features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("/data0/guanhuancong/cell/cell_train3")
FEATURES = ROOT / "vjepa21_vitb_video_v3_phasewise/features"
META = ROOT / "cycle_count_topk_bestvideo_v1/oof_discrete_topk.csv"
OUT = ROOT / "vjepa21_vitb_video_v3_phasewise"

sys.path.insert(0, str(ROOT))
from evaluate_vjepa21_nested_rank_topk import fit_predict

sys.path.insert(0, str(ROOT / "raw_video_sibling_ssl_v1"))
from common import pair_auc


def load_features(n):
    # 每个样本：(4阶段, 8个时间tubelet, 3×3空间格, 768通道)。
    # rows分片保存全局行号，用它恢复META的顺序，不能简单串联分片。
    # 当前seen只检查覆盖，未检查跨分片重复行；应在前置审计中核对唯一性。
    result = np.empty((n, 4, 8, 9, 768), dtype=np.float16)
    valid = np.empty((n, 4), dtype=bool)
    seen = np.zeros(n, dtype=bool)
    for shard in range(6):
        rows = np.load(FEATURES / f"rows_{shard}.npy")
        result[rows] = np.load(FEATURES / f"features_{shard}.npy")
        valid[rows] = np.load(FEATURES / f"phase_valid_{shard}.npy")
        seen[rows] = True
    if not seen.all():
        raise ValueError(f"missing {int((~seen).sum())} rows")
    return result, valid


def evaluate(name, features, components, data, output, summary):
    # 每个表征单独固定五折训练，PCA与线性排序头仅拟合其训练折。
    # 输出是排序分数，不是概率；没有在这里寻找分类阈值。
    labels = data.label.to_numpy(int)
    groups = data.group_id.astype(str).to_numpy()
    folds = data.fold.to_numpy(int)
    prediction = np.zeros(len(data), float)
    fold_rows = []
    for outer in sorted(np.unique(folds)):
        train, test = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        prediction[test] = fit_predict(features, labels, groups, train, test, components, 0.03)
        fold_rows.append({
            "fold": int(outer),
            "auc": float(roc_auc_score(labels[test], prediction[test])),
            "pair_auc": pair_auc(labels[test], prediction[test], groups[test]),
        })
    row = {
        "components": components, "C": 0.03,
        "auc": float(roc_auc_score(labels, prediction)),
        "pair_auc": pair_auc(labels, prediction, groups), "folds": fold_rows,
    }
    summary["models"][name] = row
    output[name] = prediction
    print(name, json.dumps(row), flush=True)


def main():
    # 从冻结视觉特征构造多个低容量候选，供下一层小残差选择使用。
    data = pd.read_csv(META, dtype={"sample_id": str, "group_id": str})
    grid, valid = load_features(len(data))
    labels = data.label.to_numpy(int)
    groups = data.group_id.astype(str).to_numpy()
    # 空间平均得到阶段内时间序列；时间平均得到每阶段全局表示。
    # dynamics用相邻tubelet差分，spatial保留3×3位置但平均掉时间。
    temporal = grid.astype(np.float32).mean(axis=3)  # N, phase, tubelet, dim
    phase = temporal.mean(axis=2)
    dynamics = np.diff(temporal, axis=2)
    spatial = grid.astype(np.float32).mean(axis=2)
    mask = valid.astype(np.float32)
    # early实际取前3个阶段，不是仅0–100h第一阶段。
    # 无效阶段的特征为0，同时附加mask供排序器区分缺失与真实零特征。
    representations = {
        "phase_global_all": np.concatenate([phase.reshape(len(data), -1), mask], axis=1),
        "phase_global_early": np.concatenate([phase[:, :3].reshape(len(data), -1), mask[:, :3]], axis=1),
        "within_dynamics_all": np.concatenate([dynamics.reshape(len(data), -1), mask], axis=1),
        "within_dynamics_early": np.concatenate([dynamics[:, :3].reshape(len(data), -1), mask[:, :3]], axis=1),
        "spatial_early": np.concatenate([spatial[:, :3].reshape(len(data), -1), mask[:, :3]], axis=1),
    }
    summary = {
        "protocol": "fixed strict 5-fold phase-wise continuous V-JEPA2.1 pairwise ranking",
        "current_pair_auc": pair_auc(labels, data.video_rank.to_numpy(float), groups),
        "current_auc": float(roc_auc_score(labels, data.video_rank.to_numpy(float))),
        "phase_valid_rate": valid.mean(0).tolist(), "models": {},
    }
    output = data[["sample_id", "group_id", "fold", "label"]].copy()
    # 所有候选固定PCA32、C=0.03。后续选择复用这些OOF时，
    # 开发折分数的训练来源可能含当前下游外折，需单独审计跨层泄漏。
    for name, features in representations.items():
        evaluate(name, features, 32, data, output, summary)
    OUT.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT / "oof_fixed_screen.csv", index=False)
    (OUT / "fixed_screen_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
