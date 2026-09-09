#!/usr/bin/env python3
# 最终融合入口：阶段V-JEPA排序小修正 + 周期数量分布 + 保守Top-K。
# 输入为两个已生成的私有OOF文件，输出nested_topk/oof.csv及summary.json。
# 本文件对应归档ACC 0.67419，不负责从原始数据重训整个上游。
# 注释版不改变算法。评估限制、输入约定和运行顺序见docs/目录。
"""Conservative leave-one-fold selection of phase-wise V-JEPA residual ranks."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/data0/guanhuancong/cell/cell_train3")
OUT = ROOT / "vjepa21_vitb_video_v3_phasewise/nested_topk"
sys.path.insert(0, str(ROOT / "raw_video_sibling_ssl_v1"))
from common import pair_auc
sys.path.insert(0, str(ROOT))
from evaluate_updated_table_video_topk import crossfit_dist, decode, select_policy
sys.path.insert(0, str(ROOT / "cycle_count_topk_bestvideo_v1"))
from run_best_video_topk import metrics

# 候选是“已有排序分数 + 小权重阶段分数”，不是重新训练大网络。
# current 是零修正回退项；负权重候选只是一项历史探索，不代表特征无价值。
CANDIDATES = {
    "current": {},
    "global05": {"phase_global_all": 0.05},
    "global10": {"phase_global_all": 0.10},
    "early05": {"phase_global_early": 0.05},
    "early075": {"phase_global_early": 0.075},
    "dynneg025": {"within_dynamics_all": -0.025},
    "dynneg05": {"within_dynamics_all": -0.05},
    "spatial025": {"spatial_early": 0.025},
    "spatial05": {"spatial_early": 0.05},
    "spatial075": {"spatial_early": 0.075},
    "spatial05_dynneg025": {"spatial_early": 0.05, "within_dynamics_all": -0.025},
    "global05_dynneg025": {"phase_global_all": 0.05, "within_dynamics_all": -0.025},
}


def standardize(values, folds):
    # 分数在各自折内转成零均值、单位方差，便于不同来源的分数相加。
    # 注意：这里使用该折整批无标签分数的统计量，是转导式批次变换。
    # 它不是训练折拟合的部署 scaler，不能直接声称支持单样本在线推理。
    result = np.zeros(len(values), float)
    for fold in np.unique(folds):
        mask = folds == fold
        result[mask] = (values[mask] - values[mask].mean()) / (values[mask].std() + 1e-8)
    return result


def main():
    # 上游必须先生成两个私有中间文件。此入口不训练四路表格专家，
    # 也不从原始视频开始训练；它消费已保存的表格概率和阶段排序 OOF。
    table = pd.read_csv(ROOT / "qwen_pls2_video_topk_v1/oof_pls2_topk.csv",
                        dtype={"sample_id": str, "group_id": str, "video_id": str})
    phase = pd.read_csv(ROOT / "vjepa21_vitb_video_v3_phasewise/oof_fixed_screen.csv",
                        dtype={"sample_id": str})
    # one_to_one 防止重复行膨胀。默认 inner merge 会丢弃未匹配行，
    # 历史1240例分母来源于既有对齐流程，运行前应独立审计覆盖率。
    data = table.merge(phase, on=["sample_id", "group_id", "fold", "label"], validate="one_to_one")
    y = data.label.to_numpy(int)
    groups = data.group_id.astype(str).to_numpy()
    folds = data.fold.to_numpy(int)
    values = {"current": standardize(data.video_rank.to_numpy(float), folds)}
    for feature in {key for weights in CANDIDATES.values() for key in weights}:
        values[feature] = standardize(data[feature].to_numpy(float), folds)

    def candidate_score(weights):
        # 所有分数均与 data 行顺序一致；权重只来自预先列出的有限候选。
        score = values["current"].copy()
        for feature, weight in weights.items():
            score += weight * values[feature]
        return score

    fused = np.zeros(len(data), float)
    topk_pred = np.zeros(len(data), int)
    topk_prob = data.p_classify.to_numpy(float).copy()
    selections, policies = [], []
    for outer in sorted(np.unique(folds)):
        # 当前外折只用于最后评价。候选排序权重和解码策略在其他折上选择。
        # 这只保证本层选择隔离，不能自动证明上游堆叠模型也严格嵌套。
        development, test = folds != outer, folds == outer
        dev_folds = sorted(np.unique(folds[development]))
        base_fold = np.asarray([pair_auc(y[folds == f], values["current"][folds == f], groups[folds == f]) for f in dev_folds])
        rows = []
        for name, weights in CANDIDATES.items():
            score = candidate_score(weights)
            fold_pair = np.asarray([pair_auc(y[folds == f], score[folds == f], groups[folds == f]) for f in dev_folds])
            l1 = sum(abs(weight) for weight in weights.values())
            rows.append({"name": name, "weights": weights,
                         "pair_auc": pair_auc(y[development], score[development], groups[development]),
                         "fold_pair_auc": fold_pair.tolist(),
                         "positive_folds": int(np.sum(fold_pair > base_fold)),
                         "nonnegative_folds": int(np.sum(fold_pair >= base_fold)),
                         "worst_delta": float(np.min(fold_pair - base_fold)),
                         "objective": float(pair_auc(y[development], score[development], groups[development])
                                            - 0.05*np.std(fold_pair) - 0.002*l1)})
        # 至少两折提高、至少三折不下降、最差折损失不超过0.015。
        # 目标中再惩罚折间波动和权重L1，抑制单折偶然提升。
        valid = [row for row in rows if row["positive_folds"] >= 2 and row["nonnegative_folds"] >= 3
                 and row["worst_delta"] >= -0.015]
        chosen = max(valid, key=lambda row: row["objective"]) if valid else rows[0]
        score = candidate_score(chosen["weights"])
        fused[test] = score[test]
        chosen["outer"] = int(outer)
        # 测试指标在 chosen 确定以后记录，不进入上述候选筛选条件。
        chosen["test_pair_auc"] = pair_auc(y[test], score[test], groups[test])
        selections.append(chosen)

        dev_frame, test_frame = data.loc[development].copy(), data.loc[test].copy()
        dev_frame["video_rank"], test_frame["video_rank"] = score[development], score[test]
        inner_by_c, test_by_c = {}, {}
        # C 是数量模型 LogisticRegression 的逆正则强度，不是视频权重。
        # crossfit_dist 为开发部分产生内层折外数量预测，供策略筛选使用。
        for count_c in (0.05, 0.2, 1.0):
            inner_by_c[count_c], test_by_c[count_c] = crossfit_dist(data, int(outer), count_c)
        policy = select_policy(dev_frame, inner_by_c)
        distribution = test_by_c[policy["C"]]
        pred, prob, acted = decode(test_frame, distribution, policy["post_thr"], policy["margin_thr"],
                                   policy["entropy_thr"], policy["rank_gap_thr"], policy["max_changes"])
        topk_pred[test], topk_prob[test] = pred, prob
        policy["outer"], policy["test_acted"] = int(outer), int(acted.sum())
        policies.append(policy)
        print(json.dumps({"selection": chosen, "policy": policy}), flush=True)

    current_pred = data.pred_pls2_topk.to_numpy(int)
    table_pred = (data.p_classify.to_numpy(float) >= 0.5).astype(int)
    # 排序AUC/Pairs-AUC与最终分类AUC分别计算，不应混用。
    # 历史protocol文本保留；其“strict”字样不是全链路无泄漏的证明。
    summary = {
        "protocol": "leave-one-fold conservative phase-wise V-JEPA residual selection + strict Top-K",
        "video": {"current_pair_auc": pair_auc(y, values["current"], groups),
                  "fused_pair_auc": pair_auc(y, fused, groups),
                  "current_auc": float(roc_auc_score(y, values["current"])),
                  "fused_auc": float(roc_auc_score(y, fused))},
        "fusion": {"table": metrics(y, table_pred, data.p_classify.to_numpy(float)),
                   "current_topk": metrics(y, current_pred, data.prob_pls2_topk.to_numpy(float)),
                   "phasewise_topk": metrics(y, topk_pred, topk_prob),
                   "delta_vs_current": float(np.mean(topk_pred == y)-np.mean(current_pred == y)),
                   "won_vs_current": int(np.sum((topk_pred == y)&(current_pred != y))),
                   "lost_vs_current": int(np.sum((topk_pred != y)&(current_pred == y)))},
        "selections": selections, "policies": policies,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result = data.copy()
    result["phasewise_fused_rank"], result["phasewise_topk_pred"], result["phasewise_topk_prob"] = fused, topk_pred, topk_prob
    # 此CSV含逐样本标识和标签，只能保存在私有工作区，禁止提交到Git。
    result.to_csv(OUT / "oof.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
