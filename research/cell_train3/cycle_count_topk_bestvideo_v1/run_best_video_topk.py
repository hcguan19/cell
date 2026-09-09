#!/usr/bin/env python3
# 核心数量模型与Top-K决策实现，建议先阅读expand_ordinal和decode。
# 本层的嵌套选择不自动保证上游OOF全链路隔离，详见评估限制说明。
"""Discrete P(K|cycle) + video rank + conservative Top-K, strict nested OOF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "cycle_count_topk_v1"))
from run_cycle_count_topk import cycle_frame, paired_cycle_bootstrap  # noqa: E402

INPUT = HERE / "input_best_video.csv"
SEED = 20260822
# p为周期表格概率，n为周期样本数，np=n*p，j为“至少j个阳性”的阈值。
# pgt_type只作为类别条件。k_true由标签汇总，仅用于训练目标及事后诊断。
NUM = ["p", "logit_p", "n", "np", "j", "j_frac", "p_minus_jfrac"]
CAT = ["pgt_type"]


def expand_ordinal(cycles: pd.DataFrame) -> pd.DataFrame:
    # 将每个含n个样本的周期展开成n条有序二分类记录：目标I(K>=j)。
    # 大周期贡献更多训练行，这是历史实现的隐式加权，不能解释成等周期权重。
    rows = []
    for r in cycles.itertuples(index=False):
        for j in range(1, int(r.n) + 1):
            rows.append({"group_id": r.group_id, "p": r.p, "logit_p": r.logit_p,
                         "n": r.n, "np": r.np, "pgt_type": r.pgt_type,
                         "j": j, "j_frac": j / r.n, "p_minus_jfrac": r.p - j / r.n,
                         "target": int(r.k_true >= j)})
    return pd.DataFrame(rows)


def fit_ordinal(cycles: pd.DataFrame, C: float):
    # StandardScaler与OneHotEncoder随当前训练子集拟合；未知类别不报错。
    # class_weight=balanced 会改变概率校准，输出需结合验证门控使用。
    z = expand_ordinal(cycles)
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    model = make_pipeline(prep, LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                                    random_state=SEED))
    model.fit(z[NUM + CAT], z.target)
    return model


def predict_distributions(model, cycles: pd.DataFrame) -> pd.DataFrame:
    # 输入仅使用NUM+CAT。k_true会写入返回表作评价，但不传给predict_proba。
    # 先预测S(j)=P(K>=j)，再用相邻差分还原K=0,...,n的离散分布。
    rows = []
    for r in cycles.itertuples(index=False):
        js = np.arange(1, int(r.n) + 1)
        x = pd.DataFrame({"p": r.p, "logit_p": r.logit_p, "n": r.n, "np": r.np,
                          "pgt_type": r.pgt_type, "j": js, "j_frac": js / r.n,
                          "p_minus_jfrac": r.p - js / r.n})
        surv = model.predict_proba(x[NUM + CAT])[:, 1]
        # Project onto non-increasing cumulative probabilities.
        # 最小值累积保证 S(1)>=S(2)>=...，这不是完整的等距回归校准。
        surv = np.minimum.accumulate(surv)
        probs = np.empty(int(r.n) + 1, float)
        probs[0] = 1.0 - surv[0]
        if r.n > 1:
            probs[1:-1] = surv[:-1] - surv[1:]
        probs[-1] = surv[-1]
        probs = np.clip(probs, 0, None); probs /= probs.sum()
        order = np.argsort(-probs)
        # 归一化熵越大表示后验越分散；最大后验与前两名差值衡量置信度。
        entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs))
        rows.append({"group_id": r.group_id, "fold": int(r.fold), "n": int(r.n),
                     "k_true": int(r.k_true), "k_mode": int(order[0]),
                     "k_second": int(order[1]), "post_max": float(probs[order[0]]),
                     "post_margin": float(probs[order[0]] - probs[order[1]]),
                     "post_entropy": float(entropy),
                     "posterior": json.dumps(probs.tolist(), separators=(",", ":"))})
    return pd.DataFrame(rows)


def boundary_rank_gap(x: pd.DataFrame, k: int) -> float:
    # 比较第K名与第K+1名的分数间隔，并除以周期内分数标准差。
    # K=0或K=n没有内部边界，返回极小值使保守策略拒绝执行。
    n = len(x)
    if k <= 0 or k >= n:
        return -1e9
    s = np.sort(x.video_rank.to_numpy(float))[::-1]
    sd = np.std(s)
    return float((s[k - 1] - s[k]) / (sd + 1e-8))


def decode(df: pd.DataFrame, dist: pd.DataFrame, post_thr: float, margin_thr: float,
           entropy_thr: float, rank_gap_thr: float, max_changes: int):
    # 返回(pred, prob, acted)，长度和顺序与输入df一致。
    # 默认复制表格概率和0.5阈值分类；任何门控条件失败都原样回退。
    # 注意prob：改判周期使用由名次构造的sigmoid分数，不是校准后的后验概率。
    lookup = dist.set_index("group_id")
    base = (df.p_classify.to_numpy(float) >= 0.5).astype(int)
    pred = base.copy(); prob = df.p_classify.to_numpy(float).copy()
    acted = np.zeros(len(df), bool)
    for gid, idx in df.groupby("group_id", sort=False).groups.items():
        # groupby给出DataFrame索引标签，get_indexer转成输出数组的整数位置。
        pos = df.index.get_indexer(idx); row = lookup.loc[gid]
        n, k = int(row.n), int(row.k_mode); base_k = int(base[pos].sum())
        gap = boundary_rank_gap(df.loc[idx], k)
        # 同时限制阳性数量变化、数量后验、后验间隔、熵与排序边界。
        # max_changes实际限制|K-base_K|，不是逐样本Hamming距离。
        # 当同周期表格预测一致时两者等价；若输入不一致则不能保证只改1–2例。
        ok = (0 < k < n and k != base_k and abs(k - base_k) <= max_changes
              and row.post_max >= post_thr and row.post_margin >= margin_thr
              and row.post_entropy <= entropy_thr and gap >= rank_gap_thr)
        if not ok:
            continue
        scores = df.loc[idx, "video_rank"].to_numpy(float)
        # 稳定排序让相同分数保留输入次序；前K个判正，其余判负。
        order = np.argsort(-scores, kind="stable")
        local = np.zeros(n, int); local[order[:k]] = 1; pred[pos] = local
        ranks = np.empty(n, int); ranks[order] = np.arange(n)
        centered = (k - 0.5 - ranks) / max(1.0, n / 3.0)
        # 0.5分界与Top-K类别一致。acted标记整个执行周期，
        # 并非“该样本类别发生变化”；变更数应另算pred != base。
        prob[pos] = 1 / (1 + np.exp(-centered)); acted[pos] = True
    return pred, prob, acted


def crossfit_dist(df: pd.DataFrame, outer: int, C: float):
    # 外层f为最终留出折。开发部分内的vf也留出：
    # 拟合使用 folds\{f,vf}，预测vf；再用 folds\{f} 拟合一次预测f。
    # 返回开发OOF后验和最终外折后验。分组号必须只属于一个fold。
    cyc = cycle_frame(df); train_folds = sorted(set(cyc.fold) - {outer})
    inner = []
    for vf in train_folds:
        tr = cyc[(cyc.fold != outer) & (cyc.fold != vf)]
        inner.append(predict_distributions(fit_ordinal(tr, C), cyc[cyc.fold == vf]))
    tr = cyc[cyc.fold != outer]
    test = predict_distributions(fit_ordinal(tr, C), cyc[cyc.fold == outer])
    return pd.concat(inner, ignore_index=True), test


def select_policy(train_df: pd.DataFrame, candidates: dict[float, pd.DataFrame]):
    # 仅允许传入开发数据与crossfit_dist产生的开发OOF后验。
    # 遍历5×4×4×4×2个门控组合，每个C各640种，默认3个C共1920种。
    y = train_df.label.to_numpy(int); base = (train_df.p_classify.to_numpy(float) >= .5).astype(int)
    base_acc = accuracy_score(y, base); rows = []
    for C, dist in candidates.items():
        for pt in [0.20, 0.25, 0.30, 0.35, 0.40]:
            for mt in [0.02, 0.05, 0.10, 0.15]:
                for et in [0.65, 0.75, 0.85, 0.95]:
                    for rt in [0.0, 0.25, 0.5, 0.75]:
                        for mc in [1, 2]:
                            pred, _, acted = decode(train_df, dist, pt, mt, et, rt, mc)
                            acc = accuracy_score(y, pred); deltas = []
                            for f in sorted(train_df.fold.unique()):
                                m = train_df.fold.to_numpy() == f
                                deltas.append(float(np.mean(pred[m] == y[m]) - np.mean(base[m] == y[m])))
                            rows.append({"C": C, "post_thr": pt, "margin_thr": mt,
                                         "entropy_thr": et, "rank_gap_thr": rt,
                                         "max_changes": mc, "acc": float(acc),
                                         "delta": float(acc - base_acc), "action_frac": float(acted.mean()),
                                         "fold_deltas": deltas,
                                         "positive_folds": int(np.sum(np.asarray(deltas) > 0)),
                                         "nonnegative_folds": int(np.sum(np.asarray(deltas) >= 0))})
    # Require gain in >=2 inner folds, no harmed inner fold, and >=2 net corrections.
    # 需要开发集净增至少2例正确、至少2个内折改善、每个内折都不下降。
    valid = [r for r in rows if r["delta"] >= 2 / len(train_df)
             and r["positive_folds"] >= 2 and r["nonnegative_folds"] == len(r["fold_deltas"])]
    if not valid:
        # 不可达阈值+max_changes=0使decode完全退回表格。
        return {"fallback": True, "C": 0.2, "post_thr": 2.0, "margin_thr": 2.0,
                "entropy_thr": 0.0, "rank_gap_thr": 1e9, "max_changes": 0,
                "acc": float(base_acc), "delta": 0.0, "action_frac": 0.0,
                "fold_deltas": [0.0] * 4, "positive_folds": 0, "nonnegative_folds": 4}
    # 在Acc上减去动作比例惩罚，同分时选择更少动作，保留保守性。
    best = max(valid, key=lambda r: (r["acc"] - .004 * r["action_frac"], -r["action_frac"]))
    best["fallback"] = False
    return best


def metrics(y, pred, prob):
    # ACC/F1由离散pred计算，AUC由连续prob计算；固定矩阵顺序TN/FP/FN/TP。
    # 函数假设评价集合同时有正负两类，否则roc_auc_score可能不可定义。
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, prob)), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)}


def main():
    # 此main用于旧的“已有视频排序+数量模型”入口。
    # 最终67.42%版本由上一级evaluate_phasewise_vjepa_nested_topk.py调用这些函数。
    df = pd.read_csv(INPUT, dtype={"sample_id": str, "video_id": str, "group_id": str}).reset_index(drop=True)
    df.label = df.label.astype(int); df.fold = df.fold.astype(int)
    outs, folds, policies, all_dist = [], [], [], []
    Cs = [0.05, 0.2, 1.0]
    for outer in sorted(df.fold.unique()):
        train, test = df[df.fold != outer].copy(), df[df.fold == outer].copy()
        inner_by_c, test_by_c = {}, {}
        for C in Cs:
            inner_by_c[C], test_by_c[C] = crossfit_dist(df, int(outer), C)
        policy = select_policy(train, inner_by_c); policy["outer_fold"] = int(outer)
        td = test_by_c[policy["C"]]
        pred, prob, acted = decode(test, td, policy["post_thr"], policy["margin_thr"],
                                   policy["entropy_thr"], policy["rank_gap_thr"], policy["max_changes"])
        base = (test.p_classify.to_numpy(float) >= .5).astype(int)
        out = test.copy(); out["pred_base"] = base; out["pred_topk_v2"] = pred
        out["prob_topk_v2"] = prob; out["acted"] = acted
        out = out.merge(td, on=["group_id", "fold"], how="left")
        outs.append(out); all_dist.append(td); policies.append(policy)
        folds.append({"fold": int(outer), "n": len(out), "base_acc": float(np.mean(base == out.label)),
                      "topk_acc": float(np.mean(pred == out.label)),
                      "delta": float(np.mean(pred == out.label) - np.mean(base == out.label)),
                      "action_frac": float(acted.mean()), "k_exact": float(np.mean(td.k_mode == td.k_true)),
                      "k_within1": float(np.mean(abs(td.k_mode - td.k_true) <= 1))})
    oof = pd.concat(outs, ignore_index=True).sort_values(["fold", "group_id", "sample_id"])
    y = oof.label.to_numpy(int); pb = oof.pred_base.to_numpy(int); pn = oof.pred_topk_v2.to_numpy(int)
    summary = {"protocol": "strict nested ordinal posterior P(K) + conservative Top-K",
               "n_embryo": len(oof), "n_cycles": int(oof.group_id.nunique()),
               "baseline": metrics(y, pb, oof.p_classify),
               "discrete_topk": metrics(y, pn, oof.prob_topk_v2),
               "delta_accuracy": float(np.mean(pn == y) - np.mean(pb == y)),
               "n_changed": int(np.sum(pn != pb)), "n_acted": int(oof.acted.sum()),
               "k_exact": float(np.mean(pd.concat(all_dist).k_mode == pd.concat(all_dist).k_true)),
               "k_within1": float(np.mean(abs(pd.concat(all_dist).k_mode - pd.concat(all_dist).k_true) <= 1)),
               "bootstrap": paired_cycle_bootstrap(oof, pb, pn), "folds": folds, "policies": policies}
    HERE.mkdir(exist_ok=True); oof.to_csv(HERE / "oof_discrete_topk.csv", index=False)
    pd.concat(all_dist, ignore_index=True).to_csv(HERE / "oof_count_posteriors.csv", index=False)
    pd.DataFrame(folds).to_csv(HERE / "fold_metrics.csv", index=False)
    with open(HERE / "summary.json", "w", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
