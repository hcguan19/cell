#!/usr/bin/env python3
# 早期表格多候选继续实验，保留验证选参与阈值探索的来源。
# 历史名称V2不等于当前完整融合流水线版本。
"""Continue pure-table Acc search on no-mosaic final_class labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
CV_ROOT = HERE / "group_cv5"
DOCTOR = [
    "女方建档年龄", "男方建档年龄", "内分泌检查AMH", "BMI",
    "内分泌检查-FSH", "内分泌检查E2", "Gn-AFC", "Gn启动剂量",
]
OLD20 = DOCTOR + [
    "内分泌T", "HCG日LH", "HCG日P", "HCG日E2", "HCG日总卵泡数", "hcg日17卵泡",
    "【16 22】", "MII卵数", "可利用胚胎数", "囊胚培养数", "优质囊胚总数",
    "可利用囊胚数", "月经_经期天数", "月经_周期天数", "月经_不规律",
]
CONFIGS = [
    {"max_depth": 1, "min_child_weight": 3, "reg_alpha": 0.2, "reg_lambda": 4.0},
    {"max_depth": 2, "min_child_weight": 6, "reg_alpha": 0.5, "reg_lambda": 8.0},
    {"max_depth": 3, "min_child_weight": 6, "reg_alpha": 0.8, "reg_lambda": 10.0},
]
SHRINKAGE = [0.0, 2.0, 5.0]
MAJOR = ("PGT-A", "PGT-M", "PGT-SR")


def metric(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def soft_ll(t, p, w):
    t = np.asarray(t, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    w = np.asarray(w, float)
    return float(np.average(-(t * np.log(p) + (1 - t) * np.log(1 - p)), weights=w))


def gside(tt, p):
    true_side = (tt.positive / tt.n >= 0.5).astype(int).to_numpy()
    pred_side = (np.asarray(p) >= 0.5).astype(int)
    return float((true_side == pred_side).mean())


def model(cfg, seed):
    return xgb.XGBRegressor(
        objective="reg:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.05,
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        subsample=0.9,
        colsample_bytree=0.85,
        **cfg,
    )


def encode_pgt(train, frames):
    vals = train["pgt_type"].astype("string").str.strip().fillna("UNKNOWN")
    vocab = sorted(vals.value_counts().loc[lambda s: s >= 5].index.tolist())
    cols = []
    for i, v in enumerate(vocab):
        name = f"pgt__{i:02d}"
        cols.append(name)
        for fr in frames.values():
            fr[name] = fr["pgt_type"].astype("string").str.strip().fillna("UNKNOWN").eq(v).astype(float)
    return cols


def build(frame, cols):
    g = frame.groupby("group_id", sort=False)
    X = g[cols].first().copy()
    T = g["label"].agg(["sum", "size"]).rename(columns={"sum": "positive", "size": "n"})
    X["cohort_size"] = T["n"].astype(float)
    X["log1p_cohort_size"] = np.log1p(T["n"].astype(float))
    rows = frame[["sample_id", "label", "group_id", "pgt_type"]].copy()
    return X, T, rows


def fit_variant(tx, tt, vx, vt, fold, tag, mode):
    trials = []
    prior = float(tt.positive.sum() / max(tt.n.sum(), 1))
    for strength in SHRINKAGE:
        if mode == "majority":
            y = (tt.positive / tt.n >= 0.5).astype(float)
            clarity = (2.0 * (tt.positive / tt.n) - 1.0).abs().clip(0.05, 1.0)
            w = (tt.n.astype(float) * clarity).to_numpy(float)
            strengths = [0.0]
            if strength != 0.0:
                continue
        elif mode == "balanced_soft":
            rate = tt.positive / tt.n
            y = (tt.positive + strength * prior) / (tt.n + strength)
            # upweight minority-side groups
            side = (rate >= 0.5).astype(float)
            pos_frac = float(side.mean()) if len(side) else 0.5
            bal = np.where(side.to_numpy() >= 0.5, 1.0 / max(pos_frac, 1e-3), 1.0 / max(1 - pos_frac, 1e-3))
            w = ((tt.n + strength).to_numpy(float) * bal)
            strengths = None
        else:
            y = (tt.positive + strength * prior) / (tt.n + strength)
            w = (tt.n + strength).to_numpy(float)
            strengths = None

        for ci, cfg in enumerate(CONFIGS):
            seed = 71260815 + fold * 1000 + (sum(map(ord, tag + mode)) % 97) * 10 + ci + int(strength)
            m = model(cfg, seed)
            m.fit(tx.astype(np.float32), np.asarray(y, float), sample_weight=w)
            p = m.predict(vx.astype(np.float32)).clip(1e-5, 1 - 1e-5)
            if mode == "majority":
                yv = (vt.positive / vt.n >= 0.5).astype(int).to_numpy()
                vll = float(log_loss(yv, p, labels=[0, 1]))
            else:
                vll = soft_ll((vt.positive / vt.n).to_numpy(float), p, vt.n.to_numpy(float))
            trials.append(
                {
                    "config": cfg,
                    "strength": float(strength),
                    "seed": seed,
                    "val_group_side_acc": gside(vt, p),
                    "val_logloss": vll,
                    "val_probability": p,
                }
            )
    return min(trials, key=lambda q: (-q["val_group_side_acc"], q["val_logloss"]))


def refit_variant(dx, dt, qx, ch, mode):
    prior = float(dt.positive.sum() / max(dt.n.sum(), 1))
    strength = ch["strength"]
    if mode == "majority":
        y = (dt.positive / dt.n >= 0.5).astype(float)
        clarity = (2.0 * (dt.positive / dt.n) - 1.0).abs().clip(0.05, 1.0)
        w = (dt.n.astype(float) * clarity).to_numpy(float)
    elif mode == "balanced_soft":
        rate = dt.positive / dt.n
        y = (dt.positive + strength * prior) / (dt.n + strength)
        side = (rate >= 0.5).astype(float)
        pos_frac = float(side.mean()) if len(side) else 0.5
        bal = np.where(side.to_numpy() >= 0.5, 1.0 / max(pos_frac, 1e-3), 1.0 / max(1 - pos_frac, 1e-3))
        w = ((dt.n + strength).to_numpy(float) * bal)
    else:
        y = (dt.positive + strength * prior) / (dt.n + strength)
        w = (dt.n + strength).to_numpy(float)
    m = model(ch["config"], ch["seed"])
    m.fit(dx.astype(np.float32), np.asarray(y, float), sample_weight=w)
    return m.predict(qx.astype(np.float32)).clip(1e-5, 1 - 1e-5)


def run_one(name, base_cols, mode="soft", hier=True):
    parts = []
    for fold in range(5):
        frames = {sp: pd.read_csv(CV_ROOT / f"fold_{fold}/{sp}.csv") for sp in ("train", "val", "test")}
        pgt = encode_pgt(frames["train"], frames)
        if base_cols is None:
            skip = {
                "sample_id", "video_id", "group_id", "label", "source", "pgt_type",
                "对应的video", "周期号", "数据集", "整倍体与否", "最终分类_来自标签表",
                "final_class", "IVF/ICSI/PGT1号", "女方不孕类型(原发不孕继发不孕)",
                "用药方案类型", *pgt,
            }
            cols = [
                c for c in frames["train"].columns
                if c not in skip and pd.api.types.is_numeric_dtype(frames["train"][c])
            ]
        else:
            cols = [c for c in base_cols if c in frames["train"].columns]
        cols = [
            c for c in cols + pgt
            if frames["train"][c].notna().any() and frames["train"][c].nunique(dropna=True) > 1
        ]
        grouped = {s: build(frames[s], cols) for s in frames}
        tx, tt, _ = grouped["train"]
        vx, vt, _ = grouped["val"]
        qx, qt, test_rows = grouped["test"]
        names = list(tx.columns)
        pmap = {s: grouped[s][2].groupby("group_id")["pgt_type"].first() for s in grouped}

        valp = pd.Series(index=vt.index, dtype=float)
        tesp = pd.Series(index=qt.index, dtype=float)
        ch = fit_variant(tx[names], tt, vx[names], vt, fold, "global", mode)
        valp.loc[vt.index] = ch.pop("val_probability")
        tesp.loc[qt.index] = refit_variant(
            pd.concat([tx[names], vx[names]]), pd.concat([tt, vt]), qx[names], ch, mode
        )
        gval, gtest = valp.copy(), tesp.copy()

        if hier:
            for tname in MAJOR:
                tr = pmap["train"].index[pmap["train"].eq(tname)]
                va = pmap["val"].index[pmap["val"].eq(tname)]
                te = pmap["test"].index[pmap["test"].eq(tname)]
                if min(len(tr), len(va), len(te)) < 4:
                    continue
                ch = fit_variant(
                    tx.loc[tr, names], tt.loc[tr], vx.loc[va, names], vt.loc[va],
                    fold, f"t_{tname}", mode,
                )
                tv = ch.pop("val_probability")
                yv = (vt.loc[va, "positive"] / vt.loc[va, "n"] >= 0.5).astype(int).to_numpy()
                gv = gval.loc[va].to_numpy(float)
                blends = []
                for a in np.arange(0.0, 1.01, 0.1):
                    p = (1 - a) * gv + a * tv
                    blends.append(
                        {
                            "alpha": float(a),
                            "group_side_acc": float(accuracy_score(yv, p >= 0.5)),
                            "logloss": float(log_loss(yv, np.clip(p, 1e-5, 1 - 1e-5), labels=[0, 1])),
                        }
                    )
                b = max(blends, key=lambda q: (q["group_side_acc"], -q["logloss"]))
                valp.loc[va] = (1 - b["alpha"]) * gv + b["alpha"] * tv
                ttst = refit_variant(
                    pd.concat([tx.loc[tr, names], vx.loc[va, names]]),
                    pd.concat([tt.loc[tr], vt.loc[va]]),
                    qx.loc[te, names],
                    ch,
                    mode,
                )
                tesp.loc[te] = (1 - b["alpha"]) * gtest.loc[te].to_numpy(float) + b["alpha"] * ttst

        out = test_rows.copy()
        out["probability"] = test_rows["group_id"].map(tesp)
        out["fold"] = fold
        parts.append(out)
        print(f"{name} fold={fold} acc={accuracy_score(out.label, out.probability >= 0.5):.4f}", flush=True)

    oof = pd.concat(parts, ignore_index=True)
    oof.to_csv(HERE / f"oof_{name}.csv", index=False)
    score = metric(oof["label"], oof["probability"])
    g = oof.groupby("group_id").agg(y=("label", "mean"), p=("probability", "first"))
    score["group_side_acc"] = float(((g.y >= 0.5).astype(int) == (g.p >= 0.5).astype(int)).mean())
    maj = oof.groupby("group_id")["label"].transform(lambda s: int(s.mean() >= 0.5))
    score["majority_oracle_acc"] = float((oof["label"] == maj).mean())
    score["gap_to_majority_oracle"] = score["majority_oracle_acc"] - score["accuracy"]
    score["method"] = name
    print(f"[OOF] {name}: Acc={score['accuracy']:.4f} AUC={score['roc_auc']:.4f} "
          f"group_side={score['group_side_acc']:.4f}", flush=True)
    return score, oof


def nested_threshold(oof: pd.DataFrame, name: str):
    parts = []
    for fold in range(5):
        dev = oof[oof.fold != fold]
        te = oof[oof.fold == fold].copy()
        best_t, best_a = 0.5, -1.0
        for t in np.linspace(0.30, 0.70, 81):
            a = accuracy_score(dev.label, dev.probability >= t)
            if a > best_a:
                best_a, best_t = a, float(t)
        te["probability_thr"] = te["probability"]
        te["used_threshold"] = best_t
        te["prediction"] = (te["probability"] >= best_t).astype(int)
        parts.append(te)
    adj = pd.concat(parts, ignore_index=True)
    # encode thresholded decision as shifted probability for metric() compatibility
    p = np.where(adj["prediction"] == 1, np.maximum(adj["probability"], 0.5001), np.minimum(adj["probability"], 0.4999))
    score = metric(adj["label"], p)
    # keep true AUC from raw probs
    score["roc_auc"] = float(roc_auc_score(adj["label"], adj["probability"]))
    g = adj.groupby("group_id").agg(y=("label", "mean"), pred=("prediction", "first"))
    score["group_side_acc"] = float(((g.y >= 0.5).astype(int) == g.pred).mean())
    maj = adj.groupby("group_id")["label"].transform(lambda s: int(s.mean() >= 0.5))
    score["majority_oracle_acc"] = float((adj["label"] == maj).mean())
    score["gap_to_majority_oracle"] = score["majority_oracle_acc"] - score["accuracy"]
    score["method"] = name
    adj.to_csv(HERE / f"oof_{name}.csv", index=False)
    print(f"[OOF] {name}: Acc={score['accuracy']:.4f} AUC={score['roc_auc']:.4f}", flush=True)
    return score


def main():
    variants = [
        ("doctor_soft", DOCTOR, "soft"),
        ("doctor_balanced", DOCTOR, "balanced_soft"),
        ("doctor_majority", DOCTOR, "majority"),
        ("old20_soft", OLD20, "soft"),
        ("full_soft", None, "soft"),
    ]
    rows = []
    oofs = {}
    for name, cols, mode in variants:
        score, oof = run_one(name, cols, mode=mode, hier=True)
        rows.append(score)
        oofs[name] = oof

    # nested Acc threshold on best soft doctor
    rows.append(nested_threshold(oofs["doctor_soft"], "doctor_soft_nested_thr"))

    # strict OOF logreg gate over complementary views
    merged = oofs["doctor_soft"][["sample_id", "label", "group_id", "fold", "probability"]].rename(
        columns={"probability": "doctor_soft"}
    )
    for name in ("doctor_majority", "old20_soft", "full_soft"):
        merged = merged.merge(
            oofs[name][["sample_id", "probability"]].rename(columns={"probability": name}),
            on="sample_id",
            how="inner",
        )
    gate_parts = []
    xcols = ["doctor_soft", "doctor_majority", "old20_soft", "full_soft"]
    for fold in range(5):
        tr = merged[merged.fold != fold]
        te = merged[merged.fold == fold].copy()
        clf = LogisticRegression(max_iter=2000)
        clf.fit(tr[xcols].to_numpy(float), tr["label"].to_numpy(int))
        te["probability"] = clf.predict_proba(te[xcols].to_numpy(float))[:, 1]
        gate_parts.append(te)
    gate = pd.concat(gate_parts, ignore_index=True)
    gate.to_csv(HERE / "oof_view_logreg_gate.csv", index=False)
    gscore = metric(gate["label"], gate["probability"])
    g = gate.groupby("group_id").agg(y=("label", "mean"), p=("probability", "first"))
    gscore["group_side_acc"] = float(((g.y >= 0.5).astype(int) == (g.p >= 0.5).astype(int)).mean())
    maj = gate.groupby("group_id")["label"].transform(lambda s: int(s.mean() >= 0.5))
    gscore["majority_oracle_acc"] = float((gate["label"] == maj).mean())
    gscore["gap_to_majority_oracle"] = gscore["majority_oracle_acc"] - gscore["accuracy"]
    gscore["method"] = "view_logreg_gate"
    rows.append(gscore)
    print(f"[OOF] view_logreg_gate: Acc={gscore['accuracy']:.4f} AUC={gscore['roc_auc']:.4f}", flush=True)

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    summary.to_csv(HERE / "summary_views.csv", index=False)
    summary.to_csv(HERE / "summary_all.csv", index=False)
    best = summary.iloc[0].to_dict()
    # keep recommended alias
    best_name = best["method"]
    src = HERE / f"oof_{best_name}.csv"
    if src.exists():
        out = pd.read_csv(src)
        if "probability" not in out.columns and "probability_thr" in out.columns:
            out["probability"] = out["probability_thr"]
        out.to_csv(HERE / "oof_view_doctor_core.csv", index=False)
    (HERE / "summary_views.json").write_text(
        json.dumps({"results": rows, "best": best}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary[["method", "n", "accuracy", "roc_auc", "group_side_acc", "majority_oracle_acc", "gap_to_majority_oracle"]].to_string(index=False))
    print("BEST", best)


if __name__ == "__main__":
    main()
