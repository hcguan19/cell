#!/usr/bin/env python3
# 历史已有排序分数的堆叠入口，融合current/X3D-stage/VideoMAE等来源。
# current含既有融合成分，因此输出不是独立纯视频分类器。
"""Strict OOF stacking for X3D raw + VideoMAE raw video features.

This script does not retrain video backbones.  It consumes existing out-of-fold
probabilities and trains a second-level model only on other folds before
predicting each held-out fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data0/guanhuancong/cell/cell_train3")
OUT = ROOT / "dual_video_stacking_v1"

PATHS = {
    "hier": ROOT / "pgt_hierarchical_table_v1/oof_hierarchical_video.csv",
    "table": ROOT / "table_primary_v1/oof_predictions.csv",
    "x3d_raw_full5": ROOT / "pgt_type_conditional_fusion_v1/oof_type_video_raw_full5.csv",
    "x3d_stage_f16": ROOT / "raw_video_stage_contrast_v2/oof_f16_multiseed_fusion.csv",
    "videomae_raw4": ROOT / "videomae_4segment_raw_v1/oof_videomae_raw4.csv",
}


def norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)


def read_prob(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str})
    df["sample_id"] = norm_id(df["sample_id"])
    keep = ["sample_id"] + list(columns)
    out = df[keep].rename(columns=columns).copy()
    if out["sample_id"].duplicated().any():
        dup = out.loc[out["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"{path} has duplicate sample_id, examples={dup}")
    return out


def load_frame() -> pd.DataFrame:
    hier = read_prob(
        PATHS["hier"],
        {
            "label": "label",
            "group_id": "group_id",
            "fold": "fold",
            "global": "p_table_global",
            "type_probability": "p_table_type",
            "old_video": "p_old_video",
            "hierarchical_video": "p_current_best",
        },
    )
    table = read_prob(PATHS["table"], {"final": "p_table_primary"})
    x3d_raw = read_prob(
        PATHS["x3d_raw_full5"],
        {
            "candidate": "p_x3d_raw_full5",
            "fusion": "p_current_plus_x3d_raw_full5",
        },
    )
    x3d_stage = read_prob(
        PATHS["x3d_stage_f16"],
        {
            "candidate": "p_x3d_stage_f16_candidate",
            "new_video_ensemble": "p_x3d_stage_f16_ensemble",
            "fusion": "p_current_plus_x3d_stage_f16",
        },
    )
    vmae = read_prob(
        PATHS["videomae_raw4"],
        {
            "candidate": "p_videomae_raw4",
            "baseline_plus_candidate": "p_current_plus_videomae_raw4",
        },
    )
    df = hier
    for other in (table, x3d_raw, x3d_stage, vmae):
        df = df.merge(other, on="sample_id", how="inner", validate="one_to_one")
    if len(df) != 1410:
        raise ValueError(f"expected 1410 merged rows, got {len(df)}")
    if df["label"].isna().any():
        raise ValueError("missing labels")
    return df.sort_values(["fold", "sample_id"]).reset_index(drop=True)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def fold_metrics(df: pd.DataFrame, prob_col: str) -> list[dict]:
    out = []
    for fold, part in df.groupby("fold"):
        m = metrics(part["label"].to_numpy(), part[prob_col].to_numpy())
        m["fold"] = int(fold)
        out.append(m)
    return out


def fit_oof_stack(df: pd.DataFrame, feature_cols: list[str], name: str) -> tuple[pd.DataFrame, dict]:
    y = df["label"].to_numpy().astype(int)
    oof = np.zeros(len(df), dtype=float)
    fold_rows = []
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

    for fold in sorted(df["fold"].unique()):
        test_idx = df.index[df["fold"] == fold].to_numpy()
        train_idx = df.index[df["fold"] != fold].to_numpy()

        # Inner fold selection on the training folds only.
        best = None
        for c in c_grid:
            inner_pred = np.zeros(len(train_idx), dtype=float)
            train_folds = df.loc[train_idx, "fold"].to_numpy()
            for inner_fold in sorted(np.unique(train_folds)):
                inner_val_mask = train_folds == inner_fold
                inner_val_idx = train_idx[inner_val_mask]
                inner_tr_idx = train_idx[~inner_val_mask]
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "lr",
                            LogisticRegression(
                                C=c,
                                penalty="l2",
                                solver="lbfgs",
                                class_weight="balanced",
                                max_iter=2000,
                                random_state=20260814,
                            ),
                        ),
                    ]
                )
                model.fit(df.loc[inner_tr_idx, feature_cols], y[inner_tr_idx])
                inner_pred[inner_val_mask] = model.predict_proba(df.loc[inner_val_idx, feature_cols])[:, 1]
            acc = accuracy_score(y[train_idx], inner_pred >= 0.5)
            auc = roc_auc_score(y[train_idx], inner_pred)
            score = (acc, auc, -c)
            cand = {"c": c, "inner_accuracy": float(acc), "inner_auc": float(auc)}
            if best is None or score > best["score"]:
                best = {"score": score, **cand}

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=best["c"],
                        penalty="l2",
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=20260814,
                    ),
                ),
            ]
        )
        model.fit(df.loc[train_idx, feature_cols], y[train_idx])
        oof[test_idx] = model.predict_proba(df.loc[test_idx, feature_cols])[:, 1]
        coef = model.named_steps["lr"].coef_[0].tolist()
        fold_rows.append(
            {
                "fold": int(fold),
                "chosen_c": float(best["c"]),
                "inner_accuracy": best["inner_accuracy"],
                "inner_auc": best["inner_auc"],
                "coefficients": dict(zip(feature_cols, [float(v) for v in coef])),
                **metrics(y[test_idx], oof[test_idx]),
            }
        )

    out = df[["sample_id", "label", "group_id", "fold"]].copy()
    out[name] = oof
    summary = {
        "name": name,
        "features": feature_cols,
        "overall": metrics(y, oof),
        "folds": fold_rows,
    }
    return out, summary


def grid_blend_oof(df: pd.DataFrame, cols: list[str], name: str, step: float = 0.05) -> tuple[pd.DataFrame, dict]:
    y = df["label"].to_numpy().astype(int)
    oof = np.zeros(len(df), dtype=float)
    fold_rows = []
    weights = np.arange(0, 1 + 1e-9, step)

    def enumerate_weights(k: int):
        if k == 2:
            for w0 in weights:
                yield np.array([w0, 1 - w0])
        elif k == 3:
            for w0 in weights:
                for w1 in weights:
                    w2 = 1 - w0 - w1
                    if w2 >= -1e-9:
                        yield np.array([w0, w1, max(0.0, w2)])
        elif k == 4:
            # Keep this conservative; full 0.05 simplex is still small enough.
            for w0 in weights:
                for w1 in weights:
                    for w2 in weights:
                        w3 = 1 - w0 - w1 - w2
                        if w3 >= -1e-9:
                            yield np.array([w0, w1, w2, max(0.0, w3)])
        else:
            raise ValueError(k)

    for fold in sorted(df["fold"].unique()):
        test_idx = df.index[df["fold"] == fold].to_numpy()
        train_idx = df.index[df["fold"] != fold].to_numpy()
        x_train = df.loc[train_idx, cols].to_numpy()
        x_test = df.loc[test_idx, cols].to_numpy()
        best = None
        for w in enumerate_weights(len(cols)):
            p = x_train @ w
            acc = accuracy_score(y[train_idx], p >= 0.5)
            auc = roc_auc_score(y[train_idx], p)
            score = (acc, auc, -float(np.count_nonzero(w)))
            if best is None or score > best["score"]:
                best = {"score": score, "weights": w, "inner_accuracy": float(acc), "inner_auc": float(auc)}
        oof[test_idx] = x_test @ best["weights"]
        fold_rows.append(
            {
                "fold": int(fold),
                "weights": dict(zip(cols, [float(v) for v in best["weights"]])),
                "inner_accuracy": best["inner_accuracy"],
                "inner_auc": best["inner_auc"],
                **metrics(y[test_idx], oof[test_idx]),
            }
        )

    out = df[["sample_id", "label", "group_id", "fold"]].copy()
    out[name] = oof
    summary = {
        "name": name,
        "features": cols,
        "overall": metrics(y, oof),
        "folds": fold_rows,
    }
    return out, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    df.to_csv(OUT / "merged_oof_inputs.csv", index=False)

    baseline_cols = [
        "p_table_primary",
        "p_current_best",
        "p_x3d_raw_full5",
        "p_x3d_stage_f16_ensemble",
        "p_videomae_raw4",
    ]
    baseline = {}
    for col in baseline_cols:
        baseline[col] = {"overall": metrics(df["label"].to_numpy(), df[col].to_numpy()), "folds": fold_metrics(df, col)}

    experiments = {
        "stack_current_x3d_vmae_lr": ["p_current_best", "p_x3d_raw_full5", "p_videomae_raw4"],
        "stack_table_x3d_vmae_lr": ["p_table_primary", "p_x3d_raw_full5", "p_videomae_raw4"],
        "stack_current_x3dstage_vmae_lr": ["p_current_best", "p_x3d_stage_f16_ensemble", "p_videomae_raw4"],
        "stack_current_two_x3d_vmae_lr": [
            "p_current_best",
            "p_x3d_raw_full5",
            "p_x3d_stage_f16_ensemble",
            "p_videomae_raw4",
        ],
    }
    blend_experiments = {
        "blend_current_x3d_vmae_grid": ["p_current_best", "p_x3d_raw_full5", "p_videomae_raw4"],
        "blend_current_x3dstage_vmae_grid": ["p_current_best", "p_x3d_stage_f16_ensemble", "p_videomae_raw4"],
        "blend_current_two_x3d_vmae_grid": [
            "p_current_best",
            "p_x3d_raw_full5",
            "p_x3d_stage_f16_ensemble",
            "p_videomae_raw4",
        ],
    }

    summaries = {"inputs": {k: str(v) for k, v in PATHS.items()}, "baselines": baseline, "experiments": {}}
    all_pred = df[["sample_id", "label", "group_id", "fold"]].copy()

    for name, cols in experiments.items():
        pred, summary = fit_oof_stack(df, cols, name)
        pred.to_csv(OUT / f"oof_{name}.csv", index=False)
        all_pred = all_pred.merge(pred[["sample_id", name]], on="sample_id", how="left")
        summaries["experiments"][name] = summary

    for name, cols in blend_experiments.items():
        pred, summary = grid_blend_oof(df, cols, name)
        pred.to_csv(OUT / f"oof_{name}.csv", index=False)
        all_pred = all_pred.merge(pred[["sample_id", name]], on="sample_id", how="left")
        summaries["experiments"][name] = summary

    all_pred.to_csv(OUT / "oof_all_dual_video_stacking.csv", index=False)

    rows = []
    for name, item in baseline.items():
        rows.append({"name": name, "kind": "input", **item["overall"]})
    for name, item in summaries["experiments"].items():
        rows.append({"name": name, "kind": "stacking", **item["overall"]})
    pd.DataFrame(rows).sort_values(["accuracy", "roc_auc"], ascending=False).to_csv(
        OUT / "summary_table.csv", index=False
    )

    (OUT / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(OUT), "summary_table": str(OUT / "summary_table.csv")}, ensure_ascii=False))
    print(pd.read_csv(OUT / "summary_table.csv").to_string(index=False))


if __name__ == "__main__":
    main()
