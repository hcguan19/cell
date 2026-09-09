#!/usr/bin/env python3
# 原始表格清洗、标签映射、周期分组及外折生成的历史入口。
# 标签过滤和ID清洗需要与目标数据版本核对，不自动修复后续映射差异。
"""Prepare clinical table data and strict group 5-fold splits.

Source:
  /data0/guanhuancong/cell/merged_two_datasets_clinical_labels.xlsx

Label rules (最终分类_来自标签表):
  - positive (1): 二倍体, 整倍体
  - negative (0): all other non-missing values except 仅嵌合
  - drop rows where 最终分类_来自标签表 is missing
  - drop rows where 最终分类_来自标签表 == 仅嵌合
    (also drop 整倍体与否 == 仅嵌合 if present)

Also:
  - drop exact duplicate (周期号, video)
  - group by 周期号 for leakage-safe CV
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
SOURCE_XLSX = Path("/data0/guanhuancong/cell/merged_two_datasets_clinical_labels.xlsx")
SEED = 20260815

RAW_LABEL_COL = "最终分类_来自标签表"
POSITIVE_LABELS = {"二倍体", "整倍体"}
GROUP_COL = "周期号"
VIDEO_COL = "对应的video"
PGT_COL = "IVF/ICSI/PGT1号"

NUMERIC_FEATURES = [
    "女方建档年龄",
    "男方建档年龄",
    "BMI",
    "内分泌检查AMH",
    "内分泌检查-FSH",
    "内分泌检查LH",
    "内分泌检查E2",
    "内分泌P",
    "内分泌T",
    "内分泌检查PRL",
    "Gn启动剂量",
    "Gn用药总量",
    "Gn用药总天数",
    "Gn-AFC",
    "AFC",
    "HCG日E2",
    "HCG日LH",
    "HCG日P",
    "HCG日总卵泡数",
    "hcg日17卵泡",
    "【16 22】",
    "获卵数",
    "MII卵数",
    "成熟卵子数",
    "可利用胚胎数",
    "D3优胚数",
    "囊胚培养数",
    "囊胚形成数",
    "优质囊胚总数",
    "可利用囊胚数",
    "2PN数",
    "婚育史-孕",
    "婚育史-产",
    "婚育史-自然流产",
]

CATEGORICAL_FEATURES = [
    PGT_COL,
    "女方不孕类型(原发不孕继发不孕)",
    "用药方案类型",
]

NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def normalize_video(value) -> str:
    text = str(value).strip()
    if text.lower().endswith(".avi"):
        text = text[:-4]
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return f"{int(digits):06d}"
    return text


def parse_numeric(series: pd.Series) -> pd.Series:
    out = []
    for value in series.astype("string"):
        if value is pd.NA or value is None or str(value).strip() in {"", "nan", "None", "NA", "NaN"}:
            out.append(np.nan)
            continue
        text = str(value).strip().replace(",", "")
        match = NUMERIC_RE.search(text)
        out.append(float(match.group(0)) if match else np.nan)
    return pd.Series(out, index=series.index, dtype=float)


def parse_menses(series: pd.Series) -> pd.DataFrame:
    period, cycle, irregular = [], [], []
    for value in series.astype("string"):
        text = "" if value is pd.NA or value is None else str(value).strip()
        if not text or text.lower() in {"nan", "none", "na"}:
            period.append(np.nan)
            cycle.append(np.nan)
            irregular.append(np.nan)
            continue
        irregular.append(1.0 if any(tok in text for tok in ("不规律", "紊乱", "不规则")) else 0.0)
        nums = [float(x) for x in NUMERIC_RE.findall(text.replace("～", "-").replace("~", "-"))]
        if len(nums) >= 2:
            period.append(nums[0])
            cycle.append(nums[1])
        elif len(nums) == 1:
            period.append(nums[0])
            cycle.append(np.nan)
        else:
            period.append(np.nan)
            cycle.append(np.nan)
    return pd.DataFrame(
        {
            "月经_经期天数": period,
            "月经_周期天数": cycle,
            "月经_不规律": irregular,
        },
        index=series.index,
    )


def balanced_outer_folds(labels, groups, sources, seed, trials=64):
    indices = np.arange(len(labels))
    target_n = len(indices) / 5.0
    target_pos = float(labels.mean())
    source_values = sorted(set(sources.tolist()))
    target_source = {v: float(np.mean(sources == v)) for v in source_values}
    best = None
    for offset in range(trials):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + offset)
        folds = list(cv.split(indices, labels, groups))
        score = 0.0
        for _, test in folds:
            score += 4.0 * abs(len(test) - target_n) / len(indices)
            score += abs(float(labels[test].mean()) - target_pos)
            score += 0.25 * sum(
                abs(float(np.mean(sources[test] == v)) - target_source[v]) for v in source_values
            )
        cand = (score, folds, seed + offset)
        if best is None or score < best[0]:
            best = cand
    return best[1], {"seed": best[2], "search_trials": trials, "score": float(best[0])}


def inner_split(indices, labels, groups, sources, seed):
    global_pos = float(labels[indices].mean())
    source_values = sorted(set(sources[indices].tolist()))
    global_source = {v: float(np.mean(sources[indices] == v)) for v in source_values}
    best = None
    for offset in range(8):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + offset)
        for fold, (train_rel, val_rel) in enumerate(cv.split(indices, labels[indices], groups[indices])):
            train, val = indices[train_rel], indices[val_rel]
            score = abs(len(val) / len(indices) - 0.2)
            score += abs(float(labels[val].mean()) - global_pos)
            score += 0.25 * sum(
                abs(float(np.mean(sources[val] == v)) - global_source[v]) for v in source_values
            )
            cand = (score, train, val, seed + offset, fold)
            if best is None or score < best[0]:
                best = cand
    return best[1], best[2], {"seed": int(best[3]), "inner_fold": int(best[4]), "score": float(best[0])}


def majority_oracle(labels: np.ndarray, groups: np.ndarray) -> dict:
    frame = pd.DataFrame({"label": labels, "group_id": groups})
    maj = frame.groupby("group_id")["label"].transform(lambda s: int(s.mean() >= 0.5))
    conflict = frame.groupby("group_id")["label"].transform(lambda s: int(s.nunique() > 1))
    return {
        "n_samples": int(len(frame)),
        "n_groups": int(frame["group_id"].nunique()),
        "conflict_groups": int(frame.loc[conflict == 1, "group_id"].nunique()),
        "conflict_samples": int(conflict.sum()),
        "majority_oracle_acc": float((frame["label"] == maj).mean()),
    }


def main():
    raw = pd.read_excel(SOURCE_XLSX, dtype={VIDEO_COL: str, GROUP_COL: str})
    summary = {
        "source_xlsx": str(SOURCE_XLSX),
        "raw_rows": int(len(raw)),
        "raw_label_col": RAW_LABEL_COL,
        "positive_labels": sorted(POSITIVE_LABELS),
        "raw_label_value_counts": {
            str(k): int(v) for k, v in raw[RAW_LABEL_COL].value_counts(dropna=False).items()
        },
    }

    missing_mask = raw[RAW_LABEL_COL].isna()
    # Use fillna(False): pandas StringDtype .eq() yields <NA> on missing cells,
    # and those would otherwise be dropped by boolean indexing.
    mosaic_mask = raw[RAW_LABEL_COL].astype("string").str.strip().eq("仅嵌合").fillna(False)
    if "整倍体与否" in raw.columns:
        mosaic_mask = mosaic_mask | raw["整倍体与否"].astype("string").str.strip().eq("仅嵌合").fillna(False)
    mosaic_mask = mosaic_mask.astype(bool)
    summary["dropped_missing_final_class"] = int(missing_mask.sum())
    summary["dropped_only_mosaic"] = int((~missing_mask & mosaic_mask).sum())
    df = raw.loc[~missing_mask & ~mosaic_mask].copy()
    summary["after_drop_missing_and_mosaic_rows"] = int(len(df))

    before_dedup = len(df)
    df = df.drop_duplicates([GROUP_COL, VIDEO_COL], keep="first").copy()
    summary["dropped_exact_cycle_video_dups"] = int(before_dedup - len(df))

    df["sample_id"] = (
        df[GROUP_COL].astype(str).str.strip()
        + "::"
        + df[VIDEO_COL].map(normalize_video)
    )
    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id still duplicated after cycle::video construction")

    final_class = df[RAW_LABEL_COL].astype("string").str.strip()
    df["final_class"] = final_class
    df["label"] = final_class.isin(POSITIVE_LABELS).astype(int)
    summary["label_from_final_class_counts"] = {
        str(k): int(v) for k, v in final_class.value_counts(dropna=False).items()
    }
    df["group_id"] = df[GROUP_COL].astype(str).str.strip()
    df["source"] = df["数据集"].astype(str)
    df["pgt_type"] = df[PGT_COL].astype("string").str.strip().fillna("UNKNOWN")
    df["video_id"] = df[VIDEO_COL].map(normalize_video)

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = parse_numeric(df[col])
        else:
            df[col] = np.nan

    menses = parse_menses(df["月经史-月经周期"] if "月经史-月经周期" in df.columns else pd.Series(index=df.index, dtype="string"))
    df = pd.concat([df, menses], axis=1)

    feature_numeric = [c for c in NUMERIC_FEATURES if c in df.columns] + [
        "月经_经期天数",
        "月经_周期天数",
        "月经_不规律",
    ]
    feature_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    keep_cols = [
        "sample_id",
        "video_id",
        "group_id",
        "label",
        "final_class",
        "source",
        "pgt_type",
        VIDEO_COL,
        GROUP_COL,
        "数据集",
        "整倍体与否",
        "最终分类_来自标签表",
        *feature_numeric,
        *feature_categorical,
    ]
    clean = df[keep_cols].reset_index(drop=True)
    oracle = majority_oracle(clean["label"].to_numpy(), clean["group_id"].to_numpy())
    summary["majority_oracle"] = oracle
    summary["label_counts"] = clean["label"].value_counts().sort_index().to_dict()
    summary["source_counts"] = clean["source"].value_counts().to_dict()
    summary["pgt_counts"] = clean["pgt_type"].value_counts().to_dict()
    summary["feature_numeric"] = feature_numeric
    summary["feature_categorical"] = feature_categorical

    out_data = HERE / "data"
    out_cv = HERE / "group_cv5"
    out_data.mkdir(parents=True, exist_ok=True)
    out_cv.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_data / "clean_table.csv", index=False)

    labels = clean["label"].to_numpy()
    groups = clean["group_id"].to_numpy()
    sources = clean["source"].to_numpy()
    outer_folds, outer_meta = balanced_outer_folds(labels, groups, sources, SEED)
    manifest = {
        "seed": SEED,
        "n_folds": 5,
        "outer": outer_meta,
        "n_samples": int(len(clean)),
        "n_groups": int(clean["group_id"].nunique()),
        "folds": [],
    }

    oof_seen = np.zeros(len(clean), dtype=int)
    for fold, (development, test) in enumerate(outer_folds):
        train, val, inner_meta = inner_split(
            development, labels, groups, sources, SEED + 1000 + fold * 100
        )
        oof_seen[test] += 1
        fold_dir = out_cv / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        assignments = []
        for split_name, idx in (("train", train), ("val", val), ("test", test)):
            part = clean.iloc[idx].copy()
            part.to_csv(fold_dir / f"{split_name}.csv", index=False)
            for _, row in part.iterrows():
                assignments.append(
                    {
                        "sample_id": row["sample_id"],
                        "group_id": row["group_id"],
                        "label": int(row["label"]),
                        "split": split_name,
                        "fold": fold,
                    }
                )
        pd.DataFrame(assignments).to_csv(fold_dir / "assignments.csv", index=False)
        manifest["folds"].append(
            {
                "fold": fold,
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "n_test": int(len(test)),
                "pos_rate_test": float(labels[test].mean()),
                "inner": inner_meta,
            }
        )

    if not np.array_equal(oof_seen, np.ones(len(clean), dtype=int)):
        raise ValueError("OOF coverage is not exactly once per sample")

    (HERE / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_cv / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "manifest_folds": manifest["folds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
