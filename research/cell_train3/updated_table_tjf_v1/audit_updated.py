#!/usr/bin/env python3
# 核对补充字段、缺失变化与样本ID匹配的私有数据审计。
# 运行输出可能包含具体样本与字段值，务必留在私有目录。
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data0/guanhuancong/cell")
HERE = ROOT / "cell_train3/updated_table_tjf_v1"
NEW = ROOT / "细胞表格数据 tjf.xlsx"
OLD = ROOT / "merged_two_datasets_clinical_labels.xlsx"
LABEL = "最终分类_来自标签表"
GROUP = "周期号"
VIDEO = "对应的video"


def canonical(value) -> str:
    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".avi", ".AVI", ".mp4", ".MP4"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return str(int(text)) if text.isdigit() else text


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not {LABEL, GROUP, VIDEO}.issubset(frame.columns):
        raise ValueError(f"required columns absent: {sorted({LABEL, GROUP, VIDEO} - set(frame.columns))}")
    labels = frame[LABEL].astype("string").str.strip()
    missing = labels.isna()
    mosaic = labels.eq("仅嵌合").fillna(False)
    keep = ~(missing | mosaic)
    result = frame.loc[keep].copy()
    result["label"] = labels.loc[keep].isin(["二倍体", "整倍体"]).astype(int)
    result["group_key"] = result[GROUP].astype("string").str.strip()
    result["video_key"] = result[VIDEO].map(canonical)
    result["sample_key"] = result.group_key.astype(str) + "::" + result.video_key
    before = len(result)
    result = result.drop_duplicates(["group_key", "video_key"], keep="first").copy()
    audit = {
        "raw_rows": int(len(frame)),
        "missing_label": int(missing.sum()),
        "mosaic_only": int(mosaic.sum()),
        "deduplicated": int(before - len(result)),
        "clean_rows": int(len(result)),
        "cycles": int(result.group_key.nunique()),
        "label_0": int(result.label.eq(0).sum()),
        "label_1": int(result.label.eq(1).sum()),
    }
    return result, audit


def meaningful_missing(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return series.isna() | text.eq("").fillna(False) | text.str.lower().isin(["nan", "none", "null"])


def changed_values(old: pd.Series, new: pd.Series) -> int:
    both = ~meaningful_missing(old) & ~meaningful_missing(new)
    if not both.any():
        return 0
    old_num = pd.to_numeric(old[both], errors="coerce")
    new_num = pd.to_numeric(new[both], errors="coerce")
    numeric = old_num.notna() & new_num.notna()
    changed = pd.Series(False, index=old_num.index)
    if numeric.any():
        changed.loc[numeric] = ~np.isclose(old_num[numeric], new_num[numeric], rtol=1e-7, atol=1e-9)
    nonnumeric = ~numeric
    if nonnumeric.any():
        changed.loc[nonnumeric] = (
            old[both].loc[nonnumeric].astype("string").str.strip()
            != new[both].loc[nonnumeric].astype("string").str.strip()
        )
    return int(changed.sum())


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    new_book = pd.ExcelFile(NEW)
    sheet_audits = []
    candidate = None
    candidate_sheet = None
    for sheet in new_book.sheet_names:
        frame = pd.read_excel(NEW, sheet_name=sheet)
        sheet_audits.append({"sheet": sheet, "rows": int(len(frame)), "columns": int(len(frame.columns)), "has_required": bool({LABEL, GROUP, VIDEO}.issubset(frame.columns))})
        if candidate is None and {LABEL, GROUP, VIDEO}.issubset(frame.columns):
            candidate = frame
            candidate_sheet = sheet
    if candidate is None:
        raise RuntimeError("no sheet contains label/cycle/video columns")

    old_raw = pd.read_excel(OLD)
    new_clean, new_audit = clean(candidate)
    old_clean, old_audit = clean(old_raw)
    old_index = old_clean.set_index("sample_key", drop=False)
    new_index = new_clean.set_index("sample_key", drop=False)
    common_keys = old_index.index.intersection(new_index.index)
    old_only = old_index.index.difference(new_index.index)
    new_only = new_index.index.difference(old_index.index)

    common_columns = [
        column for column in old_clean.columns.intersection(new_clean.columns)
        if column not in {LABEL, "label", "sample_key", "group_key", "video_key", GROUP, VIDEO}
    ]
    field_rows = []
    for column in common_columns:
        old_values = old_index.loc[common_keys, column]
        new_values = new_index.loc[common_keys, column]
        old_missing = meaningful_missing(old_values)
        new_missing = meaningful_missing(new_values)
        field_rows.append({
            "column": column,
            "matched_rows": int(len(common_keys)),
            "old_missing": int(old_missing.sum()),
            "new_missing": int(new_missing.sum()),
            "newly_filled": int((old_missing & ~new_missing).sum()),
            "newly_missing": int((~old_missing & new_missing).sum()),
            "changed_nonmissing": changed_values(old_values, new_values),
            "missing_reduction": int(old_missing.sum() - new_missing.sum()),
        })
    fields = pd.DataFrame(field_rows).sort_values(["newly_filled", "missing_reduction"], ascending=False)

    label_compare = old_index.loc[common_keys, [LABEL, "label"]].rename(columns={LABEL: "old_label_text", "label": "old_label"})
    label_compare = label_compare.join(
        new_index.loc[common_keys, [LABEL, "label"]].rename(columns={LABEL: "new_label_text", "label": "new_label"})
    )
    label_changed = label_compare.old_label.ne(label_compare.new_label)
    text_changed = label_compare.old_label_text.astype("string").str.strip().ne(label_compare.new_label_text.astype("string").str.strip())

    summary = {
        "new_file": str(NEW),
        "old_file": str(OLD),
        "new_sheets": sheet_audits,
        "selected_sheet": candidate_sheet,
        "new": new_audit,
        "old": old_audit,
        "alignment": {
            "common_samples": int(len(common_keys)),
            "new_only_samples": int(len(new_only)),
            "old_only_samples": int(len(old_only)),
            "common_cycles": int(len(set(old_index.loc[common_keys, "group_key"]) & set(new_index.loc[common_keys, "group_key"]))),
            "binary_label_changes": int(label_changed.sum()),
            "label_text_changes": int(text_changed.sum()),
        },
        "columns": {
            "new_count": int(len(candidate.columns)),
            "old_count": int(len(old_raw.columns)),
            "new_only": sorted(set(candidate.columns) - set(old_raw.columns)),
            "old_only": sorted(set(old_raw.columns) - set(candidate.columns)),
        },
        "fill_summary": {
            "total_newly_filled_cells": int(fields.newly_filled.sum()),
            "fields_with_new_values": int(fields.newly_filled.gt(0).sum()),
            "total_newly_missing_cells": int(fields.newly_missing.sum()),
            "total_changed_nonmissing_cells": int(fields.changed_nonmissing.sum()),
            "top_filled_fields": fields.head(25).to_dict("records"),
        },
    }
    fields.to_csv(HERE / "field_missingness_comparison.csv", index=False)
    label_compare.loc[label_changed | text_changed].reset_index().to_csv(HERE / "label_changes.csv", index=False)
    pd.DataFrame({"sample_key": new_only}).to_csv(HERE / "new_only_samples.csv", index=False)
    pd.DataFrame({"sample_key": old_only}).to_csv(HERE / "old_only_samples.csv", index=False)
    new_clean.to_csv(HERE / "clean_updated_table.csv", index=False)
    (HERE / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
