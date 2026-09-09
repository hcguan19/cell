#!/usr/bin/env python3
# 历史概率/视频排序对齐入口。顶层代码执行读写，不能直接作为库import。
# 数字化ID会有碰撞风险，inner join会改变分母，必须另做数据审计。
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "rank_boost_v2/oof_dual_recommended_boost.csv"
VIDEO = ROOT / "dual_video_stacking_v1/oof_all_dual_video_stacking.csv"
OUT = Path(__file__).resolve().parent / "input_best_video.csv"
SCORE = "stack_current_x3dstage_vmae_lr"


def canonical(value):
    # 这是归档版本的数字化键，可能把不同字母前缀合并成同一数字。
    # 下面只删除检测到的重复键，并不能保证未检测到的配对全都正确。
    text = str(value).split("::")[-1]
    digits = re.sub(r"\D", "", text)
    return digits.lstrip("0") or "0"


# 本脚本在模块顶层运行，切勿作为无副作用模块import。
# inner join会缩小队列，运行前后应对每个被排除ID单独保存私有审计报告。
base = pd.read_csv(BASE, dtype={"sample_id": str, "video_id": str, "group_id": str})
video = pd.read_csv(VIDEO, dtype={"sample_id": str})
base["_key"] = base["video_id"].map(canonical)
video["_key"] = video["sample_id"].map(canonical)

bad_base = set(base.loc[base["_key"].duplicated(False), "_key"])
bad_video = set(video.loc[video["_key"].duplicated(False), "_key"])
bad = bad_base | bad_video
base = base[~base["_key"].isin(bad)].copy()
video = video[~video["_key"].isin(bad)].copy()

joined = base.merge(
    video[["_key", "label", SCORE]].rename(columns={"label": "video_label"}),
    on="_key",
    how="inner",
    validate="one_to_one",
)
if not (joined["label"].astype(int) == joined["video_label"].astype(int)).all():
    raise RuntimeError("label mismatch after canonical video alignment")
joined["video_rank_original"] = joined["video_rank"]
joined["video_rank"] = joined[SCORE].astype(float)
joined = joined.drop(columns=["_key", "video_label"])
OUT.parent.mkdir(parents=True, exist_ok=True)
joined.to_csv(OUT, index=False)
print({"n": len(joined), "cycles": joined.group_id.nunique(), "dropped_ambiguous_keys": len(bad), "output": str(OUT)})
