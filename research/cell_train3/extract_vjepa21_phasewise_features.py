#!/usr/bin/env python3
# 冻结V-JEPA2.1逐阶段特征抽取，保留8个时间格和3×3空间格。
# 这里只做特征缓存，无标签梯度训练；外部编码器源码/权重单独配置。
"""Extract V-JEPA2.1 from each continuous 16-frame stage independently."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import extract_vjepa21_features as base


OUT = base.ROOT / "vjepa21_vitb_video_v3_phasewise/features"


def phase_videos(cache, valid, device):
    # 输入来自旧像素缓存：每阶段有短/长跨度两个16帧视图，共8个视图。
    # 优先长视图，无效时回退短视图；并非在本函数重新均匀采样原始视频。
    # 返回有效阶段视频(B,3,16,384,384)、对应阶段序号和长度4的有效mask。
    arrays, positions = [], []
    mask = np.zeros(4, dtype=np.bool_)
    for phase in range(4):
        long_index, short_index = phase * 2 + 1, phase * 2
        location = long_index if valid[long_index] else short_index
        if bool(valid[location]):
            arrays.append(np.asarray(cache[location], dtype=np.float32) / 255.0)
            positions.append(phase)
            mask[phase] = True
    if not arrays:
        return None, positions, mask
    video = torch.from_numpy(np.stack(arrays)).to(device)
    # 原缓存为灰度160×160，复制到3通道；这不会生成新的颜色信息。
    video = video[:, :, None].repeat(1, 1, 3, 1, 1)
    batch, frames = video.shape[:2]
    video = video.reshape(batch * frames, 3, 160, 160)
    video = F.interpolate(video, size=(384, 384), mode="bilinear", align_corners=False)
    video = video.reshape(batch, frames, 3, 384, 384)
    mean = video.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
    std = video.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    video = ((video - mean) / std).permute(0, 2, 1, 3, 4)
    return video, positions, mask


def main():
    # 每个分片独立加载冻结编码器。没有optimizer、backward或训练标签输入。
    # 默认6分片需与load_features的读取约定一致；不能只改本脚本分片数。
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    frame = pd.read_csv(base.META, dtype={"sample_id": str, "group_id": str})
    # 间隔切片分配全局行号，保存rows_*.npy用于后续顺序恢复。
    rows = np.arange(len(frame))[args.shard::args.num_shards]
    stores, index = base.load_clip_index()
    encoder = base.build_encoder(16, device)
    features = np.zeros((len(rows), 4, 8, 9, 768), dtype=np.float16)
    masks = np.zeros((len(rows), 4), dtype=np.bool_)
    started = time.time()
    with torch.inference_mode():
        for local_row, global_row in enumerate(rows):
            sid = frame.iloc[global_row].sample_id
            # 这是历史ID映射约定。不得靠去掉字母来“修复”不匹配；
            # 视频键与周期归属必须在抽取前核对，避免错配另一个样本。
            cache_id = str(sid).split("::")[-1].lstrip("0") or "0"
            source_shard, source_row = index[cache_id]
            cache = stores[source_shard][0][source_row]
            valid = stores[source_shard][1][source_row]
            video, positions, mask = phase_videos(cache, valid, device)
            masks[local_row] = mask
            if video is not None:
                # 16帧以tubelet_size=2形成8个时间格，384/16=24个空间patch。
                # 空间自适应平均到3×3，保留时间顺序与粗空间位置，最终4×8×9×768。
                tokens = encoder(video).reshape(len(positions), 8, 24, 24, 768)
                grid = tokens.permute(0, 1, 4, 2, 3).reshape(len(positions) * 8, 768, 24, 24)
                grid = F.adaptive_avg_pool2d(grid, (3, 3))
                grid = grid.reshape(len(positions), 8, 768, 3, 3).permute(0, 1, 3, 4, 2)
                features[local_row, positions] = grid.reshape(len(positions), 8, 9, 768).cpu().numpy().astype(np.float16)
            if (local_row + 1) % 25 == 0:
                rate = (local_row + 1) / max(time.time() - started, 1e-6)
                print(f"shard={args.shard} {local_row+1}/{len(rows)} rate={rate:.2f}/s", flush=True)
    np.save(OUT / f"features_{args.shard}.npy", features)
    np.save(OUT / f"rows_{args.shard}.npy", rows)
    np.save(OUT / f"phase_valid_{args.shard}.npy", masks)
    print(f"done shard={args.shard} n={len(rows)} seconds={time.time()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
