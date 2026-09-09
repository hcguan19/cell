#!/usr/bin/env python3
# V-JEPA2.1编码器构建和私有像素缓存读取的共享实现。
# MODEL_REPO、CKPT、CACHE等路径是历史配置，移机时需显式修改。
"""Cache frozen V-JEPA2.1 ViT-B spatiotemporal features for aligned embryos."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path("/data0/guanhuancong/cell/cell_train3")
MODEL_REPO = Path("/data0/guanhuancong/cell/model/vjepa2_src/vjepa2-main")
CKPT = Path("/data0/guanhuancong/cell/model/vjepa2_1_vitb_dist_vitG_384.pt")
CACHE = ROOT / "raw_video_sibling_ssl_v1/clip_cache"
META = ROOT / "cycle_count_topk_bestvideo_v1/oof_discrete_topk.csv"
OUT = ROOT / "vjepa21_vitb_video_v1/features"


def build_encoder(frames: int, device: torch.device):
    # 外部V-JEPA2.1源码和权重必须由使用者按其许可证单独准备。
    # 读取EMA教师编码器，去掉分布式前缀后strict加载，防止层名不匹配被忽略。
    sys.path.insert(0, str(MODEL_REPO))
    from app.vjepa_2_1.models.vision_transformer import vit_base

    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=True)
    state = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in checkpoint["ema_encoder"].items()
    }
    encoder = vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=frames,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    encoder.load_state_dict(state, strict=True)
    del checkpoint, state
    return encoder.eval().to(device)


def load_clip_index():
    # 建立旧像素分片的ID到(分片,行号)映射；这里只读取私有缓存，不随仓库分发。
    stores, index = {}, {}
    for shard in range(6):
        ids = json.loads((CACHE / f"ids_{shard}.json").read_text())
        clips = np.load(CACHE / f"clips_{shard}.npy", mmap_mode="r")
        valid = np.load(CACHE / f"valid_{shard}.npy", mmap_mode="r")
        stores[shard] = (clips, valid)
        for row, sid in enumerate(ids):
            index[(str(sid).lstrip("0") or "0")] = (shard, row)
    return stores, index


def make_video(cache, valid, device):
    phases, mask = [], []
    # Four chronologically ordered frames from the long-coverage clip per stage.
    frame_indices = [0, 5, 10, 15]
    for phase in range(4):
        long_index, short_index = phase * 2 + 1, phase * 2
        location = long_index if valid[long_index] else short_index
        ok = bool(valid[location])
        if ok:
            value = np.asarray(cache[location][frame_indices], dtype=np.float32) / 255.0
        else:
            value = np.zeros((4, 160, 160), dtype=np.float32)
        phases.append(value)
        mask.append(ok)
    video = torch.from_numpy(np.concatenate(phases, axis=0)).to(device)
    video = video[:, None].repeat(1, 3, 1, 1)
    video = F.interpolate(video, size=(384, 384), mode="bilinear", align_corners=False)
    mean = video.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = video.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    video = ((video - mean) / std).permute(1, 0, 2, 3).unsqueeze(0)
    return video, np.asarray(mask, dtype=np.bool_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    frame = pd.read_csv(META, dtype={"sample_id": str, "group_id": str})
    rows = np.arange(len(frame))[args.shard :: args.num_shards]
    stores, index = load_clip_index()
    encoder = build_encoder(16, device)
    features = np.empty((len(rows), 8, 9, 768), dtype=np.float16)
    masks = np.empty((len(rows), 4), dtype=np.bool_)
    started = time.time()
    with torch.inference_mode():
        for local_row, global_row in enumerate(rows):
            sid = frame.iloc[global_row].sample_id
            cache_id = (str(sid).split("::")[-1].lstrip("0") or "0")
            source_shard, source_row = index[cache_id]
            cache, valid = stores[source_shard][0][source_row], stores[source_shard][1][source_row]
            video, phase_mask = make_video(cache, valid, device)
            tokens = encoder(video).reshape(1, 8, 24, 24, 768)
            grid = tokens.permute(0, 1, 4, 2, 3).reshape(8, 768, 24, 24)
            grid = F.adaptive_avg_pool2d(grid, (3, 3)).permute(0, 2, 3, 1).reshape(8, 9, 768)
            features[local_row] = grid.float().cpu().numpy().astype(np.float16)
            masks[local_row] = phase_mask
            if (local_row + 1) % 25 == 0:
                rate = (local_row + 1) / max(time.time() - started, 1e-6)
                print(f"shard={args.shard} {local_row + 1}/{len(rows)} rate={rate:.2f}/s", flush=True)
    np.save(OUT / f"features_{args.shard}.npy", features)
    np.save(OUT / f"rows_{args.shard}.npy", rows)
    np.save(OUT / f"phase_valid_{args.shard}.npy", masks)
    print(f"done shard={args.shard} n={len(rows)} seconds={time.time()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
