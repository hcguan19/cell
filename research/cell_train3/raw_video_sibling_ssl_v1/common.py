#!/usr/bin/env python3
# 历史像素数据集、X3D模型和同周期排序工具。
# 当前最终排序流程只从这里导入pair_auc；完整模块导入仍需torch/pytorchvideo。
"""Raw-pixel multiscale embryo clips and models."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorchvideo.models.hub import x3d_s
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler


HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; CELL=ROOT.parent
CV_ROOT=CELL/"cell_deal/merged1410_group_cv5"
ROI_PATH=CELL/"paper_reproductions/video_roi_temporal_v1/roi_109_115.json"
PHASES=("0_100","109_115","133_139","157_163")
MEAN=torch.tensor([.45,.45,.45])[:,None,None]; STD=torch.tensor([.225,.225,.225])[:,None,None]
CACHE_ROOT=HERE/"clip_cache"


def norm_id(value):
    text=str(value).strip(); return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def load_records():
    root=CV_ROOT/"fold_0/processed_data"; result={}
    for split in ("train","val","test"):
        for row in json.loads((root/f"{split}_indexed.json").read_text()):
            row=dict(row); row["sample_id"]=norm_id(row["sample_id"]); result[row["sample_id"]]=row
    if len(result)!=1410: raise ValueError(f"expected 1410 records, got {len(result)}")
    return result


def metadata(fold):
    frame=pd.read_csv(CV_ROOT/f"fold_{fold}/audit/assignments.csv",dtype={"sample_id":str,"group_id":str})
    frame["sample_id"]=frame.sample_id.map(norm_id); return frame


def phase_paths(record,phase):
    root=Path(record["video_dir"]); names=record.get("segments",{}).get(phase,[])
    times=record.get("segments_times",{}).get(phase,[])
    return [(root/name,float(time)) for name,time in sorted(zip(names,times),key=lambda z:z[1])]


def indices(length,stride,randomize,rng):
    count=16
    if length<count: return None
    span=1+(count-1)*stride
    if length>=span:
        start=rng.randint(0,length-span) if randomize else (length-span)//2
        return [start+i*stride for i in range(count)]
    # Late windows are short: use 16 distinct real frames over the available span.
    return np.rint(np.linspace(0,length-1,count)).astype(int).tolist()


def clip_tensor(items,roi,stride,randomize,rng,augment=False):
    chosen=indices(len(items),stride,randomize,rng)
    if chosen is None: return torch.zeros(3,16,160,160),False
    frames=[]; x=float(roi["x"]); y=float(roi["y"]); size=max(256,int(round(float(roi["radius"])*2.8)))
    for index in chosen:
        with Image.open(items[index][0]) as image: gray=np.asarray(image.convert("L"),dtype=np.uint8)
        half=size//2; padded=cv2.copyMakeBorder(gray,half,half,half,half,cv2.BORDER_REFLECT_101)
        crop=padded[int(round(y)):int(round(y))+size,int(round(x)):int(round(x))+size]
        crop=cv2.resize(crop,(160,160),interpolation=cv2.INTER_AREA); frames.append(crop)
    stack=np.stack(frames).astype(np.float32); lo,hi=np.percentile(stack,[1,99]); stack=np.clip((stack-lo)/max(hi-lo,1.),0,1)
    if augment:
        if rng.random()<.5: stack=stack[:,:,::-1].copy()
        if rng.random()<.5: stack=stack[:,::-1,:].copy()
        stack=np.clip(stack*(.9+.2*rng.random())+(.04*rng.random()-.02),0,1)
    tensor=torch.from_numpy(stack.copy()).unsqueeze(0).repeat(3,1,1,1)
    return (tensor-MEAN.unsqueeze(1))/STD.unsqueeze(1),True


class PhasePairs(Dataset):
    def __init__(self,records,meta,split,seed=0):
        self.records=records; self.rows=meta.loc[meta.split.eq(split)].reset_index(drop=True); self.roi=json.loads(ROI_PATH.read_text()); self.seed=seed; self.epoch=0
        self.items=[(row.sample_id,p) for row in self.rows.itertuples() for p,phase in enumerate(PHASES)
                    if len(phase_paths(records[row.sample_id],phase))>=16]
    def __len__(self): return len(self.items)
    def __getitem__(self,index):
        sid,phase=self.items[index]; rng=random.Random(self.seed+self.epoch*100000+index); items=phase_paths(self.records[sid],PHASES[phase])
        short,_=clip_tensor(items,self.roi[sid],5,True,rng,True); long,_=clip_tensor(items,self.roi[sid],10,True,rng,True)
        return short,long,phase


class Embryos(Dataset):
    def __init__(self,records,meta,split,seed=0,train=False):
        self.records=records; self.rows=meta.loc[meta.split.eq(split)].reset_index(drop=True); self.roi=json.loads(ROI_PATH.read_text()); self.seed=seed; self.epoch=0; self.train=train
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows.iloc[index]; sid=row.sample_id; rng=random.Random(self.seed+self.epoch*100000+index); clips=[];valid=[]
        # Alternate short/long coverage by epoch during training; fixed long view for evaluation.
        stride=(5 if self.train and (self.epoch+index)%2==0 else 10)
        for phase in PHASES:
            clip,ok=clip_tensor(phase_paths(self.records[sid],phase),self.roi[sid],stride,self.train,rng,self.train)
            clips.append(clip);valid.append(ok)
        return {"clips":torch.stack(clips),"valid":torch.tensor(valid),"label":int(row.label),
                "sample_id":sid,"group_id":str(row.group_id)}


def load_clip_cache():
    stores={}; index={}
    for shard in range(6):
        ids=json.loads((CACHE_ROOT/f"ids_{shard}.json").read_text())
        clips=np.load(CACHE_ROOT/f"clips_{shard}.npy",mmap_mode="r")
        valid=np.load(CACHE_ROOT/f"valid_{shard}.npy",mmap_mode="r")
        stores[shard]=(clips,valid)
        for row,sid in enumerate(ids): index[sid]=(shard,row)
    if len(index)!=1410: raise ValueError(f"clip cache has {len(index)} embryos")
    return stores,index


def cached_tensor(array,augment,rng):
    value=np.asarray(array,dtype=np.float32)/255.
    if augment:
        if rng.random()<.5:value=value[:,:,::-1].copy()
        if rng.random()<.5:value=value[:,::-1,:].copy()
        value=np.clip(value*(.9+.2*rng.random())+(.04*rng.random()-.02),0,1)
    tensor=torch.from_numpy(value.copy()).unsqueeze(0).repeat(3,1,1,1)
    return (tensor-MEAN.unsqueeze(1))/STD.unsqueeze(1)


class CachedPhasePairs(Dataset):
    def __init__(self,meta,split,seed=0):
        self.stores,self.index=load_clip_cache();self.rows=meta.loc[meta.split.eq(split)].reset_index(drop=True);self.seed=seed;self.epoch=0
        self.items=[]
        for row in self.rows.itertuples():
            shard,pos=self.index[row.sample_id];valid=self.stores[shard][1][pos]
            self.items.extend((row.sample_id,phase) for phase in range(4) if valid[phase*2] and valid[phase*2+1])
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        sid,phase=self.items[i];shard,pos=self.index[sid];clips=self.stores[shard][0][pos];rng=random.Random(self.seed+self.epoch*100000+i)
        return cached_tensor(clips[phase*2],True,rng),cached_tensor(clips[phase*2+1],True,rng),phase


class CachedEmbryos(Dataset):
    def __init__(self,meta,split,seed=0,train=False):
        self.stores,self.index=load_clip_cache();self.rows=meta.loc[meta.split.eq(split)].reset_index(drop=True);self.seed=seed;self.epoch=0;self.train=train
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows.iloc[i];sid=row.sample_id;shard,pos=self.index[sid];cache,valid=self.stores[shard][0][pos],self.stores[shard][1][pos];rng=random.Random(self.seed+self.epoch*100000+i);clips=[];mask=[]
        scale=((self.epoch+i)%2 if self.train else 1)
        for phase in range(4):
            location=phase*2+scale;ok=bool(valid[location]);clips.append(cached_tensor(cache[location],self.train,rng) if ok else torch.zeros(3,16,160,160));mask.append(ok)
        return {"clips":torch.stack(clips),"valid":torch.tensor(mask),"label":int(row.label),"sample_id":sid,"group_id":str(row.group_id)}


class GroupSampler(Sampler):
    def __init__(self,dataset,groups_per_batch=3,seed=0):
        self.groups=defaultdict(list);self.seed=seed;self.epoch=0;self.groups_per_batch=groups_per_batch
        for i,row in dataset.rows.iterrows(): self.groups[str(row.group_id)].append(i)
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch);self.epoch+=1;names=list(self.groups);rng.shuffle(names)
        for start in range(0,len(names),self.groups_per_batch):
            yield [i for name in names[start:start+self.groups_per_batch] for i in self.groups[name]]
    def __len__(self): return (len(self.groups)+self.groups_per_batch-1)//self.groups_per_batch


def backbone(trainable=True):
    model=x3d_s(pretrained=True); model.blocks[-1].proj=nn.Identity();model.blocks[-1].activation=nn.Identity()
    for p in model.parameters(): p.requires_grad=False
    if trainable:
        for block in model.blocks[4:]:
            for p in block.parameters(): p.requires_grad=True
    return model


def freeze_bn(module):
    for child in module.modules():
        if isinstance(child,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)): child.eval()


class SSLModel(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=backbone(True);self.project=nn.Sequential(nn.Linear(2048,256),nn.GELU(),nn.Linear(256,128));self.phase=nn.Linear(2048,4)
    def forward(self,x):
        feature=self.backbone(x).flatten(1);return feature,F.normalize(self.project(feature),dim=-1),self.phase(feature)


def contrastive(a,b,temp=.15):
    logits=a@b.T/temp;target=torch.arange(len(a),device=a.device)
    return (F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2


class EmbryoModel(nn.Module):
    def __init__(self,ssl_state=None):
        super().__init__();self.backbone=backbone(True)
        if ssl_state is not None:self.backbone.load_state_dict(ssl_state)
        self.input=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,192),nn.GELU())
        self.phase=nn.Parameter(torch.randn(4,192)*.02)
        layer=nn.TransformerEncoderLayer(192,4,384,.15,batch_first=True,norm_first=True,activation="gelu")
        self.temporal=nn.TransformerEncoder(layer,1);self.attention=nn.Linear(192,1);self.score=nn.Linear(192,1)
    def forward(self,clips,valid):
        b,p=clips.shape[:2];flat=clips.reshape(b*p,*clips.shape[2:]);mask=valid.reshape(-1)
        if mask.any():
            encoded=self.backbone(flat[mask]).flatten(1)
            # autocast makes X3D output fp16 while the input/cache is fp32.
            # Allocate missing-phase placeholders in the encoded dtype so the
            # indexed write remains valid in both train and evaluation modes.
            feature=torch.zeros((b*p,2048),device=flat.device,dtype=encoded.dtype)
            feature[mask]=encoded
        else:
            feature=flat.new_zeros(b*p,2048)
        tokens=self.input(feature).reshape(b,p,-1)+self.phase.unsqueeze(0);tokens=self.temporal(tokens,src_key_padding_mask=~valid)
        weight=self.attention(tokens).squeeze(-1).masked_fill(~valid,-1e4).softmax(1)
        pooled=(tokens*weight.unsqueeze(-1)).sum(1);return self.score(pooled).squeeze(1)


def centered(scores,groups):
    output=scores.clone()
    for group in sorted(set(groups)):
        idx=torch.tensor([i for i,g in enumerate(groups) if g==group],device=scores.device);output[idx]-=scores[idx].mean()
    return output


def rank_loss(scores,labels,groups):
    losses=[]
    for group in sorted(set(groups)):
        idx=torch.tensor([i for i,g in enumerate(groups) if g==group],device=scores.device); y=labels[idx];s=scores[idx]
        pos,neg=s[y>0],s[y==0]
        if len(pos) and len(neg):losses.append(F.softplus(-(pos[:,None]-neg[None,:])).mean())
    return torch.stack(losses).mean() if losses else scores.sum()*0


def pair_auc(labels,scores,groups):
    values=[]
    for group in sorted(set(groups)):
        idx=np.asarray([i for i,g in enumerate(groups) if g==group]);y=np.asarray(labels)[idx];s=np.asarray(scores)[idx];pos=s[y==1];neg=s[y==0]
        if len(pos) and len(neg):
            diff=(pos[:,None]-neg[None,:]).ravel();values.extend((diff>0)+.5*(diff==0))
    return float(np.mean(values)) if values else .5
