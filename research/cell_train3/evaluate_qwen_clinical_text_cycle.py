#!/usr/bin/env python3
# 同周期文本拼接、Qwen3-Embedding冻结特征抽取与早期监督头。
# 原始文本、周期ID与embedding缓存不可随公开代码上传。
"""Strict cycle CV using Qwen3 embeddings of diagnosis/chromosome/treatment text."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path("/data0/guanhuancong/cell/cell_train3");RAW=ROOT.parent/"细胞表格数据 tjf.xlsx"
ANCHOR=ROOT/"updated_table_tjf_v1/locked_endocrine_residual_oof.csv";OUT=ROOT/"qwen_clinical_text_cycle_v1"
MODEL=Path("/data0/guanhuancong/model/hf_models/Qwen__Qwen3-Embedding-0.6B")
TEXT_FIELDS=["诊断与方案-诊断","女方-不孕因素","女方染色体","男方染色体","用药方案类型","IVF/ICSI/PGT1号","Gn用药名称"]
def key(v):
 # 历史兼容逻辑只保留数字，可能碰撞含字母的不同视频ID。
 # 本次注释版保留原行为以便追溯，不应在新数据上未经审计直接沿用。
 s="".join(c for c in str(v).split('.')[0] if c.isdigit());return f"{int(s):06d}" if s else str(v)
def metric(y,p):
 q=(p>=.5).astype(int);return {"accuracy":float(accuracy_score(y,q)),"balanced_accuracy":float(balanced_accuracy_score(y,q)),
 "f1":float(f1_score(y,q)),"auc":float(roc_auc_score(y,p))}
def text(row):
 # 将已记录的7类字段拼成文本，跳过缺失值；不把最终标签作为编码输入。
 pieces=[]
 for field in TEXT_FIELDS:
  value=row[field]
  if pd.notna(value) and str(value).strip() not in ("","nan","无","0"):pieces.append(f"{field}：{str(value).strip()}")
 return "；".join(pieces) if pieces else "无已记录的诊断或治疗文本"
def main():
 # 原始Excel与文本缓存均属私有数据。Qwen3-Embedding只提取表示，
 # 后续Ridge/PLS才使用监督目标；本文件不使用Qwen2.5-VL视频模型。
 anchor=pd.read_csv(ANCHOR,dtype={"sample_id":str,"group_id":str});anchor.label=anchor.label.astype(int);anchor.fold=anchor.fold.astype(int)
 raw=pd.read_excel(RAW);raw["_key"]=raw.video_key_numeric.map(key);raw["group_id"]=raw["周期号"].astype(str);raw["sample_id"]=raw.group_id+"::"+raw._key
 raw=raw.drop_duplicates("sample_id");data=anchor.merge(raw[["sample_id","group_id"]+TEXT_FIELDS],on=["sample_id","group_id"],validate="one_to_one")
 cycles=[]
 for gid,part in data.groupby("group_id",sort=False):
  cycles.append({"group_id":gid,"fold":int(part.fold.iloc[0]),"n":len(part),"k":int(part.label.sum()),"rate":float(part.label.mean()),
                 "side":int(part.label.mean()>=.5),"text":text(part.iloc[0])})
 cycles=pd.DataFrame(cycles);OUT.mkdir(parents=True,exist_ok=True);cache=OUT/"cycle_text_embeddings.pt"
 if cache.exists():payload=torch.load(cache,map_location="cpu",weights_only=True);embedding=payload["embedding"].float().numpy()
 else:
  model=SentenceTransformer(str(MODEL),device="cuda");embedding=model.encode(cycles.text.tolist(),batch_size=16,
    prompt_name="document",normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
  torch.save({"group_id":cycles.group_id.tolist(),"embedding":torch.from_numpy(embedding).half(),"fields":TEXT_FIELDS},cache)
 predictions={"ridge16":np.zeros(len(anchor)),"ridge32":np.zeros(len(anchor)),"side32":np.zeros(len(anchor))};folds=[]
 for fold in range(5):
  tr=np.flatnonzero(cycles.fold.ne(fold));te=np.flatnonzero(cycles.fold.eq(fold));fold_result={"fold":fold}
  for components in (16,32):
   scaler=StandardScaler().fit(embedding[tr]);a=scaler.transform(embedding[tr]);b=scaler.transform(embedding[te])
   pca=PCA(n_components=components,whiten=True,random_state=260905).fit(a);a=pca.transform(a);b=pca.transform(b)
   ridge=Ridge(alpha=20.).fit(a,cycles.rate.iloc[tr],sample_weight=cycles.n.iloc[tr]);value=np.clip(ridge.predict(b),.001,.999)
   mapping=dict(zip(cycles.group_id.iloc[te],value));mask=anchor.fold.eq(fold).to_numpy();predictions[f"ridge{components}"][mask]=anchor.loc[mask,"group_id"].map(mapping)
   if components==32:
    side=LogisticRegression(C=.03,class_weight="balanced",max_iter=3000,random_state=260905).fit(a,cycles.side.iloc[tr],sample_weight=np.maximum(abs(2*cycles.k.iloc[tr]-cycles.n.iloc[tr]),1))
    value=side.predict_proba(b)[:,1];mapping=dict(zip(cycles.group_id.iloc[te],value));predictions["side32"][mask]=anchor.loc[mask,"group_id"].map(mapping)
  folds.append(fold_result);print(json.dumps(fold_result),flush=True)
 y=anchor.label.to_numpy(int);base=anchor.base_probability.to_numpy(float);results={"anchor":metric(y,base)}
 for name,value in predictions.items():
  results[name]=metric(y,value)
  for weight in (.05,.1,.2,.3):results[f"anchor_{name}_{weight:g}"]=metric(y,(1-weight)*base+weight*value)
 summary={"protocol":"Qwen3-Embedding-0.6B frozen clinical-text embeddings; strict grouped five-fold PCA+regularized heads",
          "n":len(anchor),"cycles":len(cycles),"fields":TEXT_FIELDS,"results":results}
 output=anchor.copy()
 for name,value in predictions.items():output[f"qwen_text_{name}"]=value
 output.to_csv(OUT/"oof_qwen_text_cycle.csv",index=False);(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
