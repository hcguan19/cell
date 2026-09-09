#!/usr/bin/env python3
# 冻结临床文本Embedding之后的低容量监督头比较。
# 最终采用pls_2概率，其他Ridge/LogReg/PLS维数只保留作历史对照。
"""Alternative regularized heads over cached Qwen3 clinical-text embeddings."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path("/data0/guanhuancong/cell/cell_train3");ANCHOR=ROOT/"updated_table_tjf_v1/locked_endocrine_residual_oof.csv"
SOURCE=ROOT/"qwen_clinical_text_cycle_v1";OUT=ROOT/"qwen_text_heads_v2"
def metric(y,p):
 q=(p>=.5).astype(int);return {"acc":float(accuracy_score(y,q)),"bacc":float(balanced_accuracy_score(y,q)),"f1":float(f1_score(y,q)),"auc":float(roc_auc_score(y,p))}
def main():
 # 文本编码器冻结，只训练小容量统计头。每个周期只有一个1024维向量。
 # 字段、缓存ID顺序和fold必须与表格锚点一致；不能把样本行序当周期行序。
 anchor=pd.read_csv(ANCHOR,dtype={"sample_id":str,"group_id":str});anchor.label=anchor.label.astype(int);anchor.fold=anchor.fold.astype(int)
 old=pd.read_csv(SOURCE/"oof_qwen_text_cycle.csv",dtype={"group_id":str})
 payload=torch.load(SOURCE/"cycle_text_embeddings.pt",map_location="cpu",weights_only=True);gid=payload["group_id"];x=payload["embedding"].float().numpy()
 cycles=anchor.groupby("group_id",sort=False).agg(fold=("fold","first"),n=("label","size"),k=("label","sum"),rate=("label","mean")).reset_index()
 position={g:i for i,g in enumerate(gid)};x=np.asarray([x[position[g]] for g in cycles.group_id]);heads={}
 for alpha in (10.,100.,1000.):heads[f"ridge_{alpha:g}"]=np.zeros(len(anchor))
 for comp in (2,4,8):heads[f"pls_{comp}"]=np.zeros(len(anchor))
 for c in (.001,.01,.1):heads[f"logreg_{c:g}"]=np.zeros(len(anchor))
 for fold in range(5):
  # 特征标准化限于训练周期；本折测试周期只transform。
  tr=np.flatnonzero(cycles.fold.ne(fold));te=np.flatnonzero(cycles.fold.eq(fold));scaler=StandardScaler().fit(x[tr]);a=scaler.transform(x[tr]);b=scaler.transform(x[te])
  values={}
  for alpha in (10.,100.,1000.):values[f"ridge_{alpha:g}"]=np.clip(Ridge(alpha=alpha).fit(a,cycles.rate.iloc[tr],sample_weight=cycles.n.iloc[tr]).predict(b),.001,.999)
  # Repeat cycle rows according to cohort size to approximate embryo-weighted PLS.
  # 重复周期行近似按周期样本数加权PLS，目标仍是周期阳性比例。
  # 最终模型只取pls_2；其余头及混合权重是历史筛选对照，非联合集成。
  repeat=np.repeat(np.arange(len(tr)),cycles.n.iloc[tr].to_numpy(int));ar=a[repeat];yr=cycles.rate.iloc[tr].to_numpy()[repeat]
  for comp in (2,4,8):values[f"pls_{comp}"]=np.clip(PLSRegression(n_components=comp,scale=False,max_iter=1000).fit(ar,yr).predict(b).ravel(),.001,.999)
  for c in (.001,.01,.1):
   model=LogisticRegression(C=c,class_weight="balanced",max_iter=3000,random_state=260905)
   model.fit(a,cycles.rate.iloc[tr].ge(.5).astype(int),sample_weight=np.maximum(abs(2*cycles.k.iloc[tr]-cycles.n.iloc[tr]),1))
   values[f"logreg_{c:g}"]=model.predict_proba(b)[:,1]
  mask=anchor.fold.eq(fold).to_numpy()
  for name,value in values.items():heads[name][mask]=anchor.loc[mask,"group_id"].map(dict(zip(cycles.group_id.iloc[te],value)))
  print(json.dumps({"fold":fold}),flush=True)
 y=anchor.label.to_numpy(int);base=anchor.base_probability.to_numpy(float);results={"anchor":metric(y,base)}
 for name,p in heads.items():
  results[name]=metric(y,p)
  for w in (.025,.05,.075,.1,.15,.2):results[f"anchor_{name}_{w:g}"]=metric(y,(1-w)*base+w*p)
 # 这里汇总全量OOF后按成绩查看候选，属于探索性模型选择。
 # 不能把同一份OOF上的最优值当作未参与选择的独立确认集成绩。
 best=sorted(({"name":k,**v} for k,v in results.items() if k.startswith("anchor_")),key=lambda r:(r["acc"],r["auc"]),reverse=True)[:15]
 summary={"protocol":"cached frozen Qwen3 text; strict fixed cycle five-fold; direct Ridge/PLS/logistic exploratory heads","results":results,"best":best}
 OUT.mkdir(parents=True,exist_ok=True);output=anchor.copy()
 for name,p in heads.items():output[name]=p
 output.to_csv(OUT/"oof_text_heads.csv",index=False);(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"anchor":results["anchor"],"best":best},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
