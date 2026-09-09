#!/usr/bin/env python3
# 公共数量解码评估器及历史更新表格对照入口。
# 核心evaluate()供PLS2版本调用。所依赖的概率与排序必须先核查折外来源。
"""Re-run the locked video Top-K protocol on the actually best updated table OOF."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT=Path("/data0/guanhuancong/cell/cell_train3");BEST=ROOT/"cycle_count_topk_bestvideo_v1"
ANCHOR=ROOT/"updated_table_tjf_v1/locked_endocrine_residual_oof.csv"
TEXT=ROOT/"qwen_clinical_text_cycle_v1/oof_qwen_text_cycle.csv";OUT=ROOT/"updated_table_video_topk_v1"
sys.path.insert(0,str(BEST));from run_best_video_topk import crossfit_dist,decode,select_policy,metrics

def evaluate(data):
 # 共享评估器：每个外折单独拟合数量模型并选择门控，输出顺序保持与data一致。
 # 调用方负责上游概率的折外来源，以及sample_id和group_id的一致性检查。
 pred=np.zeros(len(data),int);prob=data.p_classify.to_numpy(float).copy();policies=[]
 for outer in sorted(data.fold.unique()):
  development=data[data.fold.ne(outer)].copy();test=data[data.fold.eq(outer)].copy();inner_by_c={};test_by_c={}
  for c in (.05,.2,1.):inner_by_c[c],test_by_c[c]=crossfit_dist(data,int(outer),c)
  policy=select_policy(development,inner_by_c);policy["outer"]=int(outer);dist=test_by_c[policy["C"]]
  fp,fq,acted=decode(test,dist,policy["post_thr"],policy["margin_thr"],policy["entropy_thr"],policy["rank_gap_thr"],policy["max_changes"])
  mask=data.fold.eq(outer).to_numpy();pred[mask]=fp;prob[mask]=fq;policy["test_acted"]=int(acted.sum());policies.append(policy)
 return pred,prob,policies

def main():
 # 历史对照入口，生成后续PLS2入口需要的旧版本比较文件。
 # 这里的5%文本Ridge版本不是最终锁定的2.5%文本PLS2版本。
 data=pd.read_csv(BEST/"input_best_video.csv",dtype={"sample_id":str,"group_id":str,"video_id":str}).reset_index(drop=True)
 anchor=pd.read_csv(ANCHOR,dtype={"sample_id":str})[["sample_id","base_probability"]]
 text=pd.read_csv(TEXT,dtype={"sample_id":str})[["sample_id","qwen_text_ridge16"]]
 data=data.merge(anchor,on="sample_id",validate="one_to_one").merge(text,on="sample_id",validate="one_to_one")
 data.label=data.label.astype(int);data.fold=data.fold.astype(int);old_p=data.p_classify.to_numpy(float).copy()
 variants={"updated_anchor":data.base_probability.to_numpy(float),
           "updated_anchor_qwen05":.95*data.base_probability.to_numpy(float)+.05*data.qwen_text_ridge16.to_numpy(float)}
 y=data.label.to_numpy(int);retained=pd.read_csv(BEST/"oof_discrete_topk.csv",dtype={"sample_id":str}).set_index("sample_id").loc[data.sample_id]
 summary={"protocol":"best updated table OOF replaces stale two-cycle probabilities; full strict nested P(K)+current video Top-K rerun",
          "old_table":metrics(y,(old_p>=.5).astype(int),old_p),
          "old_retained_topk":metrics(y,retained.pred_topk_v2.to_numpy(int),retained.prob_topk_v2.to_numpy(float)),"variants":{}}
 output=data.copy()
 for name,value in variants.items():
  trial=data.copy();trial["p_classify"]=value;pred,prob,policies=evaluate(trial)
  summary["variants"][name]={"table":metrics(y,(value>=.5).astype(int),value),"topk":metrics(y,pred,prob),
    "delta_topk_vs_table":float(np.mean(pred==y)-np.mean((value>=.5)==y)),
    "delta_vs_old_retained":float(np.mean(pred==y)-np.mean(retained.pred_topk_v2.to_numpy(int)==y)),
    "changed_vs_old_retained":int(np.sum(pred!=retained.pred_topk_v2.to_numpy(int))),"policies":policies}
  output[f"pred_{name}_topk"]=pred;output[f"prob_{name}_topk"]=prob
 output.to_csv(OUT/"oof_updated_table_video_topk.csv",index=False) if OUT.exists() else None
 OUT.mkdir(parents=True,exist_ok=True);output.to_csv(OUT/"oof_updated_table_video_topk.csv",index=False)
 (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
