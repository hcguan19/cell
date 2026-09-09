#!/usr/bin/env python3
# 表格锚点97.5% + 临床文本PLS2概率2.5%的固定组合入口。
# 输出供最终阶段视频融合使用，历史比较文件也是原脚本的必要输入。
"""Full nested Top-K evaluation on the Qwen-PLS2 2.5% table residual."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path("/data0/guanhuancong/cell/cell_train3");BEST=ROOT/"cycle_count_topk_bestvideo_v1";OUT=ROOT/"qwen_pls2_video_topk_v1"
sys.path.insert(0,str(ROOT));from evaluate_updated_table_video_topk import evaluate
sys.path.insert(0,str(BEST));from run_best_video_topk import metrics
def main():
 # 消费已生成的周期表格锚点、文本PLS2概率和既有视频排序，不重训视觉主干。
 data=pd.read_csv(BEST/"input_best_video.csv",dtype={"sample_id":str,"group_id":str,"video_id":str}).reset_index(drop=True)
 anchor=pd.read_csv(ROOT/"updated_table_tjf_v1/locked_endocrine_residual_oof.csv",dtype={"sample_id":str})[["sample_id","base_probability"]]
 text=pd.read_csv(ROOT/"qwen_text_heads_v2/oof_text_heads.csv",dtype={"sample_id":str})[["sample_id","pls_2"]]
 data=data.merge(anchor,on="sample_id",validate="one_to_one").merge(text,on="sample_id",validate="one_to_one")
 # 2.5%是历史锁定的文本概率混合系数，不是端到端可训练门控。
 # PLS在上游拟合周期阳性比例，周期概率展开回样本后在这里混合。
 data.label=data.label.astype(int);data.fold=data.fold.astype(int);p=.975*data.base_probability.to_numpy(float)+.025*data.pls_2.to_numpy(float)
 data["p_classify"]=p;pred,prob,policies=evaluate(data);y=data.label.to_numpy(int)
 # 旧版本文件仅用于配对对照统计，但原入口仍要求其存在。
 old=pd.read_csv(ROOT/"updated_table_video_topk_v1/oof_updated_table_video_topk.csv",dtype={"sample_id":str}).set_index("sample_id").loc[data.sample_id]
 oldpred=old.pred_updated_anchor_qwen05_topk.to_numpy(int)
 summary={"protocol":"fixed Qwen PLS2 2.5% residual + full strict nested P(K) current-video Top-K",
  "table":metrics(y,(p>=.5).astype(int),p),"topk":metrics(y,pred,prob),
  "delta_topk_vs_table":float(np.mean(pred==y)-np.mean((p>=.5)==y)),
  "delta_vs_previous_best":float(np.mean(pred==y)-np.mean(oldpred==y)),
  "won_vs_previous":int(np.sum((pred==y)&(oldpred!=y))),"lost_vs_previous":int(np.sum((pred!=y)&(oldpred==y))),"policies":policies}
 OUT.mkdir(parents=True,exist_ok=True);result=data.copy();result["pred_pls2_topk"]=pred;result["prob_pls2_topk"]=prob
 result.to_csv(OUT/"oof_pls2_topk.csv",index=False);(OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
