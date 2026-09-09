# 23个归档源码的定位

## 当前入口与共享算法

- `evaluate_phasewise_vjepa_nested_topk.py`：最终阶段分数选择和Top-K汇总。
- `cycle_count_topk_bestvideo_v1/run_best_video_topk.py`：P(K)、内层交叉拟合、门控及解码。
- `cycle_count_topk_v1/run_cycle_count_topk.py`：周期聚合与bootstrap共享函数，其他回归器是早期对照。
- `evaluate_updated_table_video_topk.py`：共用数量解码评估器与历史更新对照。
- `evaluate_pls2_video_topk.py`：2.5%文本PLS2表格组合。

## 视频特征与排序

- `extract_vjepa21_features.py`：外部编码器与像素分片读取。
- `extract_vjepa21_phasewise_features.py`：逐阶段特征缓存。
- `evaluate_vjepa21_nested_rank_topk.py`：成对样本与fit_predict共享函数，其main是早期实验。
- `evaluate_vjepa21_phasewise_fixed_oof.py`：阶段候选固定五折排序。
- `raw_video_sibling_ssl_v1/common.py`：历史像素/X3D工具，当前入口复用pair_auc。
- `dual_video_stacking_v1/run_dual_video_stacking.py`：既有分数堆叠。
- `cycle_count_topk_bestvideo_v1/prepare_best_video_input.py`：历史视频键对齐。

## 表格与文本上游参考

- `table_group_side_v1/prepare_data.py`：历史清洗与分组划分。
- `table_group_side_v1/run_cv.py`：早期周期XGBoost对照。
- `table_group_side_v1/run_continue_v2.py`：继续探索的表格候选。
- `table_group_side_v1/run_doctor_core_fast.py`：核心字段专家。
- `table_group_side_v1/run_multiview.py`：四视图专家与门控。
- `deployable_table_distill_v1/train_distill.py`：历史规则及蒸馏参考。
- `updated_table_tjf_v1/audit_updated.py`：更新字段审计。
- `updated_table_tjf_v1/run_update_variants.py`：更新字段与旧教师对比。
- `updated_table_tjf_v1/run_locked_endocrine_residual.py`：更新字段小残差。
- `evaluate_qwen_clinical_text_cycle.py`：周期文本Embedding。
- `evaluate_qwen_text_heads_v2.py`：PLS/Ridge/LogReg探索与PLS2输出。

`source_manifest.json`记录服务器取回时的原文件SHA256；`annotated_manifest.json`记录加注释后的文件SHA256和AST摘要。两种源码字节不同，但注释版的AST应与原始AST完全一致。
