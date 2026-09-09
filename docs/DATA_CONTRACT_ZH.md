# 输入输出与隐私约定

## 最终入口需要的两个私有文件

相对于研究根目录：

1. `qwen_pls2_video_topk_v1/oof_pls2_topk.csv`
2. `vjepa21_vitb_video_v3_phasewise/oof_fixed_screen.csv`

表格文件需要：`sample_id, group_id, fold, label, p_classify, video_rank, pgt_type, pred_pls2_topk, prob_pls2_topk`。阶段文件需要四个连接键`sample_id, group_id, fold, label`以及候选分数列`phase_global_all, phase_global_early, within_dynamics_all, spatial_early`。筛选入口还生成`within_dynamics_early`供历史分析。

运行前检查：

- 每行是一个唯一样本，sample_id无重复；保留字符串ID前导零。
- 同一group_id只属于一个fold，目标外折为0至4。
- 标签为0/1且评价折有两类，p_classify有限且在[0,1]。
- 同周期p_classify一致，或明确验证max_changes的行为是否适合非一致输入。
- 两文件匹配覆盖率、标签及周期一致性通过检查，不能仅靠inner merge静默删除。
- 每一列OOF的训练/验证来源有记录，不只检查最终CSV折号。

## 特征缓存

6个分片各有`features_i.npy`、`rows_i.npy`、`phase_valid_i.npy`。全局行号必须唯一并完整覆盖META；缓存shape为(N,4,8,9,768)。缺失阶段应为零特征并带mask，不应伪造图像。

## 输出

最终`nested_topk/oof.csv`保留上游数据并新增`phasewise_fused_rank, phasewise_topk_pred, phasewise_topk_prob`。`summary.json`包含指标、选中权重及门控策略。**逐样本文件必须留在私有存储**。

仓库中的`archived_metrics.json`仅含匿名汇总，不含这些输入输出明细。测试目录里的小数据全部人工构造，与真实样本无关。
