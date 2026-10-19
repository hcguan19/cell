# 老师阅读指南：只看这一版

## 先明确是什么模型

当前归档最佳是“周期表格先验 + 阶段视频排序 + 保守Top-K”，不是Qwen2.5-VL直接分类，也不是一个端到端权重文件。
历史1240例汇总：ACC 67.42%，AUC 0.6758，F1 0.6672；表格对照ACC 67.10%，净多判对4例。评估局限见 [EVALUATION_LIMITS_ZH.md](EVALUATION_LIMITS_ZH.md)，不能将该数字解释为独立临床验证结论。

## 推荐阅读：三个文件即可抓住主线

下列路径均相对于 `research/cell_train3/`。

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | evaluate_phasewise_vjepa_nested_topk.py | 表格概率和阶段排序如何接入最终选择，输出哪些指标 |
| 2 | cycle_count_topk_bestvideo_v1/run_best_video_topk.py | 周期阳性数量分布P(K)、置信度门控、Top-K与表格回退 |
| 3 | evaluate_vjepa21_phasewise_fixed_oof.py | 冻结阶段特征、PCA、成对排序与五折候选分数 |

这三个文件是阅读入口，不是独立可运行的三文件最小包。其余20个源码保留原路径和导入关系。

## 想深入某一分支时再看

- 表格专家：`table_group_side_v1/run_multiview.py`。
- 文本小权重混合：`evaluate_pls2_video_topk.py`。这里用Qwen3-Embedding，不是Qwen2.5-VL。
- 视频特征提取：`extract_vjepa21_phasewise_features.py`。
- 全部依赖定位：[SOURCE_INDEX_ZH.md](SOURCE_INDEX_ZH.md)。

不要直接运行其余探索脚本的main来“选择最佳版”。部分文件被保留只因为主流程导入其中的共享函数，部分用于追溯上游。

## 运行前必须知道

- 本仓库不含数据、逐样本预测、缓存或权重。
- 学校交付入口为 `/data0/public/guanhuancong/cell_best_model/`，数据入口在其 `data/`，是软链接；本次GitHub整理没有同步修改服务器代码。
- 脚本保留历史绝对路径。准备独立副本并检查ROOT/OUT等常量，不能直接覆盖最佳结果。
- 有最终两个输入产物时可运行最终入口；缺上游产物时不能宣称能从零精确复现。
- 详细步骤见 [RUNNING_ZH.md](RUNNING_ZH.md)。

## 本次精简范围

原 `research/cell_train3` 有130个Python源码；新版本保留23个有中文注释的核心/上游参考源码，移出107个同目录历史候选，另将论文复现实验与旧图移出默认工作树。
原内容仍在Git提交 `e23082b` 和备份分支中，不是不可恢复删除。23个文件与此前交付 `hcguan19/cell` 的注释版一致，并以manifest校验；算法未修改。
