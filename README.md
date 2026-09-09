# 表格与分阶段视频的周期约束分类

本仓库归档一套**周期表格风险 + 阶段视频排序 + 保守 Top-K**研究代码，并为关键数据流、训练目标、张量维度、模型选择和回退条件添加中文注释。具体业务目标不在公开说明中展开。

这是当前汇报采用的历史最佳组合，**不是 Qwen2.5-VL 直接分类版本**，也不是一个可以单独加载的端到端权重文件。代码从研究服务器于2026-09-09取回，本次没有重新训练，没有修改归档算法。

## 结构

1. **周期表格风险**：四路周期XGBoost专家与历史后处理形成表格锚点，更新字段做保守残差校正，再与冻结Qwen3-Embedding文本特征的PLS2输出按97.5%/2.5%固定混合。
2. **周期内排序**：在已有堆叠分数上，用冻结V-JEPA2.1的阶段特征训练PCA32和成对线性排序头，选择小权重修正。
3. **保守决策**：有序LogisticRegression估计周期阳性数量分布P(K)，按数量后验、熵与排序边界控制Top-K改判，条件不满足时保留表格结果。

已有堆叠分数含历史融合成分，不能称为独立纯视频模型。

## 已归档结果

相同1240例历史外折汇总，非本次重新训练：

| 方法 | ACC | AUC | F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---|
| 表格对照 | 67.10% | 0.6711 | 0.6650 | 427 / 207 / 201 / 405 |
| 阶段排序＋保守Top-K | **67.42%** | **0.6758** | **0.6672** | 431 / 203 / 201 / 405 |

全精度匿名汇总见 [archived_metrics.json](archived_metrics.json)。相对表格净增4例正确分类，不能据此认定泛化收益稳定。

**评估限制必须一起报告**：候选方案经过多轮历史探索；最终选择层复用上游OOF，上游训练来源仍需要完整的嵌套审计；最终排序按测试折无标签分数整批标准化。代码原有`strict`字样不是全链路严格独立测试的证明。详见 [评估边界](docs/EVALUATION_LIMITS_ZH.md)。

## 推荐阅读顺序

| 内容 | 源码 |
|---|---|
| 最终融合入口 | [evaluate_phasewise_vjepa_nested_topk.py](research/cell_train3/evaluate_phasewise_vjepa_nested_topk.py) |
| 数量模型、置信度门控、Top-K | [run_best_video_topk.py](research/cell_train3/cycle_count_topk_bestvideo_v1/run_best_video_topk.py) |
| 阶段特征与排序头 | [evaluate_vjepa21_phasewise_fixed_oof.py](research/cell_train3/evaluate_vjepa21_phasewise_fixed_oof.py) |
| 冻结视觉特征提取 | [extract_vjepa21_phasewise_features.py](research/cell_train3/extract_vjepa21_phasewise_features.py) |
| 表格与文本固定混合 | [evaluate_pls2_video_topk.py](research/cell_train3/evaluate_pls2_video_topk.py) |
| 四路表格专家 | [run_multiview.py](research/cell_train3/table_group_side_v1/run_multiview.py) |

## 运行与复现边界

仓库只包含源码、匿名汇总和文档，**不包含**原始视频、表格、患者/周期标识、逐样本预测、嵌入缓存、预训练权重、凭据及完整历史教师产物。

原脚本保留服务器路径和原有导入关系，以便追溯。不能直接在缺少私有中间文件的新电脑上一键重现67.42%。本仓库也不冒充完整的从零训练或单样本部署包。

- [运行顺序与依赖](docs/RUNNING_ZH.md)
- [核心算法与参数](docs/MODEL_GUIDE_ZH.md)
- [数据接口与隐私](docs/DATA_CONTRACT_ZH.md)
- [源码索引](docs/SOURCE_INDEX_ZH.md)

先运行不访问真实数据的测试：

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

测试仅需要NumPy和pandas，不需要GPU或私有数据。完整历史源码导入还需要对应机器学习依赖。依赖清单不是锁定环境，未声称跨平台完整训练验证通过。

本仓库按所有者确认公开，仅用于方法研究，不作为实际业务决策依据。第三方模型与库需遵守各自许可证；未随本仓库重新分发。
