# 当前最佳模型：清晰版代码

统一运行命令请先阅读 `RUN_COMMANDS_ZH.md`。

完整目录框架见 `项目目录结构.md`。

本目录将历史归档代码整理为五个主要文件，算法目标不变：

```text
表格周期风险 → 阶段视频排序 → 周期阳性数量P(K) → 保守Top-K
```

## 文件结构

| 文件 | 作用 |
|---|---|
| `model_table.py` | 表格输入契约、表格概率和历史表格Top-K对照 |
| `model_video.py` | 阶段分数标准化、Pair-AUC、固定候选和外折选择 |
| `model.py` | 有序数量模型P(K)、策略选择与Top-K解码 |
| `train.py` | 五折训练、视频候选选择、数量模型和策略选择，输出OOF预测 |
| `eval.py` | 独立读取OOF预测，输出Acc/AUC/F1/混淆矩阵 |
| `config.py` | 数据路径、输出路径和固定候选配置 |

## 运行

```bash
cd /data0/guanhuancong/cell/cell_train3/cell_best_model_clean

/data0/guanhuancong/anaconda3/envs/cell/bin/python train.py
/data0/guanhuancong/anaconda3/envs/cell/bin/python eval.py
```

默认输入：

```text
/data0/public/guanhuancong/cell_best_model/data/final_table_input.csv
/data0/public/guanhuancong/cell_best_model/data/final_stage_input.csv
```

默认输出：

```text
outputs/default_run/oof_predictions.csv
outputs/default_run/training_artifact.json
outputs/default_run/metrics.json
```

指定输入输出：

```bash
/data0/guanhuancong/anaconda3/envs/cell/bin/python train.py \
  --table-input /path/to/final_table_input.csv \
  --phase-input /path/to/final_stage_input.csv \
  --output-dir /path/to/output

/data0/guanhuancong/anaconda3/envs/cell/bin/python eval.py \
  --predictions /path/to/output/oof_predictions.csv \
  --output /path/to/output/metrics.json
```

## 边界

`train.py`训练的是最终五折排序/数量/门控层。它消费已有表格和阶段视频OOF，
不从原始Excel重新训练历史四路XGBoost教师，也不从原始帧重新提取V-JEPA特征。
这一区分是为了保持现有1240例归档结果的输入定义，不虚构缺失的早期训练链路。

## 从原始数据开始

表格 XGBoost、冻结 Qwen 文本分支、PNG clip cache、冻结 V-JEPA 视频特征及最终融合的代码位置和分阶段命令见：

```text
docs/FROM_RAW_DATA_ZH.md
```

历史核心源码位于 `research/cell_train3/`，V-JEPA2.1 模型实现源码位于 `third_party/vjepa2/`。
