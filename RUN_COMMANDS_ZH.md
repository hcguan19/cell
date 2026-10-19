# 当前最佳模型统一运行说明

## 1. 项目位置

```text
/data0/public/guanhuancong/cell_best_model
```

Python 环境：

```text
/data0/guanhuancong/anaconda3/envs/cell/bin/python
```

进入项目并设置变量：

```bash
cd /data0/public/guanhuancong/cell_best_model
export ROOT=/data0/public/guanhuancong/cell_best_model
export PY=/data0/guanhuancong/anaconda3/envs/cell/bin/python
```

## 2. 最简单的运行方式（推荐）

该方式使用已经生成好的表格 OOF 和视频阶段 OOF，只重新训练最终的五折视频候选选择、周期阳性数量模型及保守 Top-K 决策层。

```bash
cd /data0/public/guanhuancong/cell_best_model
/data0/guanhuancong/anaconda3/envs/cell/bin/python train.py
/data0/guanhuancong/anaconda3/envs/cell/bin/python eval.py
```

默认输入：

```text
data/final_table_input.csv
data/final_stage_input.csv
```

默认输出：

```text
outputs/default_run/oof_predictions.csv
outputs/default_run/training_artifact.json
outputs/default_run/metrics.json
```

其中：

- `oof_predictions.csv`：每个样本的 OOF 概率和预测标签。
- `training_artifact.json`：五折候选模型、数量模型和门控策略。
- `metrics.json`：Acc、BAcc、F1、AUC 和混淆矩阵。

如需自行指定输入和输出：

```bash
"$PY" train.py \
  --table-input /path/to/final_table_input.csv \
  --phase-input /path/to/final_stage_input.csv \
  --output-dir /path/to/output

"$PY" eval.py \
  --predictions /path/to/output/oof_predictions.csv \
  --output /path/to/output/metrics.json
```

## 3. 从临床 Excel 重新训练表格 XGBoost

默认合并临床表格：

```text
/data0/guanhuancong/cell/merged_two_datasets_clinical_labels.xlsx
```

数据处理规则包括：

- 使用最终分类生成二分类标签；
- 删除标签缺失样本；
- 删除“仅嵌合”样本；
- 清洗连续特征和离散特征；
- 按患者周期生成严格五折划分。

运行命令：

```bash
cd "$ROOT/research/cell_train3/table_group_side_v1"
"$PY" prepare_data.py
"$PY" run_multiview.py
```

主要代码：

```text
research/cell_train3/table_group_side_v1/prepare_data.py
research/cell_train3/table_group_side_v1/run_multiview.py
```

`run_multiview.py` 会训练 doctor-core、old20-like 和 full-numeric 三类周期级 XGBoost，并训练组合门控。

## 4. 从阶段 PNG 重新生成视频 clip cache

当前最终视频流程读取的是已经按时间阶段整理的 PNG，而不是直接读取 AVI：

```text
0–100 h
109–115 h
133–139 h
157–163 h
```

每个阶段构造 short/long 两个 16 帧视图，最终 V-JEPA 阶段分支优先使用 long view，缺失时回退到 short view。

生成六个 clip cache 分片：

```bash
cd "$ROOT/research/cell_train3/raw_video_sibling_ssl_v1"

for shard in 0 1 2 3 4 5; do
  "$PY" cache_clips.py --shard "$shard" --shards 6
done
```

相关代码：

```text
research/cell_train3/raw_video_sibling_ssl_v1/common.py
research/cell_train3/raw_video_sibling_ssl_v1/cache_clips.py
```

注意：历史 AVI 到逐帧 PNG 的导出程序不属于最终模型归档。当前可复现起点是已经整理好的阶段 PNG。

## 5. 提取冻结 V-JEPA2.1 四阶段视频特征

V-JEPA2.1 模型源码：

```text
third_party/vjepa2
```

服务器预训练权重：

```text
/data0/guanhuancong/cell/model/vjepa2_1_vitb_dist_vitG_384.pt
```

六张 GPU 并行提取：

```bash
cd "$ROOT/research/cell_train3"

for shard in 0 1 2 3 4 5; do
  "$PY" extract_vjepa21_phasewise_features.py \
    --shard "$shard" \
    --num-shards 6 \
    --gpu "$shard" \
    > "vjepa_shard_${shard}.log" 2>&1 &
done
wait
```

然后训练/筛选视频阶段排序器：

```bash
"$PY" evaluate_vjepa21_phasewise_fixed_oof.py
```

相关代码：

```text
research/cell_train3/extract_vjepa21_features.py
research/cell_train3/extract_vjepa21_phasewise_features.py
research/cell_train3/evaluate_vjepa21_phasewise_fixed_oof.py
```

这里的 V-JEPA 主干被冻结，不进行全参数微调；训练的是下游 PCA 和周期内 pairwise Logistic Regression 排序器。

## 6. 生成冻结 Qwen 文本特征并训练下游头

服务器 Qwen 权重：

```text
/data0/guanhuancong/model/hf_models/Qwen__Qwen3-Embedding-0.6B
```

运行命令：

```bash
cd "$ROOT/research/cell_train3"
"$PY" evaluate_qwen_clinical_text_cycle.py
"$PY" evaluate_qwen_text_heads_v2.py
```

相关代码：

```text
research/cell_train3/evaluate_qwen_clinical_text_cycle.py
research/cell_train3/evaluate_qwen_text_heads_v2.py
```

Qwen3-Embedding-0.6B 同样被冻结，不进行大模型全参数训练。Qwen 负责将诊断、染色体和治疗文字编码为 1024 维向量，下游训练 PCA、PLS、Ridge 和 Logistic Regression；最终使用 PLS2 小权重修正表格概率。

## 7. 最终多模态融合

表格和视频上游 OOF 生成完成后：

```bash
cd "$ROOT"
"$PY" train.py
"$PY" eval.py
```

融合方式不是神经网络特征拼接，而是决策级融合：

1. 表格分支输出每个周期的基础风险概率；
2. 视频分支对同一周期内的胚胎进行排序；
3. 数量模型预测该周期阳性胚胎数量的分布 `P(K)`；
4. 满足置信度、边际、熵和排序间隔条件时选择 Top-K；
5. 条件不足时回退到表格预测。

## 8. 一套完整的执行顺序

```bash
cd /data0/public/guanhuancong/cell_best_model
export ROOT=/data0/public/guanhuancong/cell_best_model
export PY=/data0/guanhuancong/anaconda3/envs/cell/bin/python

# 1. Excel 清洗、标签处理和五折划分
cd "$ROOT/research/cell_train3/table_group_side_v1"
"$PY" prepare_data.py

# 2. 表格 XGBoost
"$PY" run_multiview.py

# 3. PNG -> 16帧 clip cache
cd "$ROOT/research/cell_train3/raw_video_sibling_ssl_v1"
for shard in 0 1 2 3 4 5; do
  "$PY" cache_clips.py --shard "$shard" --shards 6
done

# 4. clip cache -> 冻结 V-JEPA 四阶段特征
cd "$ROOT/research/cell_train3"
for shard in 0 1 2 3 4 5; do
  "$PY" extract_vjepa21_phasewise_features.py \
    --shard "$shard" --num-shards 6 --gpu "$shard" \
    > "vjepa_shard_${shard}.log" 2>&1 &
done
wait

# 5. 视频排序器
"$PY" evaluate_vjepa21_phasewise_fixed_oof.py

# 6. 冻结 Qwen embedding 与下游头
"$PY" evaluate_qwen_clinical_text_cycle.py
"$PY" evaluate_qwen_text_heads_v2.py

# 7. 最终融合与评价
cd "$ROOT"
"$PY" train.py
"$PY" eval.py
```

## 9. 重要限制

1. 最简单运行方式已经验证可运行，并能复现归档最终预测。
2. 完整上游代码保留了实验服务器上的部分绝对路径；换服务器时需要修改数据、模型权重和输出路径。
3. Qwen 和 V-JEPA 权重没有复制进项目目录，两者合计约 2.8GB。
4. 部分早期表格教师、更新字段校正和 Qwen 输入存在历史中间产物依赖。因此完整上游流程不是完全无依赖的一键训练脚本。
5. 当前报告结果属于五折 OOF 结果，不是独立外部测试集结果，也不能直接解释为临床前瞻性能。

## 10. 推荐实际使用方式

只需要检查或复现当前最佳结果时，运行：

```bash
cd /data0/public/guanhuancong/cell_best_model
/data0/guanhuancong/anaconda3/envs/cell/bin/python train.py
/data0/guanhuancong/anaconda3/envs/cell/bin/python eval.py
```

需要研究或修改上游模型时，再按照第 3–8 节逐阶段运行，并为新实验使用独立输出目录，避免覆盖当前最佳归档。
