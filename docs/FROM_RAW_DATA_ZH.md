# 从原始 Excel / PNG / AVI 到最终模型

## 先说明“训练”的边界

当前最佳方法不是把 XGBoost、Qwen 和 V-JEPA 放进同一个网络端到端训练：

1. **XGBoost 表格分支会训练**：从合并后的临床 Excel 清洗、构造五折，再训练多视图 XGBoost。
2. **Qwen3-Embedding-0.6B 不微调**：冻结 Qwen，仅生成周期级文本向量；随后训练 PCA/PLS/Ridge/LogisticRegression 下游头。
3. **V-JEPA2.1 ViT-B 不微调**：冻结 V-JEPA，仅从每个胚胎四个阶段的 16 帧 clip 提取 token；随后训练 PCA 与周期内 pairwise LogisticRegression 排序器。
4. 根目录的 `train.py` 再训练数量模型、选择策略并进行保守 Top-K 决策融合。

因此，“从原始数据重跑”是多阶段流水线，不是一条 `train.py` 完成所有步骤。

## 代码位置

### 表格与 XGBoost

- `research/cell_train3/table_group_side_v1/prepare_data.py`：读取合并临床 Excel、删除缺失标签及“仅嵌合”、清洗字段并生成严格五折。
- `research/cell_train3/table_group_side_v1/run_multiview.py`：训练 doctor-core、old20-like、full-numeric 三类周期级 XGBoost 及门控。
- `research/cell_train3/updated_table_tjf_v1/`：更新字段审计及小残差修正。

默认原始表格：

```text
/data0/guanhuancong/cell/merged_two_datasets_clinical_labels.xlsx
```

### Qwen 文本分支

- `research/cell_train3/evaluate_qwen_clinical_text_cycle.py`：冻结 Qwen 生成文本 embedding，训练 PCA + Ridge/LogReg。
- `research/cell_train3/evaluate_qwen_text_heads_v2.py`：在缓存 embedding 上训练 PLS/Ridge/LogReg，并产生 PLS2 概率。

服务器模型权重：

```text
/data0/guanhuancong/model/hf_models/Qwen__Qwen3-Embedding-0.6B
```

### PNG、clip cache 与 V-JEPA 视频分支

- `research/cell_train3/raw_video_sibling_ssl_v1/common.py`：四阶段帧路径、16 帧抽取和图像处理。
- `research/cell_train3/raw_video_sibling_ssl_v1/cache_clips.py`：从 PNG 阶段帧生成六个 uint8 clip cache 分片。
- `research/cell_train3/extract_vjepa21_features.py`：V-JEPA 编码器加载及共享函数。
- `research/cell_train3/extract_vjepa21_phasewise_features.py`：分别提取四阶段 V-JEPA token。
- `research/cell_train3/evaluate_vjepa21_phasewise_fixed_oof.py`：训练/筛选阶段排序候选并输出 OOF。
- `third_party/vjepa2/`：V-JEPA2.1 模型实现源码。

服务器预训练权重：

```text
/data0/guanhuancong/cell/model/vjepa2_1_vitb_dist_vitG_384.pt
```

当前视频上游从已经按时间阶段整理的 PNG 帧读取。AVI 到逐帧 PNG 的历史导出没有进入最终算法归档；现有 `data/video_frames_old` 与 `data/video_frames_new` 是实际使用的 PNG 数据入口。不能把 V-JEPA 特征提取脚本表述为直接读取 AVI。

### 最终融合

- `train.py`：读取表格 OOF 和阶段视频 OOF，训练最终五折数量模型/策略层。
- `eval.py`：计算 Acc、BAcc、F1、AUC 与混淆矩阵。

## 在当前服务器上的分阶段命令

Python 环境：

```bash
PY=/data0/guanhuancong/anaconda3/envs/cell/bin/python
ROOT=/data0/public/guanhuancong/cell_best_model
```

### 1. 清洗 Excel 并构造五折

```bash
cd "$ROOT/research/cell_train3/table_group_side_v1"
"$PY" prepare_data.py
```

### 2. 训练表格多视图 XGBoost

```bash
cd "$ROOT/research/cell_train3/table_group_side_v1"
"$PY" run_multiview.py
```

### 3. 生成四阶段 16 帧 clip cache

此步骤读取现有阶段 PNG、分组清单和 ROI 文件。脚本中的 `CV_ROOT`、`ROI_PATH`、`CACHE_ROOT` 仍保留原服务器数据路径。

```bash
cd "$ROOT/research/cell_train3/raw_video_sibling_ssl_v1"
for shard in 0 1 2 3 4 5; do
  "$PY" cache_clips.py --shard "$shard" --shards 6
done
```

### 4. 提取冻结 V-JEPA 四阶段特征

先检查 `extract_vjepa21_features.py` 中的 `MODEL_REPO`、`CKPT`、`CACHE` 和 `META`。公共目录已经包含 V-JEPA 源码，但历史脚本默认仍指向原模型源码路径。

```bash
cd "$ROOT/research/cell_train3"
for shard in 0 1 2 3 4 5; do
  "$PY" extract_vjepa21_phasewise_features.py \
    --shard "$shard" --num-shards 6 --gpu "$shard"
done
```

### 5. 训练视频阶段排序器

```bash
cd "$ROOT/research/cell_train3"
"$PY" evaluate_vjepa21_phasewise_fixed_oof.py
```

### 6. 生成冻结 Qwen embedding 并训练下游头

这些脚本依赖历史表格教师的 OOF 锚点，不能跳过表格教师链路直接运行。

```bash
cd "$ROOT/research/cell_train3"
"$PY" evaluate_qwen_clinical_text_cycle.py
"$PY" evaluate_qwen_text_heads_v2.py
```

### 7. 最终融合训练与评估

当表格和视频上游 OOF 已经生成后：

```bash
cd "$ROOT"
"$PY" train.py
"$PY" eval.py
```

## 复现限制

- 归档保留了最佳模型所需的核心代码，但部分历史表格教师产物和更新字段校正存在前后依赖；它们不是一个全新、无中间产物环境中的单命令流水线。
- 当前 `data/` 中部分项目是指向本服务器私有数据位置的符号链接。复制到另一台机器时必须重新配置路径。
- Qwen 和 V-JEPA 权重没有重复复制到本目录；权重合计约 2.8GB，且需遵守各自许可。
- 最终结果是历史 OOF 评估，不等价于独立外部测试集或临床前瞻验证。
