# 运行顺序与环境

## 先区分复现范围

本仓库保存23个原始核心/上游参考源码，补充注释，不把未完整追溯的历史链路改写成一个声称可以从零复现的新算法。原始硬编码ROOT、模型路径和部分上游参考产物依赖保留。移机需要逐一配置。

代码会写入历史相对结果路径，有些入口会覆盖同名结果。**请在独立实验目录准备私有输入并修改ROOT/OUT等常量，不要直接覆盖最佳归档。** 仅改变shell当前目录不能覆盖这些绝对路径。

## 环境

建议在单独Linux虚拟环境使用Python 3.10或更高。`requirements.txt`列出直接依赖，不固定CUDA/PyTorch组合，也不是原服务器完整环境锁。

V-JEPA2.1模型源码和权重分别由`MODEL_REPO`及`CKPT`指定，需从权利人渠道自行准备。Qwen3-Embedding路径由文本入口的`MODEL`指定。完整导入历史`common.py`还需要PyTorchVideo、OpenCV等，即使最后只使用pair_auc。

## 有历史中间产物时

在副本中配置好数据根目录后，按以下顺序执行。命令不会自动下载或生成缺失的教师数据。

```bash
cd research/cell_train3
# 先检查docs/DATA_CONTRACT_ZH.md中的ID、折号和缓存行号条件。

# 已有原始像素缓存及V-JEPA2.1权重时，生成6个分片。
for shard in 0 1 2 3 4 5; do
  python extract_vjepa21_phasewise_features.py --shard "$shard" --num-shards 6 --gpu 0
done

# 用冻结特征训练阶段排序候选。
python evaluate_vjepa21_phasewise_fixed_oof.py

# 表格PLS2输入已生成时，运行最终选择与解码。
python evaluate_phasewise_vjepa_nested_topk.py
```

如果最终入口的两个输入文件已经存在，只需最后一条，但仍须先检查上游来源。最终数量/策略筛选在CPU上运行，GPU主要用于冻结特征抽取。

## 表格/已有排序产物缺失时

需要恢复以下私有上游，而不是用随意训练的新概率替代历史列：

- 锁定表格教师与旧规则来源、清洗后的更新字段和原train/val/test划分。
- `run_locked_endocrine_residual.py`生成的更新表格概率。
- `evaluate_qwen_clinical_text_cycle.py`的周期Embedding缓存。
- `evaluate_qwen_text_heads_v2.py`的`pls_2`输出。
- 已有堆叠排序文件与`prepare_best_video_input.py`的对齐结果。
- `evaluate_updated_table_video_topk.py`产生的历史比较文件。
- `evaluate_pls2_video_topk.py`产生最终表格/旧Top-K输入。

本包不含所有早期教师及X3D/VideoMAE训练快照，因此缺少上述产物时，不能保证从零精确复现归档数字。重训新的上游应单独命名并重新报告成绩。

## 本地无数据验证

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

测试用AST抽取独立数值函数，避免导入历史脚本时读取私有数据或加载GPU模型。它不替代真实环境的完整集成测试。
