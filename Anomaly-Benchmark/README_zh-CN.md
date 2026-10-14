# 地震波形异常数据集

本仓库是精简后的纯数据开源包，仅保留：异常注入代码、配置文件、原始数据集、自然异常标签、注入异常后的数据集与标签、两张可视化图、精确划分索引以及必要说明文件。模型代码、模型权重、性能表格、检测结果图和部署文件均不包含在内。

## 目录

```text
data/raw/                  原始地震波形
data/natural_labels/       两位专家最终共识标签及原始标签示意图
data/augmented/            注入后波形、标签、事件目录及注入示意图
data/splits/               每个划分的完整精确索引
injection/                 基准异常注入与训练增强代码及配置
```

## 原始数据与人工标注

- 数据形状：60,000 个采样点 × 200 道。
- 采样率：100 Hz，每道时长 600 s。
- 原始数据 SHA-256：`bd807a59ff60d307b2a94801ab15d9f97f7ff9d40ceceb184e6feb897998fdc7`。
- 最终自然异常：35 个区间、375 个标注点。

最终自然异常标签由两位领域专家人工复核形成。代码仅用于辅助提出候选区间；自动候选不直接作为真实标签。专家综合目标道、相邻道、波形形态和时距相干性进行判断，相干的真实地震信号保留为背景。本公开包提供最终共识标签，不公开专家身份。

![原始数据自然异常最终标注](data/natural_labels/final_label_overview.png)

## 注入异常

注入程序使用固定随机种子 `20260913`，包含阶跃、零值掉线、平直、漂移、饱和截断、增益突变、高频抖动和宽带强噪声八类合成异常。冻结目录完整保留论文版本：165 个合成事件、80,614 个合成异常点、80,989 个合并异常点，其中精确测试集包含 24,645 个合成异常点。合成异常不会与人工自然异常或精确缓冲区重叠。

配置中的 `split_ranges` 是训练、校准和测试成员关系的唯一依据，与实验主程序完全一致。`frozen_injection_sampling_ranges` 只用于保留论文冻结版本原有的随机数映射，不是另一套数据划分。验证程序会确认全部 165 个实际事件均完整位于精确实验划分内。

![自然异常与注入异常总览](data/augmented/all_anomaly_overview.png)

## 训练阶段数据增强

`injection/training_augmentation.py` 是从实际实验主程序中独立提取出的无模型增强模块，对应参数完整保存在 `injection/training_augmentation_config.json`。代码原样保留了两层处理及先后顺序：

1. 以 `[道, 时间]` 为输入的背景增强：随机平移 `[-4, 4]` 个样点（边界值复制）、幅值缩放 `[0.90, 1.10]`、零均值高斯噪声（标准差从 `[0.01, 0.05]` 均匀采样），最后执行 4 轮跨道稳健背景清理。
2. 以 `[批次, 时间]` 为输入的在线自监督增强：先叠加标准差 `0.012` 的基础高斯噪声，再以 `0.96` 的概率在每个窗口生成 2–7 个程序化异常。异常类型包括脉冲、突发、阶跃、平直、漂移、掉线、高频抖动、饱和截断、增益突变和强噪声。

训练增强使用随机种子 `20260520`，与冻结数据集生成所用的 `injection/anomaly_config.json`（种子 `20260913`）严格分开。异常事件选择使用 Python `random` 随机流，随机噪声张量使用 PyTorch 随机流；精确复现时必须同时设定二者。

## 精确划分

索引为零起点，区间使用左闭右开形式：

| 划分 | 区间 | 样本数 |
|---|---:|---:|
| 训练集 | `[0, 32000)` | 32,000 |
| 缓冲区 1 | `[32000, 36096)` | 4,096 |
| 校准集 | `[36096, 44192)` | 8,096 |
| 缓冲区 2 | `[44192, 48288)` | 4,096 |
| 测试集 | `[48288, 60000)` | 11,712 |

完整整数索引保存在 `data/splits/split_indices.npz`，可读边界表保存在 `data/splits/split_manifest.csv`。缓冲区不得用于训练、阈值选择或最终评估。

## 验证与重新生成

```bash
python -m venv .venv
python -m pip install -r requirements.txt

python injection/validate_release.py \
  --config injection/anomaly_config.json \
  --results-dir data/augmented

python injection/augment_seismic_anomalies.py \
  --config injection/anomaly_config.json \
  --output-dir reproduced-augmented

# 可选：验证从主程序提取的训练增强逻辑（需要 PyTorch）
python -m pip install -r requirements-training-augmentation.txt
python injection/training_augmentation.py \
  --config injection/training_augmentation_config.json \
  --self-test
```

重新生成时请写入新目录，不要覆盖冻结的 `data/augmented/`。
