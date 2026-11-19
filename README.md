# 医学预训练语言模型水印：课程简化复现

本项目按照 Kong 等人在 ACM Multimedia 2025 发表的论文，对 BioBERT 和 NCBI-Disease 上的 NER 水印方法做简化复现，比较有水印、无水印模型。论文 DOI：https://doi.org/10.1145/3746027.3755548 。本代码为独立简化实现，不是作者原始代码；作者仓库：https://github.com/edu-yinzhaoxia/Med-PLMW 。

## 内容

- `med_watermark_simple.py`：训练和评估脚本。
- `results_500_1epoch.json`：500 条训练样本、1 轮的实际结果。
- `results_2000_3epoch.json`：2,000 条训练样本、3 轮的实际结果。
- `requirements.txt`：已验证的主要依赖版本。

## 实验与结果

使用 BioBERT、NCBI-Disease，seed 42、batch size 4、学习率 2e-5、最大序列长度 128。正常 NER 性能使用 seqeval 实体级 F1。两组各训练一次。

| 配置               | 模型  | F1     | 触发词被识别为疾病实体   |
| ---------------- | --- | ------:| -------------:|
| 500 条、1 epoch    | 无水印 | 57.67% | 2/98，2.04%    |
| 500 条、1 epoch    | 有水印 | 57.67% | 7/98，7.14%    |
| 2,000 条、3 epochs | 无水印 | 85.17% | 4/100，4.00%   |
| 2,000 条、3 epochs | 有水印 | 85.17% | 19/100，19.00% |

论文的 BioBERT/NCBI 有水印 F1 87.51%、WACC 82.19%；本实验验证句经两个模型筛选、仅两组词对、句首插入，19% 不是原论文 WACC，不可视为同条件比较。数据集测试句数量本次 941、论文表 1 为 942，原因尚未核实。

## 环境

实测 Windows、Python 3.12、NVIDIA RTX 4060 Laptop GPU（8 GB）、torch 2.6.0+cu124、transformers 4.46.3、datasets 2.21.0、accelerate 1.1.1、seqeval 1.2.2、numpy 1.26.4、pyarrow 17.0.0。运行需要 CUDA GPU、可联网获取模型及数据；可使用指定版本自行创建隔离环境。模型和数据缓存不包含在项目中。数据集使用 Hugging Face 的自定义加载代码（固定 revision `ee07f76`，`trust_remote_code=True`）；请先审阅数据集代码并理解数据使用条款。

## 运行

在已准备上述依赖的 Windows 虚拟环境中，在包含脚本的目录执行：

```bat
python med_watermark_simple.py --train-samples 2000 --epochs 3
```

脚本输出到 `med_watermark_results/results.json`，重复运行会覆盖该文件；请事先备份。少量样本试运行：`python med_watermark_simple.py --train-samples 500 --epochs 1`。实测两组完整训练各约 95 秒（不含下载和全部评估），不同硬件和网络的耗时会不同。

##
