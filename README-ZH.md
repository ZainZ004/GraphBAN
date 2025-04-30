# GraphBAN: 一种用于增强化合物-蛋白质相互作用预测的归纳图方法

<div align="left">


[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/183LGl-eJD-ZUw7lqoKlRdw6rw4Sw73W1?usp=sharing)

</div>


## 简介
本存储库包含 GraphBAN 框架的 PyTorch 实现，详细描述于我们即将发表在《自然通讯》上的论文《GraphBAN: 一种用于增强化合物-蛋白质相互作用预测的归纳图方法》。
在本研究中，我们提出了一种基于分布外的 CPI 预测方法，使用图知识蒸馏（KD）。GraphBAN 利用一个 KD 模块，其中包括一个图分析组件（称为“教师”），和深度双线性注意力网络（BAN）。该框架通过考虑化合物和蛋白质特征的成对局部交互来连接它们的特征。此外，它还结合了领域适配模块，以对齐不同分布间的交互表示，从而提高对未见化合物和蛋白质（称为“学生”）的泛化能力。GraphBAN 在 CPI 的双分图上运行，能够对传导（例如，测试节点在训练期间可见）和归纳（例如，测试节点在训练期间不可见）链接进行预测。
我们的实验使用了五个基准数据集（BioSNAP、BindingDB、KIBA、C.elegans、PDBbind 2016），在传导和归纳设置下进行，结果表明 GraphBAN 优于六种最先进的基线模型，达到了最高的整体性能。

## 框架
![GraphBAN](image/F1.png)


## 系统要求
以下是最低依赖项要求。GraphBAN 支持任何标准计算机和操作系统（Windows/macOS/Linux），只需足够的 RAM 即可运行。无额外的非标准硬件要求。

```
torch
torch-geometric
torchmetrics
dgl
dgllife
numpy
scikit-learn
pandas
prettytable
rdkit
yacs
pyarrow
transformers
fair-esm
```
## 安装指南
克隆此 Github 仓库并设置一个新的 conda 环境。在普通台式计算机上安装通常需要大约 10 分钟。
```
# 克隆 GraphBAN 的源代码
$ git clone https://github.com/HamidHadipour/GraphBAN
$ cd GraphBAN
# 创建一个新的 conda 环境
$ conda create --name graphban python=3.11
$ conda activate graphban

# 安装所需的 Python 依赖项
$ pip install -r requirements.txt
```
## 以下说明用于获取 GraphBAN 论文中提供的结果，或测试模型在您标注的数据集上的性能。
## 如果您有实际案例需要训练和测试模型，请按照 /case_study 目录中提供的说明进行操作。

## 数据
为了训练 GraphBAN，您需要提供一个包含标题行 'SMILES'、'Protein' 和 'Y' 的 CSV 文件。<br>
'Y' 表示二进制值（即 0 和 1），是分类指标。<br>
目前，我们提供了论文中使用的五个数据集（BindingDB、BioSNAP、KIBA、C.elegans 和 PDBbind 2016）的分割数据，分为传导和归纳两种类型，每种类型包含五个不同的种子。

## 数据预处理
如果您需要将自己的数据集分割为传导和归纳类型，请使用 preprocessing/clustering 文件夹中提供的代码。
```
python preprocessing/clustering/inductive_split.py --path_your_dataset --train <path> --val <path> --test <path> --seed <int>
python preprocessing/clustering/transductive_split.py --path_your_dataset --train <path> --val <path> --test <path> --seed <int>
```
## 归纳训练
要训练 GraphBAN，请运行：
```
python run_model.py --train_path <path> --val_path <path> --test_path <path> --seed <int> --mode <[inductive, transductive]> --teacher_path <path>
```
**例如**
```
python run_model.py --train_path Data/sample_data/df_train200.csv --val_path Data/sample_data/df_val.csv --test_path Data/sample_data/df_test.csv --seed 12 --mode inductive --teacher_path Data/sample_data/df_train200_teaqcher_embeddings.parquet
```
结果将保存在名为 **result/** 的目录中，其中包括训练好的 model.pth 和 CSV 文件中的预测分数。

前三个参数是数据分割的路径。<br>
--teacher-path 是包含由模型的教师模块捕获的训练集嵌入的 parquet 文件的路径。<br>
对于本项目中提供的数据分割，所有教师嵌入已提供。<br>
如果需要为您的数据集捕获教师嵌入，请运行以下代码：<br>

```
python teacher_gae.py --train_path <path> --seed <int> --teacher_path <path> --epoch <int>
```
例如

```
python teacher_gae.py --train_path Data/sample_data/df_train200.csv --seed 12 --teacher_path Data/sample_data/test.parquet --epoch 10
```
--teacher_path 应为 parquet 文件的路径。<br>
## 归纳预测
要加载训练好的模型并进行预测，请运行 predict.py 并指定：

--test_path <path> 要预测的数据路径。<br>
--trained_model <path> 训练好的 .pth 文件路径。<br>
--save_dir <path> 要保存预测结果的路径。<br>
```
python predictions/predict.py --test_path <path> --trained_model <path> --save_dir <path>
```
**例如**，
```
python predictions/predict.py --test_path Data/biosnap/inductive/seed12/target_test_biosnap12.csv --trained_model predictions/trained_models/biosnap/inductive/seed12/best_model_epoch_45.pth --save_dir biosnap12_predictions.csv

```
## 传导训练
要以传导模式训练模型，请运行以下代码。在传导模式下，我们没有特征融合和学生模块，仅使用 LLM 提取化合物和蛋白质特征，以及 GAE 在 CPI 的双分图上进行训练。<br>
```
python transductive_mode/train_transductive_mode.py --train_path <path> --val_path <path> --test_path <path> --seed <int> --save_model <path> --metric_path <path> --prediction_path <path> --h_dimension <int> --epochs <int>
```
它接收训练集、验证集和测试集的路径。<br>
还需要设置随机化的种子。<br>
save_model 是保存 torch model.pth 文件的路径。<br>
metric_path 是保存指标的 .csv 文件的地址。<br>
prediction_path 是保存预测概率的 .csv 文件的路径。<br>
dimension 是设置每个化合物和蛋白质节点嵌入的隐藏维度。<br>
epoch 是设置所需的训练轮数。<br>
**例如**，
```
python transductive_mode/train_transductive_mode.py --train_path Data/kiba/transductive/seed12/train_kiba12.csv --val_path Data/kiba/transductive/seed12/val_kiba12.csv --test_path Data/kiba/transductive/seed12/test_kiba12.csv --seed 12 --save_model Data/kiba12_model.pth --metric_path Data/kiba12_metric.csv --prediction_path Data/kiba12_preds.csv --h_dimension 256 --epochs 10
```
## 传导预测
要使用训练好的传导模型，可以运行以下代码，<br>
```
python transductive_mode/predict_transductive.py --test_path <path> --seed <int> --trained_model <path> --metric_path <path> --pred_probs_path <path>
```
**例如**
```
python transductive_mode/predict_transductive.py --test_path Data/kiba/transductive/seed12/test_kiba12.csv --seed 12 --trained_model transductive_mode/trained_models/kiba/kiba_trans_12.pth  --metric_path transductive_mode/kiba12_test_pred_metric.csv --pred_probs_path transductive_mode/kiba12_test_pred.csv
```
## 超参数
如果需要设置超参数，可以查看 **config.py** 和/或 **GraphBAN_DA.yaml**（用于归纳设置）以及 **GraphBAN.yaml**（用于传导设置）。

## 演示
我们通过云端 Jupyter notebook 提供了 GraphBAN 的运行演示，点击 [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/183LGl-eJD-ZUw7lqoKlRdw6rw4Sw73W1?usp=sharing)。注意：提供了一个包含 200 个交互的示例数据集，用于检查训练过程。此外，还提供了一个示例，用于通过 BioSNAP 数据集上的归纳分析检索预测分数，以测试训练好的模型并重现论文中报告的结果。<br>
在 Google Colab 上安装软件包大约需要 5 分钟。<br>
克隆此存储库大约需要 3 分钟。<br>
使用 50 个 epoch 训练示例数据大约需要 8 分钟。<br>
使用训练好的模型进行预测大约需要 3 分钟。<br>
**注意：要在 Google Colab 上运行演示，必须使用支持 GPU 的 Colab 版本。**

## 其他文件夹
**Ablation_study** 是保存消融研究中训练模型的目录。<br>
**case_study** 是保存我们案例研究部分中提供的数据和训练模型的目录。<br>

## 致谢
此实现受 [DrugBAN](https://github.com/peizhenbai/DrugBAN) 的启发并部分基于其早期工作。

## 引用
如果您发现我们的工作对您的研究有用，请引用我们的 [论文]。
```
    @article{Hadipour2025graphban,
      title   = {GraphBAN: An Inductive Graph-Based Approach for Enhanced Prediction of Compound-Protein Interactions},
      author  = {Hamid Hadipour, Yan Yi Li, Yan Sun, Chutong Deng, Leann Lac, Rebecca Davis, Silvia T Cardona, Pingzhao Hu},
      journal = {Nature Communications},
      year    = {2025},
    }
```