## 案例研究部分包括论文中案例研究分析所用的所有代码和数据集。
## 由于训练模型文件较大，如有需要，请查看以下链接中每个使用不同训练集训练的模型文件。如果下载文件时遇到任何问题，请通过电子邮件联系 (hamid.hadipour@umanitoba.ca)<br>

-[BioSNAP 训练模型](https://umanitoba-my.sharepoint.com/:u:/g/personal/hamid_hadipour_umanitoba_ca1/EUm47tS6nlNEjIpQcJjDCdoBb8nh2TnTqc7VbFGIe2FMpw?e=UVvPcM)<br>
-[BindingDB 训练模型](https://umanitoba-my.sharepoint.com/:u:/g/personal/hamid_hadipour_umanitoba_ca1/EXHP3lyMCE5Lpt8xV8lAyFYBqxI5PU3JUWwO7k3X5y6KgQ?e=OB3eEO)<br>
-[KIBA 训练模型](https://umanitoba-my.sharepoint.com/:u:/g/personal/hamid_hadipour_umanitoba_ca1/EbWxe-y2PWpLpxVxVrDIxUYBdtBAvz_OSbqHE4-GcmH50w?e=2DkqaP)<br>
## 如果您有自己的数据集需要训练和预测，请运行以下命令。

```
python run_model.py --train_path <path> --val_path <path> --test_path <path> --seed <int> --mode <inductive> --teacher_path <path> --result_path <path>
```
**例如**<br>
```
python case_study/run_model.py --train_path case_study/biosnap_train_data/source_train_biosnap12.csv --val_path case_study/zinc_data/split_zinc_1.csv --test_path case_study/zinc_data/split_zinc_1.csv --seed 12 --mode inductive --teacher_path case_study/biosnap_train_data/biosnap12_inductive_teacher_emb.parquet --result_path case_study/result_biosnap12_zinc1
```
-result_path 将在您的路径目录中保存 50 个 epoch 的模型。<br>

**注意**<br>
要获取 teacher embedding，您需要进入 /inductive_mode 目录并使用以下参数运行 teacher_gae.py 文件。<br>

```
python inductive_mode/teacher_gae.py --train_path <path> --seed <int> --teacher_path <path> --epoch <int>
```
teacher_path 是您希望以 .parquet 格式保存 teacher embedding 的路径。<br>
**例如**<br>
```
python inductive_mode/teacher_gae.py --train_path Data/sample_data/df_train200.csv --seed 12 --teacher_path Data/sample_data/test.parquet --epoch 10
```
**预测**<br>
要使用训练好的模型进行预测，您可以运行以下代码。<br>
```
python predict.py --test_path <path> --folder_path <path> --save_dir <path>
```
-folder_path 是包含训练模型的文件夹路径。<br>
**注意**<br>
默认情况下，predict.py 代码使用的是在 30-50 个 epoch 训练的模型，而不是所有 50 个 epoch 的模型。原因是实验表明，模型在第 30 个 epoch 后预测结果趋于稳定。您可以根据自己的情况更改这些设置。<br>
超参数的详细信息可以在 GraphBAN_DA.yaml 和 configs.py 文件中更改。此外，预测中要考虑的模型数量可以在 predict.py 文件的第 115 行更改。<br>

**例如**<br>
```
python case_study/predict.py --test_path case_study/zinc_data/split_zinc_1.csv --folder_path case_study/result_biosnap12_zinc1 --save_dir case_study/test_zinc_new1_preds.csv
```
## 要重新运行论文中提供的结果所用的代码，请按照以下步骤操作。

-ZINC-Pin1 的完整数据集位于根目录 ZINC-Pin1.csv 中。<br>
-训练数据集的 25 个分割文件位于 /zinc_data 目录中。<br>
-用于训练模型的三个训练集位于以 biosnap_train_data、bindingdb_train_data 和 kiba_train_data 命名的三个目录中。<br>
-/predictions 文件夹包含了 GraphBAN 基于每个训练集捕获的大约 250k 次交互的概率。<br>

## 要使用每个训练集运行模型，您可以根据需要运行以下命令之一
```
python BioSNAP_run.py
```

```
python BindingDB_run.py
```

```
python KIBA_run.py
```

**要根据使用三个数据集之一训练的模型获取预测值，您可以相应地运行以下预测代码**<br>
**在运行之前，您需要从上面提供的链接下载训练模型。**
```
python BioSNAP_predict.py
```
```
python BindingDB_predict.py
```
```
python KIBA_predict.py
```