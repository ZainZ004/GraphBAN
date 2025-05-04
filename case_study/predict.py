import argparse
import os
import warnings
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import esm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import get_cfg_defaults
from dataloader import DTIDataset, DTIDataset2, MultiDataLoader
from domain_adaptator import Discriminator
from models import GraphBAN
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer_pred import Trainer
from transformers import AutoTokenizer, RobertaModel
from utils import graph_collate_func, mkdir, set_seed

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("graphban_predict.log")
    ]
)
logger = logging.getLogger("GraphBAN-Predict")

def parse_paths(input_string):
    # Split input by comma or space
    return input_string.replace(",", " ").split()

# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Load train, val, test datasets and additional parameters."
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to the test dataset."
    )
    parser.add_argument(
        "--folder_path", type=str, help="Path to the folder containing .pth files"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="path to save the csv of prediction probabilities",
    )
    parser.add_argument(
        "--start_epoch", 
        type=int, 
        default=30, 
        help="Starting epoch for model selection"
    )
    parser.add_argument(
        "--end_epoch", 
        type=int, 
        default=50, 
        help="Ending epoch for model selection"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None, 
        help="Override the batch size in config"
    )
    parser.add_argument(
        "--config_file", 
        type=str, 
        default="GraphBAN_DA.yaml", 
        help="Path to config file"
    )
    return parser.parse_args()

# 获取蛋白质特征
def get_protein_feature(p_list, batch_converter, esm_model, device):
    try:
        start_time = time.time()
        data_tmp = [(f"protein{i}", p[:1022]) for i, p in enumerate(p_list)]
        dictionary = {}
        
        # 批量处理以提高效率
        for i in range((len(data_tmp) + 4) // 5):
            data_part = data_tmp[i * 5 : (i + 1) * 5]
            if not data_part:  # 检查data_part是否为空
                continue
                
            try:
                _, _, batch_tokens = batch_converter(data_part)
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                
                for j, (_, seq) in enumerate(data_part):
                    emb_rep = token_representations[j, 1 : len(seq) + 1].mean(0).cpu().numpy()
                    dictionary[seq] = emb_rep
            except Exception as e:
                logger.error(f"Error processing protein batch {i}: {str(e)}")
                continue
                
        logger.info(f"Protein feature extraction completed in {time.time() - start_time:.2f} seconds")
        return pd.DataFrame(dictionary.items(), columns=["Protein", "esm"])
    except Exception as e:
        logger.error(f"Failed to extract protein features: {str(e)}")
        raise

# 获取化合物嵌入
def get_embeddings(df, tokenizer, model_chem, device, batch_size=32):
    try:
        start_time = time.time()
        emb_list = []
        smiles_list = df["SMILES"].tolist()
        total = len(smiles_list)
        
        # 按批次处理SMILES以优化内存使用
        for i in tqdm(range(0, total, batch_size), desc="Processing SMILES embeddings"):
            batch = smiles_list[i:i+batch_size]
            batch_emb = []
            
            for smiles in batch:
                try:
                    encodings = tokenizer(
                        smiles,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=290,
                        truncation=True,
                    )
                    # 将encodings移到设备上而不是将设备传递给模型
                    encodings = {k: v.to(device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        output = model_chem(**encodings)
                        smiles_embeddings = output.last_hidden_state[0, 0].cpu().numpy()
                        batch_emb.append(smiles_embeddings)
                except Exception as e:
                    logger.warning(f"Error processing SMILES '{smiles[:20]}...': {str(e)}")
                    # 使用零向量作为失败SMILES的嵌入
                    batch_emb.append(np.zeros(384))  # ChemBERTa维度
            
            emb_list.extend(batch_emb)
            
            # 手动释放内存
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                
        logger.info(f"SMILES embeddings extraction completed in {time.time() - start_time:.2f} seconds")
        return emb_list
    except Exception as e:
        logger.error(f"Failed to extract SMILES embeddings: {str(e)}")
        raise

# 获取所有.pth文件
def get_pth_files(folder_path, start_epoch=30, end_epoch=50):
    try:
        # 获取文件夹中所有.pth文件
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
            
        pth_files = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.endswith(".pth")
        ]
        
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in folder: {folder_path}")
            
        # 根据文件名中的epoch数字排序
        try:
            pth_files_sorted = sorted(
                pth_files, key=lambda x: int(x.split("_")[-1].replace(".pth", ""))
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Error sorting .pth files: {str(e)}. Using unsorted file list.")
            pth_files_sorted = pth_files
            
        # 返回指定范围的文件
        filtered_files = [f for f in pth_files_sorted if start_epoch <= int(f.split("_")[-1].replace(".pth", "")) <= end_epoch]
        
        if not filtered_files:
            logger.warning(f"No .pth files found between epochs {start_epoch} and {end_epoch}. Using all files.")
            return pth_files_sorted
            
        return filtered_files
    except Exception as e:
        logger.error(f"Failed to get .pth files: {str(e)}")
        raise

def main():
    # 解析命令行参数
    args = parse_arguments()
    start_time_total = time.time()
    
    try:
        # 创建结果目录
        if not os.path.exists(args.folder_path):
            logger.warning(f"Folder path {args.folder_path} does not exist. Creating it.")
            os.makedirs(args.folder_path, exist_ok=True)
        
        # 读取测试数据集
        logger.info(f"Loading test dataset from {args.test_path}")
        try:
            df_test = pd.read_csv(args.test_path)
            df_test["Protein"] = df_test["Protein"].apply(
                lambda x: x[:1022] if len(x) > 1022 else x
            )
            logger.info(f"Dataset shape: {df_test.shape}")
        except Exception as e:
            logger.error(f"Failed to load test dataset: {str(e)}")
            raise
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # 并行加载模型和处理特征
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 加载ESM模型
            esm_future = executor.submit(lambda: (
                esm.pretrained.esm1b_t33_650M_UR50S()
            ))
            
            # 加载ChemBERTa模型
            chembert_future = executor.submit(lambda: (
                "DeepChem/ChemBERTa-77M-MTR",
                AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR"),
                RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR", num_labels=2, add_pooling_layer=True)
            ))
            
            # 获取模型加载结果
            try:
                esm_model, alphabet = esm_future.result()
                esm_model = esm_model.eval().to(device)
                batch_converter = alphabet.get_batch_converter()
                logger.info("ESM model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ESM model: {str(e)}")
                raise
                
            try:
                model_name, tokenizer, model_chem = chembert_future.result()
                model_chem.to(device) # type: ignore
                logger.info("ChemBERTa model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ChemBERTa model: {str(e)}")
                raise
        
        # 提取蛋白质特征
        logger.info("Extracting protein features")
        pro_list_test = df_test["Protein"].unique()
        protein_features = get_protein_feature(list(pro_list_test), batch_converter, esm_model, device)
        df_test = pd.merge(df_test, protein_features, on="Protein", how="left")
        
        # 释放不需要的内存
        del esm_model
        torch.cuda.empty_cache()
        
        # 提取SMILES特征
        logger.info("Extracting SMILES features")
        df_test_unique = df_test.drop_duplicates(subset="SMILES")
        emb_list_test = get_embeddings(df_test_unique, tokenizer, model_chem, device)
        df_test_unique["fcfp"] = emb_list_test
        df_test = pd.merge(df_test, df_test_unique[["SMILES", "fcfp"]], on="SMILES", how="left")
        
        # 释放不需要的内存
        del model_chem, tokenizer
        torch.cuda.empty_cache()
        
        # 加载配置
        logger.info(f"Loading configuration from {args.config_file}")
        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.config_file)
        
        # 覆盖批处理大小（如果指定）
        if args.batch_size is not None:
            cfg.SOLVER.BATCH_SIZE = args.batch_size
            logger.info(f"Overriding batch size to {args.batch_size}")
            
        cfg.freeze()
        
        # 设置DataLoader
        test_dataset = DTIDataset(df_test.index.values, df_test)
        test_generator = DataLoader(
            test_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.SOLVER.NUM_WORKERS,
            drop_last=False,
            collate_fn=graph_collate_func,
        )
        
        # 加载GraphBAN模型
        modelG = GraphBAN(**cfg).to(device)
        opt = torch.optim.Adam(modelG.parameters(), lr=cfg.SOLVER.LR)
        
        # 获取.pth文件
        pth_files = get_pth_files(args.folder_path, args.start_epoch, args.end_epoch)
        logger.info(f"Found {len(pth_files)} model files for prediction")
        
        # 对每个模型进行预测
        i = 1
        for model_path in tqdm(pth_files, desc="Processing models"):
            epoch_num = int(model_path.split("_")[-1].replace(".pth", ""))
            logger.info(f"Processing model from epoch {epoch_num}")
            
            try:
                # 加载训练好的模型
                modelG.load_state_dict(torch.load(model_path, map_location=device))
                trainer = Trainer(modelG, opt, device, test_generator, **cfg)
                pred = trainer.train()
                
                # 保存结果
                df_test[f"pred{epoch_num}"] = pred
                i += 1
            except Exception as e:
                logger.error(f"Error processing model {model_path}: {str(e)}")
                continue
        
        # 清理内存并计算平均值
        logger.info("Computing average predictions")
        del df_test["esm"]
        del df_test["fcfp"]
        
        # 保存SMILES和蛋白质列
        smiles = df_test["SMILES"]
        proteins = df_test["Protein"]
        
        # 删除非预测列
        pred_columns = [col for col in df_test.columns if col.startswith("pred")]
        if not pred_columns:
            raise ValueError("No predictions were generated")
            
        df_pred_only = df_test[pred_columns]
        
        # 计算行平均值
        df_test["predicted_value"] = df_pred_only.mean(axis=1)
        
        # 创建结果数据框
        new_data = pd.DataFrame()
        new_data["SMILES"] = smiles
        new_data["Protein"] = proteins
        new_data["predicted_value"] = df_test["predicted_value"]
        
        # 保存结果
        output_path = os.path.join(args.folder_path, args.save_dir)
        new_data.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Total execution time: {time.time() - start_time_total:.2f} seconds")
        
    except Exception as e:
        logger.error(f"An error occurred during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
