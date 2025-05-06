#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 文件: create_protein_db.py

import argparse
import os
import sqlite3
import pandas as pd
from tqdm import tqdm
import logging
import sys
from Bio import SeqIO


def setup_logger(log_level=logging.INFO):
    """配置并返回一个日志记录器对象"""
    logger = logging.getLogger("protein_db_builder")
    logger.setLevel(log_level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)

    return logger


def create_db(db_path, logger):
    """创建一个新的SQLite数据库和protein_map表"""
    try:
        # 如果文件已存在，先删除它
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"已删除现有数据库: {db_path}")
        
        # 创建一个新的数据库连接
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建protein_map表
        cursor.execute('''
        CREATE TABLE protein_map (
            id TEXT,
            sequence TEXT PRIMARY KEY,
            name TEXT,
            description TEXT
        )
        ''')
        
        # 创建索引以加快查询速度
        cursor.execute('CREATE INDEX idx_sequence ON protein_map(sequence)')
        cursor.execute('CREATE INDEX idx_id ON protein_map(id)')
        
        conn.commit()
        logger.info(f"成功创建数据库: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"创建数据库错误: {e}")
        raise


def load_fasta_to_db(fasta_path, db_conn, batch_size=1000, logger=None):
    """从FASTA文件加载蛋白质序列到数据库中"""
    if logger is None:
        logger = setup_logger()
    
    cursor = db_conn.cursor()
    counter = 0
    batch = []
    
    try:
        logger.info(f"开始从 {fasta_path} 加载蛋白质序列...")
        
        # 使用SeqIO解析FASTA文件
        for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="处理FASTA记录"):
            seq_id = record.id
            sequence = str(record.seq)
            description = record.description
            name = description.split(" ")[0] if " " in description else seq_id
            
            # 添加到批处理列表
            batch.append((seq_id, sequence, name, description))
            counter += 1
            
            # 批量提交到数据库
            if len(batch) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                    batch
                )
                db_conn.commit()
                batch = []
                
        # 提交最后一批
        if batch:
            cursor.executemany(
                "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                batch
            )
            db_conn.commit()
            
        logger.info(f"成功加载 {counter} 条蛋白质记录到数据库")
        
    except Exception as e:
        logger.error(f"加载FASTA到数据库时发生错误: {e}")
        raise
        

def load_csv_to_db(csv_path, db_conn, id_col='id', seq_col='sequence', name_col='name', desc_col=None, batch_size=1000, logger=None):
    """从CSV文件加载蛋白质序列到数据库中"""
    if logger is None:
        logger = setup_logger()
    
    cursor = db_conn.cursor()
    counter = 0
    
    try:
        logger.info(f"开始从 {csv_path} 加载蛋白质序列...")
        
        # 分块读取CSV文件以处理大文件
        for chunk in tqdm(pd.read_csv(csv_path, chunksize=batch_size), desc="处理CSV块"):
            batch = []
            
            # 验证必要的列存在
            required_cols = [id_col, seq_col]
            if name_col:
                required_cols.append(name_col)
            
            missing_cols = [col for col in required_cols if col not in chunk.columns]
            if missing_cols:
                raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_cols)}")
            
            for _, row in chunk.iterrows():
                seq_id = row[id_col]
                sequence = row[seq_col]
                name = row[name_col] if name_col and name_col in row else seq_id
                description = row[desc_col] if desc_col and desc_col in row else ""
                
                # 添加到批处理列表
                batch.append((seq_id, sequence, name, description))
                counter += 1
            
            # 批量提交到数据库
            cursor.executemany(
                "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                batch
            )
            db_conn.commit()
            
        logger.info(f"成功加载 {counter} 条蛋白质记录到数据库")
        
    except Exception as e:
        logger.error(f"加载CSV到数据库时发生错误: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="创建蛋白质序列数据库并从FASTA或CSV文件加载数据")
    parser.add_argument("--db_path", required=True, help="SQLite数据库文件的路径")
    
    # 文件来源组
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--fasta", help="输入FASTA文件的路径")
    source_group.add_argument("--csv", help="输入CSV文件的路径")
    
    # CSV特定选项
    parser.add_argument("--id_col", default="id", help="CSV中ID列的名称")
    parser.add_argument("--seq_col", default="sequence", help="CSV中序列列的名称")
    parser.add_argument("--name_col", default="name", help="CSV中名称列的名称")
    parser.add_argument("--desc_col", default=None, help="CSV中描述列的名称")
    
    # 通用选项
    parser.add_argument("--batch_size", type=int, default=1000, help="批处理大小")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level)
    
    try:
        # 创建数据库和表
        db_conn = create_db(args.db_path, logger)
        
        # 根据输入类型加载数据
        if args.fasta:
            if not os.path.exists(args.fasta):
                logger.error(f"FASTA文件不存在: {args.fasta}")
                sys.exit(1)
            load_fasta_to_db(args.fasta, db_conn, args.batch_size, logger)
        elif args.csv:
            if not os.path.exists(args.csv):
                logger.error(f"CSV文件不存在: {args.csv}")
                sys.exit(1)
            load_csv_to_db(
                args.csv, db_conn, 
                args.id_col, args.seq_col, args.name_col, args.desc_col, 
                args.batch_size, logger
            )
            
        # 关闭数据库连接
        db_conn.close()
        logger.info(f"数据库创建完成: {args.db_path}")
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()