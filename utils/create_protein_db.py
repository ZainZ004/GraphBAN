import argparse
import os
import sqlite3
import pandas as pd
from tqdm import tqdm
import logging
import sys
from Bio import SeqIO


def setup_logger(log_level=logging.INFO):
    """
    Configures and returns a logger object.

    Args:
        log_level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger object.
    """
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


def create_db(db_path, logger, overwrite=False):
    """
    Creates a new SQLite database and a protein_map table, or connects to an existing one.

    Args:
        db_path (str): Path to the SQLite database file.
        logger (logging.Logger): Logger object for logging messages.
        overwrite (bool, optional): Whether to overwrite an existing database (default: False).

    Returns:
        sqlite3.Connection: A connection to the SQLite database.
    """
    try:
        db_exists = os.path.exists(db_path)
        
        if db_exists:
            if overwrite:
                os.remove(db_path)
                logger.info(f"Deleted existing DB: {db_path}")
                db_exists = False
            else:
                logger.info(f"Using existing DB: {db_path}")
        
        # 创建一个新的数据库连接或连接到现有数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 如果数据库是新创建的或被覆盖了，则创建表和索引
        if not db_exists:
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
            logger.info(f"Successfully create DB: {db_path}")
        else:
            # 检查表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='protein_map'")
            if not cursor.fetchone():
                # 如果表不存在，则创建表和索引
                cursor.execute('''
                CREATE TABLE protein_map (
                    id TEXT,
                    sequence TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT
                )
                ''')
                
                cursor.execute('CREATE INDEX idx_sequence ON protein_map(sequence)')
                cursor.execute('CREATE INDEX idx_id ON protein_map(id)')
                
                conn.commit()
                logger.info(f"Created missing table in existing DB: {db_path}")
            else:
                logger.info(f"Using existing table in DB: {db_path}")
                
        return conn
    except sqlite3.Error as e:
        logger.error(f"An error occured when creating/connecting to DB: {e}")
        raise


def validate_db(db_path, logger):
    """
    验证数据库内容，检查序列是否都有对应的蛋白名称，并打印统计信息。

    Args:
        db_path (str): 数据库文件路径。
        logger (logging.Logger): 日志记录器。
    
    Returns:
        bool: 验证是否通过。
    """
    try:
        logger.info(f"正在验证数据库：{db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取总记录数
        cursor.execute("SELECT COUNT(*) FROM protein_map")
        total_records = cursor.fetchone()[0]
        
        # 检查缺失蛋白名称的记录
        cursor.execute("SELECT COUNT(*) FROM protein_map WHERE name IS NULL OR trim(name) = ''")
        missing_name_count = cursor.fetchone()[0]
        
        # 检查缺失序列的记录
        cursor.execute("SELECT COUNT(*) FROM protein_map WHERE sequence IS NULL OR trim(sequence) = ''")
        missing_seq_count = cursor.fetchone()[0]
        
        # 检查缺失ID的记录
        cursor.execute("SELECT COUNT(*) FROM protein_map WHERE id IS NULL OR trim(id) = ''")
        missing_id_count = cursor.fetchone()[0]
        
        # 打印统计信息
        logger.info("数据库验证结果:")
        logger.info(f"  - 总记录数: {total_records}")
        logger.info(f"  - 缺失蛋白名称的记录: {missing_name_count} ({(missing_name_count/total_records*100):.2f}% 如果有)")
        logger.info(f"  - 缺失序列的记录: {missing_seq_count} ({(missing_seq_count/total_records*100):.2f}% 如果有)")
        logger.info(f"  - 缺失ID的记录: {missing_id_count} ({(missing_id_count/total_records*100):.2f}% 如果有)")
        
        # 如果有问题记录，获取一些示例
        if missing_name_count > 0:
            cursor.execute("SELECT id, sequence FROM protein_map WHERE name IS NULL OR trim(name) = '' LIMIT 5")
            examples = cursor.fetchall()
            logger.warning("缺失蛋白名称的记录示例:")
            for idx, (prot_id, seq) in enumerate(examples, 1):
                logger.warning(f"  {idx}. ID: {prot_id}, 序列前缀: {seq[:20]}...")
        
        conn.close()
        
        # 判断验证是否通过
        validation_passed = (missing_seq_count == 0)  # 序列是必须的
        status = "通过" if validation_passed else "未通过"
        logger.info(f"数据库验证{status}")
        
        return validation_passed
        
    except sqlite3.Error as e:
        logger.error(f"验证数据库时发生错误: {e}")
        return False


def fix_missing_names(db_path, logger):
    """
    修复数据库中缺失蛋白名称的记录，使用ID作为名称。
    
    Args:
        db_path (str): 数据库文件路径。
        logger (logging.Logger): 日志记录器。
        
    Returns:
        int: 修复的记录数量。
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查找缺失蛋白名称的记录并修复
        cursor.execute("UPDATE protein_map SET name = id WHERE name IS NULL OR trim(name) = ''")
        fixed_count = cursor.rowcount
        conn.commit()
        
        logger.info(f"已修复 {fixed_count} 条缺失蛋白名称的记录")
        conn.close()
        return fixed_count
    except sqlite3.Error as e:
        logger.error(f"修复缺失蛋白名称时发生错误: {e}")
        return 0


def load_fasta_to_db(fasta_path, db_conn, batch_size=1000, logger=None):
    """
    Loads protein sequences from a FASTA file into the database.

    Args:
        fasta_path (str): Path to the input FASTA file.
        db_conn (sqlite3.Connection): Connection to the SQLite database.
        batch_size (int, optional): Number of records to process in a batch (default: 1000).
        logger (logging.Logger, optional): Logger object for logging messages.
    """
    if logger is None:
        logger = setup_logger()
    
    cursor = db_conn.cursor()
    counter = 0
    batch = []
    
    try:
        logger.info(f"From {fasta_path} loding protein Sequences...")
        with open(fasta_path, "r") as fasta_file:
            # 使用SeqIO解析FASTA文件
            for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="Handling FASTA Records"):
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
                
            logger.info(f"Successfully load {counter} records of protein to DB")
        
    except Exception as e:
        logger.error(f"An error occured when processing FASTA file: {e}")
        raise
        

def load_csv_to_db(csv_path, db_conn, id_col='id', seq_col='sequence', name_col='name', desc_col=None, batch_size=1000, logger=None):
    """
    Loads protein sequences from a CSV file into the database.

    Args:
        csv_path (str): Path to the input CSV file.
        db_conn (sqlite3.Connection): Connection to the SQLite database.
        id_col (str, optional): Name of the ID column in the CSV file (default: 'id').
        seq_col (str, optional): Name of the sequence column in the CSV file (default: 'sequence').
        name_col (str, optional): Name of the name column in the CSV file (default: 'name').
        desc_col (str, optional): Name of the description column in the CSV file (default: None).
        batch_size (int, optional): Number of records to process in a batch (default: 1000).
        logger (logging.Logger, optional): Logger object for logging messages.
    """
    if logger is None:
        logger = setup_logger()
    
    cursor = db_conn.cursor()
    counter = 0
    
    try:
        logger.info(f"From {csv_path} loading protein Sequences...")
        
        # 分块读取CSV文件以处理大文件
        for chunk in tqdm(pd.read_csv(csv_path, chunksize=batch_size), desc="Handling CSV Blocks"):
            batch = []
            
            # 验证必要的列存在
            required_cols = [id_col, seq_col]
            if name_col:
                required_cols.append(name_col)
            
            missing_cols = [col for col in required_cols if col not in chunk.columns]
            if missing_cols:
                raise ValueError(f"The CSV file lacked essential columns: {', '.join(missing_cols)}")
            
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
            
        logger.info(f"Successfully load {counter} records of protein to DB")
        
    except Exception as e:
        logger.error(f"An error occured when processing CSV file: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Create a protein database from FASTA or CSV files.")
    parser.add_argument("--db_path", required=True, help="Path to the SQLite database file")
    
    # 文件来源组
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--fasta", help="Path to the input FASTA file")
    source_group.add_argument("--csv", help="Path to the input CSV file")
    
    # CSV特定选项
    parser.add_argument("--id_col", default="id", help="Name of the ID column in the CSV file")
    parser.add_argument("--seq_col", default="sequence", help="Name of the sequence column in the CSV file")
    parser.add_argument("--name_col", default="name", help="Name of the name column in the CSV file")
    parser.add_argument("--desc_col", default=None, help="Name of the description column in the CSV file")
    
    # 通用选项
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for database insertion")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing database")
    parser.add_argument("--validate", action="store_true", help="Validate database after creation")
    parser.add_argument("--auto_fix", action="store_true", help="Automatically fix issues found during validation")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level)
    
    try:
        # 创建数据库和表或连接到现有数据库
        db_conn = create_db(args.db_path, logger, args.overwrite)
        
        # 根据输入类型加载数据
        if args.fasta:
            if not os.path.exists(args.fasta):
                logger.error(f"FASTA file can not be found: {args.fasta}")
                sys.exit(1)
            load_fasta_to_db(args.fasta, db_conn, args.batch_size, logger)
        elif args.csv:
            if not os.path.exists(args.csv):
                logger.error(f"CSV file can not be found: {args.csv}")
                sys.exit(1)
            load_csv_to_db(
                args.csv, db_conn, 
                args.id_col, args.seq_col, args.name_col, args.desc_col, 
                args.batch_size, logger
            )
            
        # 关闭数据库连接
        db_conn.close()
        
        # 验证数据库
        if args.validate or logger.level <= logging.INFO:  # 如果显式要求验证或日志级别是INFO及以下
            validation_passed = validate_db(args.db_path, logger)
            if not validation_passed:
                logger.warning("数据库验证未通过，可能存在缺失或无效的数据")
                
                # 自动修复问题
                if args.auto_fix:
                    logger.info("正在尝试自动修复问题...")
                    fixed_count = fix_missing_names(args.db_path, logger)
                    if fixed_count > 0:
                        logger.info(f"修复了 {fixed_count} 条缺失蛋白名称的记录")
                        # 重新验证以确认修复效果
                        if validate_db(args.db_path, logger):
                            logger.info("修复成功，数据库验证现在通过")
                        else:
                            logger.warning("修复后仍有问题存在")
                else:
                    logger.info("使用 --auto_fix 参数可以尝试自动修复这些问题")
        
        if os.path.exists(args.db_path):
            logger.info(f"Database updated successfully: {args.db_path}")
        else:
            logger.info(f"Database created successfully: {args.db_path}")
        
    except Exception as e:
        logger.error(f"An error occured in unknown position: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()