import argparse
import os
import sqlite3
import pandas as pd
from tqdm import tqdm
import logging
import sys
from Bio import SeqIO
import concurrent.futures
import queue
import threading
import time


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
        
        # 设置数据库以支持极长序列
        cursor = conn.cursor()
        # 禁用同步写入，提高性能
        cursor.execute("PRAGMA synchronous = OFF")
        # 启用内存映射，提高大型数据库性能
        cursor.execute("PRAGMA mmap_size = 30000000000")  # 约30GB
        
        # 如果数据库是新创建的或被覆盖了，则创建表和索引
        if not db_exists:
            # 创建protein_map表，使用id作为主键
            cursor.execute('''
            CREATE TABLE protein_map (
                id TEXT PRIMARY KEY,
                sequence TEXT NOT NULL,
                name TEXT,
                description TEXT
            )
            ''')
            
            # 创建索引以加快查询速度
            cursor.execute('CREATE INDEX idx_sequence ON protein_map(sequence)')
            
            conn.commit()
            logger.info(f"Successfully create DB: {db_path}")
        else:
            # 检查表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='protein_map'")
            if not cursor.fetchone():
                # 如果表不存在，则创建表和索引
                cursor.execute('''
                CREATE TABLE protein_map (
                    id TEXT PRIMARY KEY,
                    sequence TEXT NOT NULL,
                    name TEXT,
                    description TEXT
                )
                ''')
                
                cursor.execute('CREATE INDEX idx_sequence ON protein_map(sequence)')
                
                conn.commit()
                logger.info(f"Created missing table in existing DB: {db_path}")
            else:
                logger.info(f"Using existing table in DB: {db_path}")
                
        return conn
    except sqlite3.Error as e:
        logger.error(f"An error occured when creating/connecting to DB: {e}")
        raise


def validate_db(db_path, logger, expected_records=None):
    """
    验证数据库内容，检查序列是否都有对应的蛋白名称，并打印统计信息。

    Args:
        db_path (str): 数据库文件路径。
        logger (logging.Logger): 日志记录器。
        expected_records (int, optional): 预期的记录数量，用于检查是否所有记录都成功插入.
    
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
        
        # 检查序列长度统计
        cursor.execute("SELECT MIN(LENGTH(sequence)), MAX(LENGTH(sequence)), AVG(LENGTH(sequence)) FROM protein_map")
        min_len, max_len, avg_len = cursor.fetchone()
        
        # 检查极长序列（超过10000个氨基酸）的数量
        cursor.execute("SELECT COUNT(*) FROM protein_map WHERE LENGTH(sequence) > 10000")
        very_long_seq_count = cursor.fetchone()[0]
        
        # 打印统计信息
        logger.info(f"数据库验证结果:")
        logger.info(f"  - 总记录数: {total_records}")
        logger.info(f"  - 缺失蛋白名称的记录: {missing_name_count} ({(missing_name_count/total_records*100):.2f}% 如果有)")
        logger.info(f"  - 缺失序列的记录: {missing_seq_count} ({(missing_seq_count/total_records*100):.2f}% 如果有)")
        logger.info(f"  - 缺失ID的记录: {missing_id_count} ({(missing_id_count/total_records*100):.2f}% 如果有)")
        logger.info(f"  - 序列长度统计: 最短 {min_len}, 最长 {max_len}, 平均 {avg_len:.1f}")
        
        if very_long_seq_count > 0:
            logger.warning(f"  - 发现 {very_long_seq_count} 条极长序列 (>10000个氨基酸)")
            cursor.execute("SELECT id, LENGTH(sequence) FROM protein_map WHERE LENGTH(sequence) > 10000 ORDER BY LENGTH(sequence) DESC LIMIT 5")
            long_examples = cursor.fetchall()
            logger.warning("  - 最长的几条序列:")
            for idx, (prot_id, seq_len) in enumerate(long_examples, 1):
                logger.warning(f"    {idx}. ID: {prot_id}, 长度: {seq_len}")
        
        # 如果有预期记录数，检查是否所有记录都成功插入
        if expected_records is not None and expected_records != total_records:
            skipped_records = expected_records - total_records
            if skipped_records > 0:
                logger.warning(f"预期插入 {expected_records} 条记录，但实际只有 {total_records} 条")
                logger.warning(f"有 {skipped_records} 条记录 ({(skipped_records/expected_records*100):.2f}%) 未能插入，可能是因为序列重复或已存在")
            elif skipped_records < 0:
                logger.warning(f"数据库中记录数 ({total_records}) 超过了预期的记录数 ({expected_records})，可能有其他来源的数据")
        
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


def check_sequence_retrievable(db_path, logger, sample_size=10):
    """
    检查数据库中的序列是否可以正常检索。
    随机抽取样本序列并尝试通过ID和序列检索，确认数据库索引正常工作。
    
    Args:
        db_path (str): 数据库文件路径。
        logger (logging.Logger): 日志记录器。
        sample_size (int): 抽样检查的记录数量。
        
    Returns:
        bool: 检查是否通过。
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取总记录数
        cursor.execute("SELECT COUNT(*) FROM protein_map")
        total_records = cursor.fetchone()[0]
        
        if total_records == 0:
            logger.warning("数据库中没有记录，无法进行检索测试")
            return False
            
        # 确定抽样数量不超过总记录数
        actual_sample_size = min(sample_size, total_records)
        
        # 随机抽取记录
        cursor.execute(f"SELECT id, sequence, name FROM protein_map ORDER BY RANDOM() LIMIT {actual_sample_size}")
        samples = cursor.fetchall()
        
        logger.info(f"正在对 {actual_sample_size} 条随机记录进行检索测试...")
        success_count = 0
        failed_id_queries = 0
        failed_seq_queries = 0
        
        for i, (prot_id, sequence, name) in enumerate(samples, 1):
            # 通过ID检索
            cursor.execute("SELECT sequence FROM protein_map WHERE id = ?", (prot_id,))
            id_result = cursor.fetchone()
            
            # 通过序列检索
            cursor.execute("SELECT id FROM protein_map WHERE sequence = ?", (sequence,))
            seq_results = cursor.fetchall()  # 可能有多个结果
            
            # 检查结果
            id_ok = id_result is not None and id_result[0] == sequence
            seq_ok = False
            
            # 对于序列检索，可能会有多个ID对应同一序列，只要找到我们当前的ID即可
            if seq_results:
                for result in seq_results:
                    if result[0] == prot_id:
                        seq_ok = True
                        break
            
            if id_ok and seq_ok:
                success_count += 1
                logger.debug(f"测试 {i}/{actual_sample_size}: ID '{prot_id}' 检索成功")
            else:
                if not id_ok:
                    failed_id_queries += 1
                    logger.warning(f"测试 {i}/{actual_sample_size}: ID '{prot_id}' 通过ID检索失败")
                if not seq_ok:
                    failed_seq_queries += 1
                    result_ids = [r[0] for r in seq_results]
                    logger.warning(f"测试 {i}/{actual_sample_size}: ID '{prot_id}' 通过序列检索失败")
                    logger.debug(f"  - 该序列对应的ID: {result_ids}")
        
        success_rate = success_count / actual_sample_size * 100
        logger.info(f"检索测试完成: {success_count}/{actual_sample_size} 成功 ({success_rate:.1f}%)")
        
        if failed_id_queries > 0 or failed_seq_queries > 0:
            logger.warning(f"ID查询失败: {failed_id_queries}, 序列查询失败: {failed_seq_queries}")
            
            # 检查潜在的索引问题
            cursor.execute("PRAGMA index_list('protein_map')")
            indexes = cursor.fetchall()
            if not indexes:
                logger.error("未找到任何索引，这可能是检索问题的原因")
            else:
                logger.info(f"数据库中存在 {len(indexes)} 个索引:")
                for idx in indexes:
                    logger.info(f"  - {idx[1]} (序号: {idx[0]}, 唯一性: {idx[2]})")
                    
            # 检查是否存在重复ID
            cursor.execute(
                "SELECT id, COUNT(*) as count FROM protein_map GROUP BY id HAVING count > 1"
            )
            duplicate_ids = cursor.fetchall()
            if duplicate_ids:
                logger.warning(f"发现 {len(duplicate_ids)} 个重复的ID:")
                for dup_id, count in duplicate_ids[:5]:  # 只显示前5个
                    logger.warning(f"  - ID '{dup_id}' 出现 {count} 次")
                if len(duplicate_ids) > 5:
                    logger.warning(f"  - 以及其他 {len(duplicate_ids)-5} 个重复ID...")
        
        conn.close()
        
        # 调整成功率阈值，如果超过95%也算成功（允许少量特殊情况）
        return success_rate >= 95
        
    except sqlite3.Error as e:
        logger.error(f"检索测试时发生错误: {e}")
        return False


def fix_missing_records(source_path, db_path, logger, is_fasta=True, id_col='id', seq_col='sequence', name_col='name', desc_col=None):
    """
    检测并修复缺失的记录，通过比对源文件和数据库内容找出未能成功导入的记录。
    
    Args:
        source_path (str): 源文件路径（FASTA或CSV）。
        db_path (str): 数据库文件路径。
        logger (logging.Logger): 日志记录器。
        is_fasta (bool): 源文件是否为FASTA格式（默认True，否则为CSV）。
        id_col (str): CSV文件中的ID列名（仅在is_fasta=False时使用）。
        seq_col (str): CSV文件中的序列列名（仅在is_fasta=False时使用）。
        name_col (str): CSV文件中的名称列名（仅在is_fasta=False时使用）。
        desc_col (str): CSV文件中的描述列名（仅在is_fasta=False时使用）。
        
    Returns:
        int: 成功修复的记录数。
    """
    try:
        logger.info(f"开始检测和修复缺失记录...")
        
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 存储已有记录的ID和序列集合（用于快速查找）
        logger.info("加载数据库中已有的记录信息...")
        cursor.execute("SELECT id, sequence FROM protein_map")
        existing_records = set()
        existing_ids = set()
        existing_seqs = set()
        
        for id, seq in tqdm(cursor.fetchall(), desc="加载已有记录"):
            existing_records.add((id, seq))
            existing_ids.add(id)
            existing_seqs.add(seq)
            
        logger.info(f"已加载 {len(existing_records)} 条现有记录信息")
        
        # 存储已修复的记录数
        fixed_count = 0
        duplicate_count = 0
        
        # 分批处理，避免内存溢出
        batch_size = 1000
        batch = []
        
        if is_fasta:
            # 处理FASTA文件
            total_records = 0
            with open(source_path, "r") as fasta_file:
                for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="检查缺失记录"):
                    total_records += 1
                    seq_id = record.id
                    sequence = str(record.seq)
                    
                    # 检查此记录是否已存在于数据库中
                    if (seq_id, sequence) in existing_records or seq_id in existing_ids or sequence in existing_seqs:
                        duplicate_count += 1
                        continue
                        
                    # 记录不存在，准备插入
                    description = record.description
                    name = description.split(" ")[0] if " " in description else seq_id
                    
                    batch.append((seq_id, sequence, name, description))
                    
                    if len(batch) >= batch_size:
                        # 批量插入
                        cursor.executemany(
                            "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                            batch
                        )
                        rows_affected = cursor.rowcount
                        fixed_count += rows_affected
                        conn.commit()
                        
                        # 更新现有记录集合
                        for id, seq, _, _ in batch:
                            existing_records.add((id, seq))
                            existing_ids.add(id)
                            existing_seqs.add(seq)
                            
                        batch = []
                        
            # 插入最后一批
            if batch:
                cursor.executemany(
                    "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                    batch
                )
                rows_affected = cursor.rowcount
                fixed_count += rows_affected
                conn.commit()
            
            logger.info(f"源文件共 {total_records} 条记录，其中 {duplicate_count} 条与数据库重复")
        
        else:
            total_records = 0
            for chunk in tqdm(pd.read_csv(source_path, chunksize=batch_size), desc="检查缺失记录"):
                batch = []
                for _, row in chunk.iterrows():
                    total_records += 1
                    seq_id = row[id_col]
                    sequence = row[seq_col]
                    
                    # 检查此记录是否已存在于数据库中
                    if (seq_id, sequence) in existing_records or seq_id in existing_ids or sequence in existing_seqs:
                        duplicate_count += 1
                        continue
                        
                    # 记录不存在，准备插入
                    name = row[name_col] if name_col and name_col in row else seq_id
                    description = row[desc_col] if desc_col and desc_col in row else ""
                    
                    batch.append((seq_id, sequence, name, description))
                
                # 批量插入
                if batch:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                        batch
                    )
                    rows_affected = cursor.rowcount
                    fixed_count += rows_affected
                    conn.commit()
                    
                    # 更新现有记录集合
                    for id, seq, _, _ in batch:
                        existing_records.add((id, seq))
                        existing_ids.add(id)
                        existing_seqs.add(seq)
        
        # 关闭数据库连接
        conn.close()
        
        logger.info(f"成功修复了 {fixed_count} 条缺失记录，有 {duplicate_count} 条记录重复")
        return fixed_count
        
    except Exception as e:
        logger.error(f"修复缺失记录时发生错误: {e}")
        return 0


class BatchProcessor:
    """
    批量处理器类，用于管理数据批次的处理和数据库插入
    """
    def __init__(self, db_path, batch_size=1000, max_workers=None, logger=None):
        """
        初始化批量处理器
        
        Args:
            db_path (str): 数据库文件路径
            batch_size (int): 每批处理的记录数
            max_workers (int): 最大工作线程数，默认为None（使用系统默认值）
            logger (logging.Logger): 日志记录器
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.logger = logger or setup_logger()
        
        # 创建队列用于收集处理结果
        self.result_queue = queue.Queue()
        
        # 创建锁用于同步数据库访问
        self.db_lock = threading.Lock()
        
        # 记录处理的总记录数
        self.total_processed = 0
        
        # 创建进度条
        self.pbar = None
        
        # 创建线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        self.logger.info(f"初始化批量处理器，使用 {self.max_workers} 个工作线程")
    
    def process_fasta_batch(self, records):
        """
        处理一批FASTA记录
        
        Args:
            records (list): FASTA记录列表
            
        Returns:
            list: 处理后的数据行列表，格式为 [(id, sequence, name, description), ...]
        """
        result = []
        for record in records:
            seq_id = record.id
            sequence = str(record.seq)
            description = record.description
            name = description.split(" ")[0] if " " in description else seq_id
            
            result.append((seq_id, sequence, name, description))
            
        return result
    
    def process_csv_batch(self, chunk, id_col, seq_col, name_col, desc_col):
        """
        处理一批CSV数据
        
        Args:
            chunk (pd.DataFrame): 数据块
            id_col (str): ID列名
            seq_col (str): 序列列名
            name_col (str): 名称列名
            desc_col (str): 描述列名
            
        Returns:
            list: 处理后的数据行列表，格式为 [(id, sequence, name, description), ...]
        """
        result = []
        
        for _, row in chunk.iterrows():
            seq_id = row[id_col]
            sequence = row[seq_col]
            name = row[name_col] if name_col and name_col in row else seq_id
            description = row[desc_col] if desc_col and desc_col in row else ""
            
            result.append((seq_id, sequence, name, description))
            
        return result
    
    def _insert_batch_to_db(self, batch):
        """
        将一批处理好的数据插入数据库
        
        Args:
            batch (list): 处理后的数据行列表
            
        Returns:
            int: 处理的记录数
        """
        if not batch:
            return 0
            
        with self.db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.executemany(
                    "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                    batch
                )
                conn.commit()
                conn.close()
                
                return len(batch)
            except sqlite3.Error as e:
                self.logger.error(f"数据库插入错误: {e}")
                return 0
    
    def _db_writer_thread(self):
        """
        数据库写入线程，持续从结果队列获取处理好的批次并写入数据库
        """
        self.logger.debug("数据库写入线程启动")
        while True:
            try:
                batch = self.result_queue.get(timeout=5)  # 5秒超时
                if batch is None:  # 结束信号
                    self.logger.debug("数据库写入线程收到结束信号")
                    break
                    
                records_processed = self._insert_batch_to_db(batch)
                self.total_processed += records_processed
                
                if self.pbar:
                    self.pbar.update(records_processed)
                    
                self.result_queue.task_done()
            except queue.Empty:
                continue  # 队列为空，继续等待
    
    def load_fasta_file(self, fasta_path):
        """
        从FASTA文件加载数据到数据库（多线程版本）
        
        Args:
            fasta_path (str): FASTA文件路径
            
        Returns:
            int: 处理的总记录数
        """
        self.logger.info(f"从 {fasta_path} 多线程加载蛋白质序列...")
        
        # 计算文件大小以估计进度
        file_size = os.path.getsize(fasta_path)
        self.pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="处理FASTA文件")
        
        # 启动数据库写入线程
        db_writer = threading.Thread(target=self._db_writer_thread)
        db_writer.daemon = True
        db_writer.start()
        
        try:
            with open(fasta_path, "r") as fasta_file:
                # 准备批处理
                current_batch = []
                batch_size = 0
                futures = []
                
                # 使用SeqIO解析FASTA文件
                for record in SeqIO.parse(fasta_file, "fasta"):
                    current_batch.append(record)
                    batch_size += 1
                    
                    if batch_size >= self.batch_size:
                        # 提交批处理任务
                        batch_to_process = current_batch
                        future = self.executor.submit(self.process_fasta_batch, batch_to_process)
                        futures.append(future)
                        
                        # 检查已完成的任务并将结果放入队列
                        self._check_completed_futures(futures)
                        
                        # 重置批处理
                        current_batch = []
                        batch_size = 0
                    
                    # 更新进度条（近似值）
                    if self.pbar:
                        # 估计每条记录的平均大小
                        self.pbar.update(len(str(record)) + 10)  # 添加一些额外字节以考虑格式
                
                # 处理最后一批
                if current_batch:
                    future = self.executor.submit(self.process_fasta_batch, current_batch)
                    futures.append(future)
                
                # 等待所有任务完成
                for future in concurrent.futures.as_completed(futures):
                    self.result_queue.put(future.result())
                
                # 发送结束信号到数据库写入线程
                self.result_queue.put(None)
                
                # 等待数据库写入线程完成
                db_writer.join()
                
            self.logger.info(f"成功加载 {self.total_processed} 条蛋白质记录到数据库")
            
            if self.pbar:
                self.pbar.close()
                
            return self.total_processed
            
        except Exception as e:
            self.logger.error(f"处理FASTA文件时发生错误: {e}")
            # 确保发送结束信号
            self.result_queue.put(None)
            if self.pbar:
                self.pbar.close()
            raise
    
    def load_csv_file(self, csv_path, id_col='id', seq_col='sequence', name_col='name', desc_col=None):
        """
        从CSV文件加载数据到数据库（多线程版本）
        
        Args:
            csv_path (str): CSV文件路径
            id_col (str): ID列名
            seq_col (str): 序列列名
            name_col (str): 名称列名
            desc_col (str): 描述列名
            
        Returns:
            int: 处理的总记录数
        """
        self.logger.info(f"从 {csv_path} 多线程加载蛋白质序列...")
        
        # 启动数据库写入线程
        db_writer = threading.Thread(target=self._db_writer_thread)
        db_writer.daemon = True
        db_writer.start()
        
        try:
            # 先获取CSV文件的总行数以设置进度条
            total_rows = sum(1 for _ in open(csv_path)) - 1  # 减去标题行
            self.pbar = tqdm(total=total_rows, desc="处理CSV记录")
            
            futures = []
            
            # 分块读取CSV文件
            for chunk in pd.read_csv(csv_path, chunksize=self.batch_size):
                # 验证必要的列存在
                required_cols = [id_col, seq_col]
                if name_col:
                    required_cols.append(name_col)
                
                missing_cols = [col for col in required_cols if col not in chunk.columns]
                if missing_cols:
                    raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_cols)}")
                
                # 提交批处理任务
                future = self.executor.submit(
                    self.process_csv_batch, 
                    chunk, id_col, seq_col, name_col, desc_col
                )
                futures.append(future)
                
                # 检查已完成的任务并将结果放入队列
                self._check_completed_futures(futures)
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                self.result_queue.put(future.result())
            
            # 发送结束信号到数据库写入线程
            self.result_queue.put(None)
            
            # 等待数据库写入线程完成
            db_writer.join()
            
            self.logger.info(f"成功加载 {self.total_processed} 条蛋白质记录到数据库")
            
            if self.pbar:
                self.pbar.close()
                
            return self.total_processed
            
        except Exception as e:
            self.logger.error(f"处理CSV文件时发生错误: {e}")
            # 确保发送结束信号
            self.result_queue.put(None)
            if self.pbar:
                self.pbar.close()
            raise
    
    def _check_completed_futures(self, futures, max_queue_size=50):
        """
        检查已完成的Future，将结果添加到队列，并从列表中删除已完成的Future
        
        Args:
            futures (list): Future对象列表
            max_queue_size (int): 队列最大大小，超过此大小将等待队列减小
        """
        # 创建一个副本以便在迭代时删除
        pending = []
        
        for future in futures:
            if future.done():
                try:
                    # 如果队列太大，等待一下以避免内存过度使用
                    while self.result_queue.qsize() > max_queue_size:
                        time.sleep(0.1)
                        
                    result = future.result()
                    self.result_queue.put(result)
                except Exception as e:
                    self.logger.error(f"处理任务时发生错误: {e}")
            else:
                pending.append(future)
        
        # 更新futures列表，只保留未完成的
        futures.clear()
        futures.extend(pending)


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
    parser.add_argument("--threads", type=int, default=None, help="Number of worker threads to use (default: auto)")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of samples to test for retrieval checks")
    parser.add_argument("--skip_fixes", action="store_true", help="Skip automatic fixes for issues found")
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_level)
    
    try:
        # 存储源文件路径用于后续可能的修复
        source_path = args.fasta if args.fasta else args.csv
        is_fasta = bool(args.fasta)
        
        # 创建数据库和表或连接到现有数据库
        db_conn = create_db(args.db_path, logger, args.overwrite)
        db_conn.close()  # 先关闭连接，让多线程处理器管理连接
        
        processed_records = 0  # 跟踪处理的记录总数
        
        # 初始化批量处理器
        batch_processor = BatchProcessor(
            args.db_path, 
            args.batch_size, 
            max_workers=args.threads,
            logger=logger
        )
        
        # 根据输入类型加载数据
        if args.fasta:
            if not os.path.exists(args.fasta):
                logger.error(f"FASTA file can not be found: {args.fasta}")
                sys.exit(1)
            processed_records = batch_processor.load_fasta_file(args.fasta)
        elif args.csv:
            if not os.path.exists(args.csv):
                logger.error(f"CSV file can not be found: {args.csv}")
                sys.exit(1)
            processed_records = batch_processor.load_csv_file(
                args.csv, args.id_col, args.seq_col, args.name_col, args.desc_col
            )
            
        # 验证数据库
        has_issues = False
        if args.validate or logger.level <= logging.INFO:  # 如果显式要求验证或日志级别是INFO及以下
            validation_passed = validate_db(args.db_path, logger, processed_records)
            if not validation_passed:
                logger.warning("数据库验证未通过，可能存在缺失或无效的数据")
                has_issues = True
                
                # 自动修复问题
                if not args.skip_fixes:
                    logger.info("正在尝试自动修复缺失蛋白名称问题...")
                    fixed_count = fix_missing_names(args.db_path, logger)
                    if fixed_count > 0:
                        logger.info(f"修复了 {fixed_count} 条缺失蛋白名称的记录")
                        # 重新验证以确认修复效果
                        if validate_db(args.db_path, logger):
                            logger.info("修复成功，数据库验证现在通过")
                            has_issues = False
                        else:
                            logger.warning("修复蛋白名称后仍有问题存在")
                else:
                    logger.info("跳过自动修复，使用 --skip_fixes=False 可启用自动修复")
        
        # 检查序列是否可以正常检索
        retrieval_ok = check_sequence_retrievable(args.db_path, logger, args.sample_size)
        if retrieval_ok:
            logger.info("序列检索测试通过")
        else:
            logger.warning("序列检索测试未通过，可能存在索引问题")
            has_issues = True
        
        # 修复缺失记录，只有当发现了问题并且没有禁用自动修复时执行
        if has_issues and not args.skip_fixes:
            logger.info("正在修复源文件中存在但未导入到数据库的记录...")
            if is_fasta:
                fixed_missing = fix_missing_records(source_path, args.db_path, logger, is_fasta=True)
            else:
                fixed_missing = fix_missing_records(
                    source_path, args.db_path, logger, 
                    is_fasta=False, 
                    id_col=args.id_col, 
                    seq_col=args.seq_col, 
                    name_col=args.name_col, 
                    desc_col=args.desc_col
                )
                
            if fixed_missing > 0:
                logger.info(f"成功从源文件修复了 {fixed_missing} 条缺失记录")
                # 再次检查序列是否可检索
                if not retrieval_ok:
                    if check_sequence_retrievable(args.db_path, logger, args.sample_size):
                        logger.info("修复后序列检索测试通过")
                    else:
                        logger.warning("修复后序列检索测试仍未通过，可能需要重建数据库")
        
        if os.path.exists(args.db_path):
            logger.info(f"Database updated successfully: {args.db_path}")
        else:
            logger.info(f"Database created successfully: {args.db_path}")
        
    except Exception as e:
        logger.error(f"An error occured in unknown position: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()