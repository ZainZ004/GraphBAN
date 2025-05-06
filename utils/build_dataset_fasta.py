import argparse
import logging
import os
import sqlite3
import sys

from Bio import SeqIO
from tqdm import tqdm


def setup_logger(log_level=logging.INFO):
    """
    Sets up a logger for the script.

    Args:
        log_level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger("FASTA_Processor")
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


def init_db(db_path, logger=None):
    """
    Initializes a SQLite database for storing protein sequences and IDs.

    Args:
        db_path (str): Path to the SQLite database file.
        logger (logging.Logger, optional): Logger object for logging messages.

    Returns:
        sqlite3.Connection: A connection to the SQLite database.
    """
    try:
        # Create directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create protein_map table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS protein_map (
            id TEXT,
            sequence TEXT PRIMARY KEY,
            name TEXT,
            description TEXT
        )
        """)

        # Create indexes if they don't exist
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sequence ON protein_map(sequence)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_id ON protein_map(id)")

        conn.commit()
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        return None


def process_chem(inpute_SMILES_path, logger=None):
    """
    Processes the input SMILES file to extract chemical information.
    This is a placeholder function and should be implemented based on specific requirements.

    Args:
        inpute_SMILES_path (str): Path to the input SMILES file.
        logger (logging.Logger, optional): Logger object for logging messages.

    Returns:
        SMILES
    """
    try:
        with open(inpute_SMILES_path, "r") as f:
            for line in f:
                line = line.strip()
                # Check for both English and Chinese colons
                if line.startswith("SMILES:") or line.startswith("SMILES："):
                    # Split only on the first occurrence of the colon
                    parts = (
                        line.split(":", 1)
                        if line.startswith("SMILES:")
                        else line.split("：", 1)
                    )
                    if len(parts) > 1:
                        smiles_string = parts[1].strip()
                        return smiles_string
            # If loop finishes without finding the SMILES line
            logger.error(
                f"Error: SMILES string not found in file {inpute_SMILES_path}"
            )
            return None
    except FileNotFoundError:
        logger.error(
            f"Error: Input SMILES file not found at {inpute_SMILES_path}"
        )
        return None
    except Exception as e:
        logger.error(
            f"An error occurred while processing the SMILES file: {e}"
        )
        return None


def process_fasta_record(record, logger=None):
    """
    Processes a single SeqRecord from a FASTA file.
    Extracts ID and sequence.
    (Further processing to determine chemical/protein type
     and convert to specific format can be added here based on requirements)

    Args:
        record (SeqRecord): A Biopython SeqRecord object.
        logger (logging.Logger, optional): Logger object for logging messages.

    Returns:
        tuple: A tuple containing the sequence ID and the sequence string.
               Returns None if processing fails or record is invalid.
    """
    try:
        seq_id = record.id
        sequence = str(record.seq)
        description = record.description
        # Placeholder for future logic:
        # - Determine if it's a chemical or protein based on ID, sequence, description etc.
        # - Convert to the target data format.
        # For now, just return ID and sequence.
        return seq_id, sequence, description
    except Exception as e:
        if logger:
            logger.error(f"Error processing record {record.id}: {e}")
        return None


def process_large_fasta(
    input_fasta_path, chem_SMILES, output_file_path=None, db_path=None, logger=None
):
    """
    Reads a potentially large FASTA file record by record, processes each record,
    and writes the output incrementally.

    Args:
        input_fasta_path (str): Path to the input FASTA file.
        chem_SMILES (str): SMILES string to be paired with each sequence.
        output_file_path (str, optional): Path to the output file.
                                          If None, prints to stdout. Defaults to None.
        db_path (str, optional): Path to SQLite database for storing protein sequences.
                                 If None, no database is used. Defaults to None.
        logger (logging.Logger, optional): Logger object for logging messages.
    """
    output_handle = None
    input_handle = None  # Initialize input_handle
    pbar = None  # Initialize pbar
    db_conn = None
    db_cursor = None
    batch = []
    batch_size = 1000  # Number of records to process before committing to database

    try:
        # Initialize database if path is provided
        if db_path:
            db_conn = init_db(db_path)
            if db_conn:
                db_cursor = db_conn.cursor()
                logger.info(f"Database initialized at {db_path}")
            else:
                logger.error(f"Warning: Failed to initialize database at {db_path}")

        #!todo Use with statement for file handling
        # Open the input file
        input_handle = open(input_fasta_path, "r")

        # Use SeqIO.parse with the tqdm-wrapped file handle
        fasta_iterator = SeqIO.parse(input_handle, "fasta")

        if output_file_path:
            output_handle = open(output_file_path, "w")
            # Example: Write header if outputting CSV
            output_handle.write("SMILES,Sequence\n")
        else:
            output_handle = sys.stdout

        processed_count = 0
        for record in tqdm(fasta_iterator, desc="Processing FASTA records"):
            processed_data = process_fasta_record(record)
            if processed_data:
                seq_id, sequence, description = processed_data
                # Write processed data incrementally
                output_handle.write(f"{chem_SMILES},{sequence}\n")
                processed_count += 1

                # Store in database if enabled
                if db_cursor:
                    name = description.split(" ")[0] if " " in description else seq_id
                    batch.append((seq_id, sequence, name, description))

                    # Commit batch to database
                    if len(batch) >= batch_size:
                        try:
                            db_cursor.executemany(
                                "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                                batch,
                            )
                            db_conn.commit()
                            batch = []
                        except sqlite3.Error as e:
                            logger.error(f"Database error during batch insert: {e}")

        # Commit any remaining batch items to database
        if db_cursor and batch:
            try:
                db_cursor.executemany(
                    "INSERT OR IGNORE INTO protein_map (id, sequence, name, description) VALUES (?, ?, ?, ?)",
                    batch,
                )
                db_conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Database error during final batch insert: {e}")

    except FileNotFoundError:
        logger.error(f"Error: Input FASTA file not found at {input_fasta_path}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Close the tqdm progress bar (which also closes the input file handle)
        if pbar:
            pbar.close()
        # Ensure input handle is closed if pbar wasn't initialized (e.g., FileNotFoundError)
        elif input_handle:
            input_handle.close()

        # Close the output file handle if it was opened and is not stdout
        if output_file_path and output_handle and output_handle is not sys.stdout:
            output_handle.close()
            logger.info(f"Output written to {output_file_path}")

        # Close database connection if it was opened
        if db_conn:
            db_conn.close()
            if db_path and "processed_count" in locals():
                logger.info(f"Database updated with {processed_count} protein sequences")

        # Print final record count summary regardless of output destination
        if "processed_count" in locals():  # Check if processing started
            logger.info(f"\nFinished processing. Total records processed: {processed_count}")


def create_name_sequence_mapping(input_fasta_path, output_file_path, logger=None):
    """
    Creates a CSV mapping file with 'name' and 'sequence' columns from a FASTA file.
    This mapping file can be used by post_data_process.py for protein name lookup.

    Args:
        input_fasta_path (str): Path to the input FASTA file.
        output_file_path (str): Path to the output CSV file.
        logger (logging.Logger, optional): Logger object for logging messages.
    """
    if logger is None:
        logger = setup_logger()
    
    input_handle = None
    output_handle = None
    processed_count = 0
    
    try:
        # Open the input file
        input_handle = open(input_fasta_path, "r")
        
        # Use SeqIO.parse with tqdm for progress tracking
        fasta_iterator = SeqIO.parse(input_handle, "fasta")
        
        # Open output file and write header
        output_handle = open(output_file_path, "w")
        output_handle.write("sequence,name\n")  # Header as required by post_data_process.py
        
        logger.info(f"Creating name-sequence mapping from {input_fasta_path}...")
        
        for record in tqdm(fasta_iterator, desc="Processing FASTA records"):
            processed_data = process_fasta_record(record, logger)
            if processed_data:
                seq_id, sequence, description = processed_data
                # Extract name from description or use ID if description is minimal
                name = description.split(" ")[0] if " " in description else seq_id
                
                # Write mapping entry: sequence,name
                output_handle.write(f"{sequence},{name}\n")
                processed_count += 1
                
    except FileNotFoundError:
        logger.error(f"Error: Input FASTA file not found at {input_fasta_path}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Clean up resources
        if input_handle:
            input_handle.close()
        
        if output_handle:
            output_handle.close()
            logger.info(f"Output mapping written to {output_file_path}")
        
        if processed_count > 0:
            logger.info(f"\nFinished processing. Total records processed: {processed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Process large FASTA files efficiently, converting sequences to a target format."
    )
    parser.add_argument("input_fasta", help="Path to the input FASTA file.")
    
    # Create mutually exclusive group for different operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--chem_SMILES", help="Input SMILES file to be predicted.", default=None)
    mode_group.add_argument("--create_mapping", action="store_true", 
                          help="Create a name-sequence mapping CSV file for use with post_data_process.py")
    
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output file (optional). If not provided, output goes to standard output.",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--database",
        help="Path to SQLite database for storing protein sequences (optional).",
        default=None,
    )
    # Add more arguments if needed for specifying chemical/protein types or output formats

    args = parser.parse_args()

    if not os.path.exists(args.input_fasta):
        print(f"Error: Input file '{args.input_fasta}' not found.", file=sys.stderr)
        sys.exit(1)
        
    # Setup logger
    logger = setup_logger()
    
    # Determine which operation mode to run
    if args.create_mapping:
        # Run in mapping creation mode
        if not args.output:
            print("Error: Output file path is required when creating a mapping file.", file=sys.stderr)
            sys.exit(1)
        create_name_sequence_mapping(args.input_fasta, args.output, logger)
    else:
        # Run in normal SMILES prediction mode
        chem_SMILES = process_chem(args.chem_SMILES, logger)
        if not chem_SMILES:
            print(
                f"Error: Failed to process SMILES from '{args.chem_SMILES}'",
                file=sys.stderr,
            )
            sys.exit(1)
        process_large_fasta(args.input_fasta, chem_SMILES, args.output, args.database, logger)


if __name__ == "__main__":
    main()
