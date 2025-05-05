import pandas as pd
from tqdm import tqdm
import argparse
import os
import requests  # Added for UniProt API
import sqlite3  # Added for SQLite
import time  # Added for potential API rate limiting
import concurrent.futures  # Added for parallel processing
import logging  # Added logging module
import traceback


# Configure logger
def setup_logger(log_level=logging.INFO):
    """Configure and return a logger object"""
    logger = logging.getLogger("protein_processor")
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger already has them
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Set log format
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger


# --- Mapping Functions ---


def get_uniprot_name(sequence, retries=3, delay=1, logger=None):
    """Fetches protein name from UniProt using REST API."""
    # Note: UniProt API usage policies should be reviewed for bulk queries.
    # Increased parallelism might hit rate limits faster.
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f'sequence:"{sequence}"',  # Search by exact sequence
        "fields": "protein_name",
        "format": "tsv",
        "size": 1,  # Expecting one primary match
    }
    for attempt in range(retries):
        try:
            # Consider adjusting timeout for potentially slower responses under load
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            lines = response.text.strip().split("\n")
            if (
                len(lines) > 1 and lines[1]
            ):  # Check if we got a result line after the header
                # Assuming the name is the first column after the header
                return lines[1].split("\t")[0]  # Extract the protein name
            else:
                return sequence  # Return original sequence if not found
        except requests.exceptions.RequestException as e:
            # Reduce verbosity of warnings during parallel execution
            logger.error(f"UniProt API Error: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
    return sequence  # Fallback if all retries fail


def get_name_from_csv(sequence, mapping_dict):
    """Looks up protein name from a pre-loaded CSV mapping dictionary."""
    return mapping_dict.get(sequence, sequence)


def get_name_from_sqlite(sequence, db_path=None, db_cursor=None):
    """Looks up protein name from a SQLite database."""
    # If a database path is provided, create a new connection (suitable for parallel processing)
    if db_path and not db_cursor:
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM protein_map WHERE sequence = ?", (sequence,)
            )
            result = cursor.fetchone()
            return_val = result[0] if result else sequence
            conn.close()
            return return_val
        except sqlite3.Error as e:
            logger.error(f"SQLite Error: {e}")
            if conn:
                conn.close()
            return sequence  # Return original sequence on DB error
    # If a cursor is provided, use the existing connection (suitable for serial processing)
    elif db_cursor:
        try:
            db_cursor.execute(
                "SELECT name FROM protein_map WHERE sequence = ?", (sequence,)
            )
            result = db_cursor.fetchone()
            return result[0] if result else sequence
        except sqlite3.Error as e:
            logger.error(f"SQLite Error: {e}")
            return sequence  # Return original sequence on DB error
    else:
        return sequence  # Return original if neither cursor nor path provided


# --- Parallel chunk processing function ---
def process_chunk(chunk_data):
    """
    Processes a single DataFrame chunk: applies protein name mapping.
    Designed for parallel processing, includes SQLite connection management.

    Args:
        chunk_data: A tuple containing the chunk, mapping method, and mapping data
    """
    chunk, mapping_method, mapping_data_or_path = chunk_data

    if "Protein" not in chunk.columns or "predicted_value" not in chunk.columns:
        # Skip the chunk if required columns are missing
        return None

    # Internal function to apply mapping based on the method
    def apply_mapping(sequence):
        if mapping_method == "uniprot":
            # UniProt API mapping
            return get_uniprot_name(sequence)
        elif mapping_method == "csv":
            # CSV dictionary mapping
            return get_name_from_csv(sequence, mapping_data_or_path)
        elif mapping_method == "sqlite":
            # SQLite database mapping, create a connection for each process
            return get_name_from_sqlite(sequence, db_path=mapping_data_or_path)
        else:
            return sequence  # Fallback for unknown methods

    # Apply the mapping function to the 'Protein' column
    # Note: progress_apply may not work well across processes
    # We rely on the overall progress bar to track chunks
    chunk["Protein"] = chunk["Protein"].apply(apply_mapping)
    return chunk


# --- Main Processing Logic ---


# Modified get_protein_name to act as a dispatcher
def get_protein_name(sequence, method, mapping_data):
    """
    Dispatches sequence-to-name mapping to the appropriate function based on method.
    """
    if method == "uniprot":
        return get_uniprot_name(sequence)
    elif method == "csv":
        return get_name_from_csv(sequence, mapping_data)  # mapping_data is the dict
    elif method == "sqlite":
        # For non-parallel processing, mapping_data is the cursor
        return get_name_from_sqlite(sequence, db_cursor=mapping_data)
    else:
        # Default or fallback: return original sequence
        return sequence


# Modified process_csv for parallel processing
def process_csv(
    input_path,
    output_path,
    mapping_method,
    mapping_data_or_path,
    chunk_size=10000,
    max_workers=None,
    log_level=logging.INFO,
):
    """
    Reads a CSV in chunks, processes chunks in parallel to replace 'Protein' column sequences with protein names,
    sorts by 'predicted_value', displays progress with tqdm, and saves to a new CSV.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed CSV file.
        mapping_method (str): The method to use for mapping ('uniprot', 'csv', 'sqlite').
        mapping_data_or_path: Data needed for the mapping method (dict for csv, DB path for sqlite, None for uniprot).
        chunk_size (int): Number of rows per chunk for memory-efficient processing.
        max_workers (int, optional): Maximum number of worker processes. Defaults to the number of CPU cores.
        log_level (int): Log level, defaults to INFO.
    """
    # Configure logger
    logger = setup_logger(log_level)

    # Set maximum number of worker processes
    if max_workers is None:
        max_workers = os.cpu_count()
        logger.info(f"Using default max_workers: {max_workers}")

    try:
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(
            f"Processing {input_path} using '{mapping_method}' mapping method with up to {max_workers} workers..."
        )

        # Warn about potential rate limits for UniProt API
        if mapping_method == "uniprot":
            logger.warning(
                "Using UniProt API in parallel processing may lead to rate limiting."
            )

        processed_chunks = []
        futures = []

        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Read chunks and submit them for processing
            chunk_iterator = pd.read_csv(input_path, chunksize=chunk_size)
            logger.info("Submitting chunks for parallel processing...")

            # Track the number of submitted chunks
            submitted_chunks = 0
            for chunk in chunk_iterator:
                submitted_chunks += 1
                # Prepare data package for the worker function
                chunk_data_package = (chunk, mapping_method, mapping_data_or_path)
                futures.append(executor.submit(process_chunk, chunk_data_package))

            if submitted_chunks == 0:
                logger.warning(
                    "Input CSV file appears to be empty or contains only a header."
                )
                pd.DataFrame().to_csv(output_path, index=False)
                logger.info(f"Saved empty processed file to {output_path}")
                return

            logger.info(f"Submitted {submitted_chunks} chunks. Waiting for results...")
            # Process results with tqdm progress bar
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing chunks",
            ):
                try:
                    processed_chunk = future.result()
                    if processed_chunk is not None:
                        processed_chunks.append(processed_chunk)
                except Exception as e:
                    # Log errors from worker processes
                    logger.error(f"Error processing a chunk: {e}")

        if not processed_chunks:
            logger.warning("No chunks were successfully processed.")
            pd.DataFrame().to_csv(output_path, index=False)
            logger.info(f"Saved empty processed file to {output_path}")
            return

        logger.info("Concatenating processed chunks...")
        df = pd.concat(processed_chunks, ignore_index=True)

        logger.info("Sorting data by 'predicted_value'...")
        df["predicted_value"] = pd.to_numeric(df["predicted_value"], errors="coerce")
        df = df.dropna(subset=["predicted_value"])
        df_sorted = df.sort_values(by="predicted_value", ascending=False)

        logger.info(f"Saving sorted data to {output_path}...")
        df_sorted.to_csv(output_path, index=False)
        logger.info("Processing complete.")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except KeyError as e:
        logger.error(f"Error: {e}")
    except pd.errors.EmptyDataError:
        logger.error(f"Error: Input file '{input_path}' is empty.")
    except sqlite3.Error as e:
        logger.error(f"SQLite Error: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network Error (UniProt): {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CSV: Replace protein sequences with names and sort by predicted_value using parallel processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to the input CSV file.")
    parser.add_argument("output_file", help="Path to save the processed CSV file.")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="Chunk size for reading CSV to manage memory usage.",
    )
    parser.add_argument(
        "--mapping-method",
        choices=["uniprot", "csv", "sqlite"],
        required=True,
        help="Method for mapping protein sequences to names.",
    )
    parser.add_argument(
        "--mapping-file",
        help="Path to the local mapping file (required for 'csv' and 'sqlite' methods)."
        " For CSV: expects 'sequence' and 'name' columns."
        " For SQLite: expects a DB file with table 'protein_map' and columns 'sequence', 'name'.",
    )
    # Add parallel processing option
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,  # Default will use os.cpu_count()
        help="Maximum number of worker processes for parallel processing.",
    )
    # Add log level parameter
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level.",
    )

    args = parser.parse_args()

    # Convert string log level to logging constant
    log_level = getattr(logging, args.log_level)

    # Configure logger
    logger = setup_logger(log_level)

    # Validate mapping file requirement
    if args.mapping_method in ["csv", "sqlite"] and not args.mapping_file:
        logger.error(
            "--mapping-file is required when --mapping-method is 'csv' or 'sqlite'."
        )
        parser.error(
            "--mapping-file is required when --mapping-method is 'csv' or 'sqlite'."
        )
    if args.mapping_method in ["csv", "sqlite"] and not os.path.exists(
        args.mapping_file
    ):
        logger.error(f"Mapping file not found: {args.mapping_file}")
        exit(1)  # Exit if mapping file is missing

    # --- Prepare mapping data based on method ---
    mapping_data_or_path = None  # Renamed variable for clarity
    db_conn = None  # To store DB connection for later closing

    try:
        if args.mapping_method == "csv":
            logger.info(f"Loading CSV mapping from {args.mapping_file}...")
            # Load CSV into a dictionary for faster lookup
            map_df = pd.read_csv(args.mapping_file, header=0)
            if "sequence" not in map_df.columns or "name" not in map_df.columns:
                msg = "Mapping CSV must contain 'sequence' and 'name' columns."
                logger.error(msg)
                raise ValueError(msg)
            # Load into dictionary to pass to worker processes
            mapping_data_or_path = pd.Series(
                map_df.name.values, index=map_df.sequence
            ).to_dict()
            logger.info(
                f"CSV mapping loaded into dictionary ({len(mapping_data_or_path)} entries)."
            )
            # Clear large DataFrame if memory is tight
            del map_df

        elif args.mapping_method == "sqlite":
            logger.info(f"Using SQLite mapping database: {args.mapping_file}")
            # For parallel processing, pass the database file path instead of a cursor
            # Check if DB and table are accessible
            try:
                conn_check = sqlite3.connect(args.mapping_file)
                cursor_check = conn_check.cursor()
                cursor_check.execute("SELECT sequence, name FROM protein_map LIMIT 1")
                conn_check.close()
                mapping_data_or_path = args.mapping_file  # Pass the path
                logger.info("SQLite DB structure is accessible.")
            except sqlite3.Error as e:
                msg = f"Error accessing table 'protein_map' or columns 'sequence', 'name': {e}"
                logger.error(msg)
                raise sqlite3.OperationalError(msg)

        elif args.mapping_method == "uniprot":
            logger.info(
                "Using UniProt API for mapping. This may be slow for large datasets."
            )
            mapping_data_or_path = None  # No pre-loading needed

        # --- Run processing ---
        process_csv(
            args.input_file,
            args.output_file,
            args.mapping_method,
            mapping_data_or_path,  # Pass dict, path, or None
            args.chunksize,
            args.max_workers,
            log_level,
        )

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except ValueError as e:
        logger.error(f"Error: {e}")
    except sqlite3.Error as e:
        logger.error(f"SQLite error during setup: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        logger.error(traceback.format_exc())
