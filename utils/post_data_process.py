import pandas as pd
from tqdm import tqdm
import argparse
import os
import requests  # Added for UniProt API
import sqlite3  # Added for SQLite
import time  # Added for potential API rate limiting

# --- Mapping Functions ---


def get_uniprot_name(sequence, retries=3, delay=1):
    """Fetches protein name from UniProt using REST API."""
    # Note: UniProt API usage policies should be reviewed for bulk queries.
    # This is a basic example and might need adjustments (e.g., batching, specific API endpoints).
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f'sequence:"{sequence}"',  # Search by exact sequence
        "fields": "protein_name",
        "format": "tsv",
        "size": 1,  # Expecting one primary match
    }
    for attempt in range(retries):
        try:
            response = requests.get(
                base_url, params=params, timeout=10
            )  # Added timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            lines = response.text.strip().split("\\n")
            if (
                len(lines) > 1 and lines[1]
            ):  # Check if we got a result line after the header
                # Assuming the name is the first column after the header
                return lines[1].split("\\t")[0]  # Extract the protein name
            else:
                return sequence  # Return original sequence if not found
        except requests.exceptions.RequestException as e:
            print(
                f"Warning: UniProt request failed (attempt {attempt + 1}/{retries}): {e}"
            )
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                print(
                    f"Warning: Could not fetch name for sequence {sequence[:10]}... from UniProt after {retries} attempts."
                )
                return sequence  # Return original sequence after all retries fail
    return sequence  # Fallback


def get_name_from_csv(sequence, mapping_dict):
    """Looks up protein name from a pre-loaded CSV mapping dictionary."""
    return mapping_dict.get(sequence, sequence)


def get_name_from_sqlite(sequence, db_cursor):
    """Looks up protein name from a SQLite database."""
    try:
        # Assuming a table named 'protein_map' with columns 'sequence' and 'name'
        # Make sure the 'sequence' column has an index for performance
        db_cursor.execute(
            "SELECT name FROM protein_map WHERE sequence = ?", (sequence,)
        )
        result = db_cursor.fetchone()
        return result[0] if result else sequence
    except sqlite3.Error as e:
        print(f"Warning: SQLite lookup failed for sequence {sequence[:10]}...: {e}")
        return sequence  # Return original sequence on DB error


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
        return get_name_from_sqlite(
            sequence, mapping_data
        )  # mapping_data is the cursor
    else:
        # Default or fallback: return original sequence if method is unknown
        print(
            f"Warning: Unknown mapping method '{method}'. Returning original sequence."
        )
        return sequence


# Modified process_csv to accept mapping method and data
def process_csv(
    input_path, output_path, mapping_method, mapping_data, chunk_size=10000
):
    """
    Reads a CSV, replaces protein sequences with names using the specified method,
    sorts by 'predicted_value', displays progress with tqdm, and saves to a new CSV.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed CSV file.
        mapping_method (str): The method to use for mapping ('uniprot', 'csv', 'sqlite').
        mapping_data: Data needed for the mapping method (dict for csv, cursor for sqlite, None for uniprot).
        chunk_size (int): Number of rows per chunk for memory-efficient processing.
    """
    try:
        # ... (rest of the initial checks and setup remain the same) ...
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found at {input_path}")

        total_size = os.path.getsize(input_path)
        print(f"Processing {input_path} using '{mapping_method}' mapping...")

        processed_chunks = []
        print("Reading and processing CSV in chunks...")
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Reading CSV"
        ) as pbar:
            for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                if (
                    "Protein" not in chunk.columns
                    or "predicted_value" not in chunk.columns
                ):
                    raise KeyError(
                        "Required columns 'Protein' and/or 'predicted_value' not found in the CSV."
                    )

                # Apply the selected mapping function
                # Using a lambda to pass the method and data correctly
                tqdm.pandas(
                    desc=f"Mapping Names ({mapping_method})"
                )  # Add progress for mapping step
                chunk["Protein"] = chunk["Protein"].progress_apply(
                    lambda seq: get_protein_name(seq, mapping_method, mapping_data)
                )

                processed_chunks.append(chunk)
                pbar.update(chunk.memory_usage(index=True, deep=True).sum())

        # ... (rest of the concatenation, sorting, saving, and error handling remain the same) ...
        if not processed_chunks:
            print("Warning: Input CSV file is empty or contains no data.")
            pd.DataFrame().to_csv(output_path, index=False)
            print(f"Empty processed file saved to {output_path}")
            return

        print("Concatenating processed chunks...")
        df = pd.concat(processed_chunks, ignore_index=True)

        print("Sorting data by 'predicted_value'...")
        df["predicted_value"] = pd.to_numeric(df["predicted_value"], errors="coerce")
        df = df.dropna(subset=["predicted_value"])
        df_sorted = df.sort_values(by="predicted_value", ascending=False)

        print(f"Saving sorted data to {output_path}...")
        df_sorted.to_csv(output_path, index=False)
        print("Processing complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Error: {e}")
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_path}' is empty.")
    except sqlite3.Error as e:  # Added specific SQLite error handling
        print(f"SQLite Error: {e}")
    except (
        requests.exceptions.RequestException
    ) as e:  # Added specific Request error handling
        print(f"Network Error (UniProt): {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CSV: Replace protein sequences with names and sort by predicted value.",
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
    # New arguments for mapping method
    parser.add_argument(
        "--mapping-method",
        choices=["uniprot", "csv", "sqlite"],
        required=True,  # Make method selection mandatory
        help="Method for mapping protein sequences to names.",
    )
    parser.add_argument(
        "--mapping-file",
        help="Path to the local mapping file (required for 'csv' and 'sqlite' methods)."
        " For CSV: expects columns 'sequence' and 'name'."
        " For SQLite: expects a DB file with table 'protein_map' and columns 'sequence', 'name'.",
    )

    args = parser.parse_args()

    # Validate mapping file requirement
    if args.mapping_method in ["csv", "sqlite"] and not args.mapping_file:
        parser.error(
            "--mapping-file is required when --mapping-method is 'csv' or 'sqlite'."
        )
    if args.mapping_method in ["csv", "sqlite"] and not os.path.exists(
        args.mapping_file
    ):
        print(f"Error: Mapping file not found at {args.mapping_file}")
        exit(1)  # Exit if mapping file is missing

    # --- Prepare mapping data based on method ---
    mapping_data = None
    db_conn = None  # To store DB connection for later closing
    db_cursor = None

    try:
        if args.mapping_method == "csv":
            print(f"Loading CSV mapping from {args.mapping_file}...")
            # Load CSV into a dictionary for faster lookup
            # Assumes first column is sequence, second is name
            map_df = pd.read_csv(args.mapping_file, header=0)  # Adjust header as needed
            if "sequence" not in map_df.columns or "name" not in map_df.columns:
                raise ValueError(
                    "Mapping CSV must contain 'sequence' and 'name' columns."
                )
            # Use pandas Series.to_dict for potentially better memory usage than df.to_dict
            mapping_data = pd.Series(
                map_df.name.values, index=map_df.sequence
            ).to_dict()
            print("CSV mapping loaded.")
        elif args.mapping_method == "sqlite":
            print(f"Connecting to SQLite mapping database {args.mapping_file}...")
            db_conn = sqlite3.connect(args.mapping_file)
            db_cursor = db_conn.cursor()
            # Check if table and columns exist (optional but good practice)
            try:
                db_cursor.execute("SELECT sequence, name FROM protein_map LIMIT 1")
            except sqlite3.OperationalError as e:
                raise sqlite3.OperationalError(
                    f"Error accessing table 'protein_map' or columns 'sequence', 'name' in DB: {e}"
                )

            mapping_data = db_cursor  # Pass the cursor
            print("SQLite DB connected.")
        elif args.mapping_method == "uniprot":
            print(
                "Using UniProt API for mapping. This might be slow for large datasets."
            )
            mapping_data = None  # No pre-loading needed

        # --- Run Processing ---
        process_csv(
            args.input_file,
            args.output_file,
            args.mapping_method,
            mapping_data,
            args.chunksize,
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:  # Catch potential errors during mapping data loading
        print(f"Error: {e}")
    except sqlite3.Error as e:
        print(f"SQLite Error during setup: {e}")
    except Exception as e:  # Catch any other unexpected error during setup
        print(f"An unexpected error occurred during setup: {e}")
    finally:
        # --- Cleanup ---
        if db_conn:
            print("Closing SQLite database connection.")
            db_conn.close()
