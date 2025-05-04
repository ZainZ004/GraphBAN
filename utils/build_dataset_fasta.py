import argparse
from Bio import SeqIO
import sys
import os


def process_chem(inpute_SMILES_path):
    """
    Processes the input SMILES file to extract chemical information.
    This is a placeholder function and should be implemented based on specific requirements.

    Args:
        inpute_SMILES_path (str): Path to the input SMILES file.

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
            print(
                f"Error: SMILES string not found in file {inpute_SMILES_path}",
                file=sys.stderr,
            )
            return None
    except FileNotFoundError:
        print(
            f"Error: Input SMILES file not found at {inpute_SMILES_path}",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"An error occurred while processing the SMILES file: {e}", file=sys.stderr
        )
        return None


def process_fasta_record(record):
    """
    Processes a single SeqRecord from a FASTA file.
    Extracts ID and sequence.
    (Further processing to determine chemical/protein type
     and convert to specific format can be added here based on requirements)

    Args:
        record (SeqRecord): A Biopython SeqRecord object.

    Returns:
        tuple: A tuple containing the sequence ID and the sequence string.
               Returns None if processing fails or record is invalid.
    """
    try:
        seq_id = record.id
        sequence = str(record.seq)
        # Placeholder for future logic:
        # - Determine if it's a chemical or protein based on ID, sequence, description etc.
        # - Convert to the target data format.
        # For now, just return ID and sequence.
        return seq_id, sequence
    except Exception as e:
        print(f"Error processing record {record.id}: {e}", file=sys.stderr)
        return None


def process_large_fasta(input_fasta_path, chem_SMILES, output_file_path=None):
    """
    Reads a potentially large FASTA file record by record, processes each record,
    and writes the output incrementally.

    Args:
        input_fasta_path (str): Path to the input FASTA file.
        output_file_path (str, optional): Path to the output file.
                                          If None, prints to stdout. Defaults to None.
    """
    output_handle = None
    input_handle = None  # Initialize input_handle
    pbar = None  # Initialize pbar
    try:
        # Get total file size for progress bar
        total_size = os.path.getsize(input_fasta_path)

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
        for record in fasta_iterator:
            processed_data = process_fasta_record(record)
            if processed_data:
                seq_id, sequence = processed_data
                # Write processed data incrementally
                output_handle.write(f"{chem_SMILES},{sequence}\n")
                processed_count += 1

        # tqdm handles the final progress display, so the print statement below might be redundant for progress itself
        # print(f"\nFinished processing. Total records processed: {processed_count}", file=sys.stderr) # Keep for record count summary

    except FileNotFoundError:
        print(
            f"Error: Input FASTA file not found at {input_fasta_path}", file=sys.stderr
        )
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
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
            print(f"Output written to {output_file_path}", file=sys.stderr)
        # Print final record count summary regardless of output destination
        if "processed_count" in locals():  # Check if processing started
            print(
                f"\nFinished processing. Total records processed: {processed_count}",
                file=sys.stderr,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Process large FASTA files efficiently, converting sequences to a target format."
    )
    parser.add_argument("input_fasta", help="Path to the input FASTA file.")
    parser.add_argument("chem_SMILES", help="input SMILES to be predicted.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output file (optional). If not provided, output goes to standard output.",
        default=None,
    )
    # Add more arguments if needed for specifying chemical/protein types or output formats

    args = parser.parse_args()

    if not os.path.exists(args.input_fasta):
        print(f"Error: Input file '{args.input_fasta}' not found.", file=sys.stderr)
        sys.exit(1)
    chem_SMILES = process_chem(args.chem_SMILES)
    if not chem_SMILES:
        print(f"Error: Failed to process SMILES from '{args.chem_SMILES}'", file=sys.stderr)
        sys.exit(1)
    process_large_fasta(args.input_fasta, chem_SMILES, args.output)


if __name__ == "__main__":
    main()
