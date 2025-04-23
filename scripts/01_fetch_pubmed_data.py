# scripts/01_fetch_pubmed_data.py

import argparse
import logging
import time
import sys
import os
import re # Import regex
import pandas as pd # Import pandas for CSV reading
from typing import List, Set, Optional

# Ensure the src directory is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Project Modules
from src.data_processing.pubmed_fetcher import get_pubmed_data_bulk, transform_pubmed_data
from src.config import (
    NCBI_API_KEY,
    FETCHED_DATA_CSV_PATH,
    MEDMETA_CSV_PATH, # Need the default input path
    DEFAULT_PMID_COLUMN # Import the default column name
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_pmids_from_csv(filepath: str, column_name: str) -> List[str]:
    """
    Reads a CSV file and extracts unique PMIDs from a specified column.

    Args:
        filepath: Path to the CSV file.
        column_name: Name of the column containing references/PMIDs.

    Returns:
        A sorted list of unique PMID strings found in the column.
        Returns an empty list if the file/column is not found or no PMIDs are extracted.
    """
    logging.info(f"Attempting to extract PMIDs from '{column_name}' column in '{filepath}'...")
    if not os.path.exists(filepath):
        logging.error(f"Input CSV file not found: {filepath}")
        return []

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"Failed to read CSV file '{filepath}': {e}", exc_info=True)
        return []

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' not found in CSV file '{filepath}'. Available columns: {list(df.columns)}")
        return []

    all_pmids_raw: Set[str] = set()
    processed_rows = 0
    skipped_rows = 0

    for item in df[column_name]:
        processed_rows += 1
        if isinstance(item, str) and item.strip():
            # Find all sequences of digits in the string
            pmids_in_item = re.findall(r'\d+', item)
            # Basic validation (optional): Check if length looks reasonable for PMID
            valid_pmids_in_item = {pmid for pmid in pmids_in_item if 6 <= len(pmid) <= 9}
            all_pmids_raw.update(valid_pmids_in_item)
        elif pd.isna(item) or (isinstance(item, str) and not item.strip()):
             skipped_rows += 1
             logging.debug(f"Skipping row with empty or NaN value in '{column_name}'.")
        else:
            skipped_rows += 1
            logging.warning(f"Skipping row with unexpected data type '{type(item)}' in '{column_name}': {item}")


    unique_pmids_list = sorted(list(all_pmids_raw))
    logging.info(f"Processed {processed_rows} rows from column '{column_name}'.")
    if skipped_rows > 0:
        logging.info(f"Skipped {skipped_rows} rows due to empty, NaN, or non-string values.")
    logging.info(f"Found {len(unique_pmids_list)} unique potential PMIDs.")

    return unique_pmids_list


def main(input_csv: str, pmid_column: str, output_file: str, api_key: Optional[str]):
    """
    Main function to extract PMIDs from a CSV, fetch, transform, and save PubMed data.

    Args:
        input_csv: Path to the input CSV file containing PMIDs.
        pmid_column: Name of the column with PMIDs/references.
        output_file: Path to save the final fetched data CSV file.
        api_key: Optional NCBI API key.
    """
    overall_start_time = time.time()

    # 1. Extract PMIDs from the input file
    pmids_to_fetch = extract_pmids_from_csv(filepath=input_csv, column_name=pmid_column)

    if not pmids_to_fetch:
        logging.error("No valid PMIDs extracted from the input file. Exiting.")
        return

    # 2. Fetch raw data
    logging.info(f"Starting PubMed data fetch for {len(pmids_to_fetch)} unique PMIDs...")
    fetch_start_time = time.time()
    raw_pubmed_data_df = get_pubmed_data_bulk(pmids_to_fetch, api_key=api_key)
    fetch_end_time = time.time()
    logging.info(f"Raw data fetching took {fetch_end_time - fetch_start_time:.2f} seconds.")

    if raw_pubmed_data_df.empty:
        logging.warning("No data was fetched from PubMed. Skipping transformation and saving.")
        return

    logging.info(f"Successfully fetched raw data for {len(raw_pubmed_data_df)} PMIDs.")

    # 3. Transform data
    final_df = transform_pubmed_data(raw_pubmed_data_df)

    if final_df.empty:
        logging.warning("Transformation resulted in an empty DataFrame. Nothing to save.")
        return

    # 4. Save data
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"Final formatted data saved to '{output_file}'")
        logging.info("First 5 rows of saved data:")
        with pd.option_context('display.max_colwidth', 100):
             print(final_df.head().to_string())

    except Exception as e:
        logging.error(f"Error saving final data to CSV file '{output_file}': {e}", exc_info=True)

    overall_end_time = time.time()
    logging.info(f"Total script execution time: {overall_end_time - overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PMIDs from a CSV, fetch corresponding PubMed data, transform, and save.")
    parser.add_argument(
        "--input_csv",
        default=MEDMETA_CSV_PATH, # Use the default path from config
        help=f"Path to the input CSV file containing PMIDs (default: {MEDMETA_CSV_PATH})."
    )
    parser.add_argument(
        "--pmid_column",
        default=DEFAULT_PMID_COLUMN, # Use the default column name from config
        help=f"Name of the column in the input CSV containing PMIDs/references (default: '{DEFAULT_PMID_COLUMN}')."
    )
    parser.add_argument(
        "--output",
        default=FETCHED_DATA_CSV_PATH,
        help=f"Path to save the output CSV file (default: {FETCHED_DATA_CSV_PATH})."
    )
    parser.add_argument(
        "--use-api-key",
        action='store_true', # Flag to indicate usage of API key from env
        help="Use NCBI API key from environment variable NCBI_API_KEY."
    )

    args = parser.parse_args()

    api_key_to_use = NCBI_API_KEY if args.use_api_key else None
    if args.use_api_key and not api_key_to_use:
        logging.warning("`--use-api-key` specified, but NCBI_API_KEY environment variable not found.")

    main(
        input_csv=args.input_csv,
        pmid_column=args.pmid_column,
        output_file=args.output,
        api_key=api_key_to_use
    )

    # Example usage from command line:
    # python scripts/01_fetch_pubmed_data.py
    # python scripts/01_fetch_pubmed_data.py --input_csv ./data/MedMeta.csv --pmid_column References --output ./output/pubmed_data_final.csv --use-api-key