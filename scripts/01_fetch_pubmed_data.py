import argparse
import logging
import time
import sys
import os

# Ensure the src directory is in the Python path
# This allowssibling imports for scripts run directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.pubmed_fetcher import get_pubmed_data_bulk, transform_pubmed_data
from src.config import NCBI_API_KEY, FETCHED_DATA_CSV_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(pmids: list[str], output_file: str, api_key: str | None):
    """
    Main function to fetch, transform, and save PubMed data.

    Args:
        pmids: List of PubMed IDs (strings).
        output_file: Path to save the final CSV file.
        api_key: Optional NCBI API key.
    """
    if not pmids:
        logging.error("No PMIDs provided. Exiting.")
        return

    logging.info(f"Starting PubMed data fetch for {len(pmids)} PMIDs...")
    start_time = time.time()

    # 1. Fetch raw data
    raw_pubmed_data_df = get_pubmed_data_bulk(pmids, api_key=api_key)

    end_fetch_time = time.time()
    logging.info(f"Raw data fetching took {end_fetch_time - start_time:.2f} seconds.")

    if raw_pubmed_data_df.empty:
        logging.warning("No data was fetched from PubMed. Skipping transformation and saving.")
        return

    logging.info(f"Successfully fetched raw data for {len(raw_pubmed_data_df)} PMIDs.")

    # 2. Transform data
    final_df = transform_pubmed_data(raw_pubmed_data_df)

    if final_df.empty:
        logging.warning("Transformation resulted in an empty DataFrame. Nothing to save.")
        return

    # 3. Save data
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"Final formatted data saved to '{output_file}'")
        logging.info("First 5 rows of saved data:")
        # Use pandas option to display more content if needed
        with pd.option_context('display.max_colwidth', 100):
             print(final_df.head().to_string())

    except Exception as e:
        logging.error(f"Error saving final data to CSV file '{output_file}': {e}", exc_info=True)

    end_time = time.time()
    logging.info(f"Total script execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process PubMed data for given PMIDs.")
    parser.add_argument(
        "--pmids",
        nargs='+', # Accepts one or more arguments
        required=True,
        help="List of PubMed IDs separated by spaces (e.g., 31091372 38320511)."
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

    main(pmids=args.pmids, output_file=args.output, api_key=api_key_to_use)

    # Example usage from command line:
    # python scripts/01_fetch_pubmed_data.py --pmids 31091372 38320511 29766772 --output ./output/my_pubmed_data.csv --use-api-key
    #
    # Or using the default PMIDs from Notebook 1 and default output path:
    # default_pmids = ['31091372', '38320511', '29766772', '38324415', '39297945', '37877587', '36765286', '35066509', '32243865', '26387030', '23873274']
    # print(f"Running with default PMIDs: {default_pmids}")
    # main(pmids=default_pmids, output_file=FETCHED_DATA_CSV_PATH, api_key=NCBI_API_KEY)