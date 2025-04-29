import requests
import time
import os
import pandas as pd
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import subprocess
from tqdm import tqdm

NCBI_API_KEY = "..."
EMAIL = "..."
OUTPUT_DIR = "data_pubmed"
SEARCH_TERM = (
    '(("Meta-Analysis"[Publication Type]) OR ("Systematic Review"[Publication Type])) '
    'AND ("2018/01/01"[Date - Publication] : "2025/12/31"[Date - Publication]) '
    'AND (pubmed pmc open access[Filter] OR pubmed pmc auth manuscript[Filter])'
)
ID_CONV_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
ID_CONV_MAX_RETRIES = 5
ID_CONV_BACKOFF_FACTOR = 2
ID_CONV_BATCH_SIZE = 150
ID_CONV_WORKERS = 5
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
XML_FETCH_WORKERS = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def search_pmids_edirect(search_term, api_key, email):
    """
    Search PubMed and retrieve all PMIDs using EDirect commands via subprocess.
    Handles >10k limit automatically.

    Args:
        search_term (str): The PubMed query string.
        api_key (str): NCBI API key.
        email (str): User email for NCBI.

    Returns:
        list: A list of PMID strings, or an empty list on failure.

    Requires:
        EDirect must be installed and accessible in the system PATH.
        The API key should ideally be configured for EDirect, but we also
        pass it via environment variable for this specific call.
    """
    all_pmids = []

    command = f"esearch -db pubmed -query '{search_term}' | efetch -format uid"

    logging.info(f"Executing EDirect command to retrieve PMIDs: {command}")

    # Prepare environment for the subprocess, ensuring API key is set
    process_env = os.environ.copy()
    process_env["NCBI_API_KEY"] = api_key
    # EDirect doesn't explicitly use email via env var usually, relies on config
    # process_env["EMAIL"] = email # Unlikely to be used by EDirect directly
    edirect_path = os.path.expanduser('~/edirect')
    original_path = process_env.get('PATH', '')
    process_env['PATH'] = f"{edirect_path}:{original_path}"
    logging.debug(f"Using PATH for subprocess: {process_env['PATH']}")
    
    try:
        # Run the command
        result = subprocess.run(
            command,
            shell=True,          # Needed for the pipe '|'
            capture_output=True, # Capture stdout and stderr
            text=True,           # Decode output as text (UTF-8 default)
            check=True,          # Raise CalledProcessError on non-zero exit code
            env=process_env      # Pass the environment with API key
        )

        # Process the output (PMIDs, one per line)
        output_lines = result.stdout.strip().split('\n')
        all_pmids = [line for line in output_lines if line.isdigit()] # Basic check for PMID format

        logging.info(f"EDirect command successful. Retrieved {len(all_pmids)} PMIDs.")
       
        if result.stderr:
             logging.debug(f"EDirect stderr (might contain progress/warnings): {result.stderr.strip()}")

    except FileNotFoundError:
        logging.error("EDirect command failed: 'esearch' or 'efetch' not found.")
        logging.error("Please ensure Entrez Direct (EDirect) is installed and in your system PATH.")
        return []
    except subprocess.CalledProcessError as e:
        logging.error(f"EDirect command failed with exit code {e.returncode}.")
        logging.error(f"Command: {e.cmd}")
        logging.error(f"Stderr: {e.stderr.strip()}")
        logging.error(f"Stdout: {e.stdout.strip()}") 
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while running EDirect: {e}")
        return []

    return all_pmids


def fetch_ids_batch(pmid_batch, attempt=1):
    """Fetch PMCIDs from PMIDs using batch request with retries (ID Converter API)."""
    params = {
        'format': 'json',
        'ids': ','.join(pmid_batch),
        'api_key': NCBI_API_KEY  
    }

    try:
        response = requests.get(ID_CONV_BASE_URL, params=params, timeout=20) # Increased timeout slightly

        if response.status_code == 200:
            data = response.json()
            # Ensure correct mapping even if API returns records out of order (though unlikely for this API)
            # Create a temp dict from response
            pmcid_map = {rec.get('pmid'): rec.get('pmcid')
                         for rec in data.get('records', []) if rec.get('pmid')}
            # Map back to original batch order
            return [(pmid, pmcid_map.get(pmid, 'Not Found')) for pmid in pmid_batch]

        elif response.status_code == 429: # Rate limit
            wait_time = ID_CONV_BACKOFF_FACTOR * attempt
            logging.warning(f"Rate limit exceeded! Retrying batch starting with {pmid_batch[0]} after {wait_time}s (attempt {attempt})...")
            time.sleep(wait_time)
            if attempt < ID_CONV_MAX_RETRIES:
                return fetch_ids_batch(pmid_batch, attempt + 1)
            else:
                logging.error(f"Max retries exceeded for batch starting with {pmid_batch[0]} due to rate limiting.")
                return [(pmid, 'RateLimitError') for pmid in pmid_batch]

        elif response.status_code in {500, 502, 503, 504}: # Server errors
            wait_time = ID_CONV_BACKOFF_FACTOR * attempt
            logging.warning(f"Server error {response.status_code}! Retrying batch starting with {pmid_batch[0]} after {wait_time}s (attempt {attempt})...")
            time.sleep(wait_time)
            if attempt < ID_CONV_MAX_RETRIES:
                return fetch_ids_batch(pmid_batch, attempt + 1)
            else:
                logging.error(f"Max retries exceeded for batch starting with {pmid_batch[0]} due to server error {response.status_code}.")
                return [(pmid, f'ServerError_{response.status_code}') for pmid in pmid_batch]
        else:
            logging.error(f"ID Converter request failed for batch starting with {pmid_batch[0]} with status {response.status_code}: {response.text}")
            return [(pmid, f'HTTPError_{response.status_code}') for pmid in pmid_batch]

    except requests.RequestException as e:
        logging.error(f"ID Converter request exception for batch starting with {pmid_batch[0]}: {e}")
        if attempt < ID_CONV_MAX_RETRIES:
            wait_time = ID_CONV_BACKOFF_FACTOR * attempt
            logging.warning(f"Retrying batch starting with {pmid_batch[0]} after {wait_time}s (attempt {attempt})...")
            time.sleep(wait_time)
            return fetch_ids_batch(pmid_batch, attempt + 1)
        else:
             logging.error(f"Max retries exceeded for batch starting with {pmid_batch[0]} due to RequestException.")
             return [(pmid, 'RequestException') for pmid in pmid_batch]

    # Fallback if something unexpected happens before returning
    return [(pmid, 'UnknownError') for pmid in pmid_batch]


def pmids_to_pmcids(pmids):
    """
    Convert a list of PMIDs to PMCIDs using the ID Converter API with threading and retries.
    Returns a Pandas DataFrame with 'PMID' and 'PMCID' columns.
    """
    start_time = time.time()
    logging.info(f"Starting PMID to PMCID conversion for {len(pmids)} PMIDs...")

    if not pmids:
        logging.warning("No PMIDs provided for conversion.")
        return pd.DataFrame(columns=['PMID', 'PMCID'])

    # Ensure PMIDs are strings
    pmids = [str(p) for p in pmids]

    results = []
    pmid_batches = [pmids[i:i + ID_CONV_BATCH_SIZE] for i in range(0, len(pmids), ID_CONV_BATCH_SIZE)]
    logging.info(f"Processing {len(pmids)} PMIDs in {len(pmid_batches)} batches using {ID_CONV_WORKERS} workers.")

    with ThreadPoolExecutor(max_workers=ID_CONV_WORKERS) as executor:
        # executor.map submits tasks and returns results in the order tasks were submitted
        results_list = list(executor.map(fetch_ids_batch, pmid_batches))

    # Flatten the list of lists
    results = [item for sublist in results_list if sublist for item in sublist] # Added check for empty sublist

    result_df = pd.DataFrame(results, columns=['PMID', 'PMCID'])

    # --- Analysis of Results ---
    total_processed = len(result_df)
    found_count = len(result_df[result_df['PMCID'].str.startswith('PMC', na=False)])
    not_found_count = len(result_df[result_df['PMCID'] == 'Not Found'])
    error_count = total_processed - found_count - not_found_count

    logging.info(f"PMID->PMCID Conversion Results: Total={total_processed}, Found={found_count}, Not Found={not_found_count}, Errors={error_count}")

    # Save results to CSV
    try:
        csv_filename = 'pmid_to_pmcids_results.csv'
        result_df.to_csv(csv_filename, index=False)
        logging.info(f"PMID to PMCID mapping saved to '{csv_filename}'")
    except IOError as e:
        logging.warning(f"Could not save PMID to PMCID mapping CSV: {e}")

    end_time = time.time()
    logging.info(f"PMID to PMCID conversion completed in {round(end_time - start_time, 2)} seconds.")

    return result_df

def fetch_fulltext_xml(pmcid, api_key, email):
    """
    Fetch the full-text XML for a given PMCID from PMC using EFetch.
    Returns a tuple: (pmcid, xml_data or None).
    """
    url = f"{EUTILS_BASE_URL}efetch.fcgi"
    params = { "db": "pmc",
               "id": pmcid,
               "retmode": "xml",
               "rettype": "xml",
               "api_key": api_key,
               "email": email }
    # Keep the small delay to respect NCBI per-second limits, even with threads
    time.sleep(0.11) # ~9 requests/sec/thread max

    try:
        response = requests.get(url, params=params, timeout=90) # Increased timeout for potentially large files
        response.raise_for_status()
        if response.content.strip().startswith(b'<'):
            logging.debug(f"Successfully fetched XML for {pmcid}")
            return pmcid, response.text # Return tuple on success
        else:
            logging.warning(f"EFetch for {pmcid} did not return valid XML. Status: {response.status_code}.")
            return pmcid, None # Return tuple on failure (invalid XML)
    except requests.exceptions.RequestException as e:
        logging.error(f"EFetch request failed for {pmcid}: {e}")
        return pmcid, None # Return tuple on failure (request error)
    except Exception as e:
        logging.error(f"An unexpected error occurred during EFetch for {pmcid}: {e}")
        return pmcid, None # Return tuple on failure (other error)


def main(debug=False):
    """Main function using EDirect, threaded ID conversion, and threaded XML fetch."""
    overall_start_time = time.time()
    try:
        ensure_output_dir(OUTPUT_DIR)

        # Step 1: Search PMIDs using EDirect
        logging.info("--- Starting PMID retrieval using EDirect ---")
        pmids = search_pmids_edirect(SEARCH_TERM, NCBI_API_KEY, EMAIL)
        if not pmids:
            logging.error("No PMIDs retrieved via EDirect. Exiting.")
            return
        logging.info(f"--- Successfully retrieved {len(pmids)} PMIDs using EDirect ---")

        if debug and len(pmids) > 20: # Increase debug limit slightly if needed
            logging.warning(f"Debug mode: Limiting processing from {len(pmids)} to the first 20 PMIDs.")
            pmids = pmids[:20]

        # Step 2: Convert PMIDs to PMCIDs (already threaded)
        pmcids_df = pmids_to_pmcids(pmids)
        if pmcids_df.empty:
            logging.warning("PMID to PMCID conversion resulted in an empty DataFrame. Exiting.")
            return

        # Filter for valid PMCIDs *before* starting XML fetch
        valid_pmcids_df = pmcids_df[pmcids_df['PMCID'].str.startswith('PMC', na=False)].copy()
        if valid_pmcids_df.empty:
            logging.warning("No valid PMCIDs found after conversion. Cannot fetch full text.")
            return

        pmcid_list = valid_pmcids_df['PMCID'].tolist()
        logging.info(f"Found {len(pmcid_list)} valid PMCIDs to fetch full text for.")

        # Step 3: Fetch and save full-text XML using ThreadPoolExecutor
        logging.info(f"--- Starting parallel XML fetch using {XML_FETCH_WORKERS} workers ---")
        fetched_count = 0
        error_count = 0
        futures = []

        with ThreadPoolExecutor(max_workers=XML_FETCH_WORKERS) as executor:
            # Submit all fetch tasks
            for pmcid in pmcid_list:
                futures.append(executor.submit(fetch_fulltext_xml, pmcid, NCBI_API_KEY, EMAIL))

            # Process results as they complete using as_completed and tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching XML"):
                try:
                    pmcid_result, xml_data = future.result() # Unpack the tuple

                    if xml_data:
                        filename = os.path.join(OUTPUT_DIR, f"{pmcid_result}_fulltext.xml")
                        try:
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write(xml_data)
                            # Log success only after successful write
                            # logging.info(f"Successfully saved full-text XML to {filename}") # Reduce log verbosity
                            fetched_count += 1
                        except IOError as e:
                            logging.error(f"Error writing file {filename}: {e}")
                            error_count += 1
                        except Exception as e:
                             logging.error(f"An unexpected error occurred writing file {filename}: {e}")
                             error_count += 1
                    else:
                        # Log failure if xml_data is None (fetch failed or returned invalid XML)
                        # logging.warning(f"Failed to fetch or received invalid XML for PMCID: {pmcid_result}") # Already logged in fetch function
                        error_count += 1

                except Exception as exc:
                    # This catches errors raised *during* the future's execution
                    # if not caught inside fetch_fulltext_xml, or errors from future.result() itself.
                    # We don't know which PMCID it was for easily here unless fetch_fulltext_xml *always* returns a tuple.
                    logging.error(f"An exception occurred processing a future result: {exc}")
                    error_count += 1 # Count it as an error

        logging.info(f"Finished fetching full text. Successful: {fetched_count}, Errors/Not Found: {error_count} (out of {len(pmcid_list)} attempts)")

    except Exception as e:
        logging.exception(f"An critical error occurred in the main process: {e}")
    finally:
        overall_end_time = time.time()
        logging.info(f"Total execution time: {round(overall_end_time - overall_start_time, 2)} seconds.")

if __name__ == "__main__":
    main(debug=False)