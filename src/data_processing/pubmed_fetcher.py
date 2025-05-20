# src/data_processing/pubmed_fetcher.py

import requests
import time
import pandas as pd
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import re

# Ensure imports from config are correct
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# --- Fetching Parameters (New/Modified) ---
FETCH_BATCH_SIZE = 200          # Max IDs per EFetch request is typically 200
FETCH_MAX_WORKERS = 10          # Number of threads for concurrent requests
FETCH_MAX_RETRIES = 5           # Max retries per batch request
FETCH_BACKOFF_FACTOR = 2        # Exponential backoff multiplier (seconds)
FETCH_TIMEOUT = 60              # Timeout for each HTTP request in seconds
FETCH_DELAY_PER_BATCH = 0.1     # Small delay between starting batch requests to be gentle
# --- NCBI ---
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "your_email@example.com") # Good practice to provide email
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# --- Default Columns ---
DEFAULT_PMID_COLUMN = 'References' # Or whatever column name you use

# Configure logging (optional, main script also configures)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _parse_pubmed_article_xml(article_xml: ET.Element) -> Dict[str, Any]:
    """
    Parses a single PubMed Article XML element and extracts relevant data,
    including the Year.

    Args:
        article_xml: An ElementTree element representing a single PubMedArticle.

    Returns:
        A dictionary containing extracted data. Keys should match desired raw DataFrame columns.
    """
    data: Dict[str, Any] = {}

    try:
        # Extract basic info
        medline_citation = article_xml.find('.//MedlineCitation')
        if medline_citation is not None:
            pmid_elem = medline_citation.find('.//PMID')
            if pmid_elem is not None:
                data['PMID'] = pmid_elem.text
            else:
                 data['PMID'] = None

            article_elem = medline_citation.find('.//Article')
            if article_elem is not None:
                # Title
                article_title_elem = article_elem.find('.//ArticleTitle')
                if article_title_elem is not None:
                    data['ArticleTitle'] = ''.join(article_title_elem.itertext()).strip()
                else:
                    data['ArticleTitle'] = None # Ensure key exists even if missing

                # Abstract
                abstract_elem = article_elem.find('.//Abstract/AbstractText')
                if abstract_elem is not None:
                    data['Abstract'] = ''.join(abstract_elem.itertext()).strip()
                else:
                    abstract_texts = article_elem.findall('.//Abstract/AbstractText')
                    if abstract_texts:
                        abstract_parts = []
                        for part in abstract_texts:
                             label = part.get('Label')
                             text = ''.join(part.itertext()).strip()
                             if label:
                                 abstract_parts.append(f"{label}: {text}")
                             else:
                                 abstract_parts.append(text)
                        data['Abstract'] = "\n".join(abstract_parts)
                    else:
                        data['Abstract'] = "No abstract found" # Ensure key exists even if missing


                # Publication Date and YEAR
                pub_date_elem = article_elem.find('.//Journal/JournalIssue/PubDate')
                if pub_date_elem is not None:
                    year = pub_date_elem.findtext('Year')
                    month = pub_date_elem.findtext('Month')
                    day = pub_date_elem.findtext('Day')
                    medline_date = pub_date_elem.findtext('MedlineDate')

                    # Store the full date string as PubDate (optional, but keeps original)
                    if year:
                        date_parts = [year]
                        if month: date_parts.append(month)
                        if day: date_parts.append(day)
                        data['PubDate'] = "-".join(date_parts)
                    elif medline_date:
                         data['PubDate'] = medline_date
                    else:
                        data['PubDate'] = None

                    # IMPORTANT: Extract and store the Year separately
                    # Prioritize <Year>, fallback to parsing MedlineDate if needed (basic attempt)
                    if year:
                         data['Year'] = year
                    elif medline_date:
                         # Try to extract a 4-digit year from MedlineDate string
                         year_match = re.search(r'\d{4}', medline_date)
                         data['Year'] = year_match.group(0) if year_match else None
                    else:
                        data['Year'] = None # Ensure key exists even if missing
                else:
                    data['PubDate'] = None
                    data['Year'] = None # Ensure key exists even if missing


                # Authors (Example: just list names)
                author_list_elem = article_elem.find('.//AuthorList')
                authors = []
                if author_list_elem is not None:
                    for author_elem in author_list_elem.findall('.//Author'):
                         last_name = author_elem.findtext('LastName')
                         fore_name = author_elem.findtext('ForeName')
                         initials = author_elem.findtext('Initials')
                         name = f"{last_name or ''} {fore_name or ''}".strip()
                         if not name and initials: # Fallback to initials if name parts missing
                             name = initials.strip()
                         if name:
                             authors.append(name)
                data['Authors'] = "; ".join(authors) if authors else None

            # MeSH Terms (Example: list main headings)
            mesh_list_elem = medline_citation.find('.//MeshHeadingList')
            mesh_terms = []
            if mesh_list_elem is not None:
                for mesh_heading in mesh_list_elem.findall('.//MeshHeading'):
                    descriptor = mesh_heading.find('.//DescriptorName')
                    if descriptor is not None:
                        mesh_terms.append(descriptor.text)
            data['MeshTerms'] = "; ".join(mesh_terms) if mesh_terms else None


        # Extract Publication Status from PubMedData if available
        pubmed_data = article_xml.find('.//PubMedData')
        if pubmed_data is not None:
             pub_status_list = pubmed_data.findall('.//PublicationStatus')
             if pub_status_list:
                 data['PublicationStatus'] = "; ".join([status.text for status in pub_status_list if status.text])
             else:
                 data['PublicationStatus'] = None
        else:
            data['PublicationStatus'] = None

        # Add PMID even if other parsing fails
        if 'PMID' not in data or data['PMID'] is None:
             pmid_fallback_elem = article_xml.find('.//PMID')
             if pmid_fallback_elem is not None:
                  data['PMID'] = pmid_fallback_elem.text
             else:
                  data['PMID'] = None # Should log a critical error if PMID isn't found

    except Exception as e:
        pmid_val = data.get('PMID', 'Unknown PMID')
        logging.error(f"Error parsing XML for PMID {pmid_val}: {e}", exc_info=False)
        data['ParsingError'] = str(e) # Add error info

    return data


def _fetch_pubmed_batch(pmid_batch: List[str], api_key: Optional[str], email: str, attempt: int = 1) -> List[Dict[str, Any]]:
    """
    Fetches PubMed data for a batch of PMIDs using EFetch and parses the XML.
    Includes retry logic.

    Args:
        pmid_batch: A list of PMID strings.
        api_key: The NCBI API key.
        email: The user's email.
        attempt: Current retry attempt number.

    Returns:
        A list of dictionaries, where each dictionary contains data for a PubMed article.
        Returns an empty list on failure after retries.
    """
    if not pmid_batch:
        return []

    pmids_str = ','.join(pmid_batch)
    url = f"{EUTILS_BASE_URL}efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmids_str,
        "retmode": "xml", # Fetch as XML for structured parsing
        "api_key": api_key,
        "email": email,
    }

    # Simple delay before request to be courteous
    time.sleep(FETCH_DELAY_PER_BATCH)

    try:
        response = requests.get(url, params=params, timeout=FETCH_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check if response content is valid XML
        if not response.content.strip().startswith(b'<'):
             logging.warning(f"Batch starting {pmid_batch[0]}: Did not receive valid XML. Response status: {response.status_code}. Content start: {response.content.strip()[:100]}")
             # Treat as a failure that might need retry, or just log and move on?
             # For now, log and return empty list for this batch
             return []

        # Parse the XML
        # EFetch XML for pubmed returns a single PubmedArticleSet containing multiple PubmedArticle elements
        root = ET.fromstring(response.content)
        parsed_data = []
        for article_elem in root.findall('.//PubmedArticle'):
            article_data = _parse_pubmed_article_xml(article_elem)
            if article_data.get('PMID') is not None: # Only add if PMID was successfully identified
                parsed_data.append(article_data)
            else:
                logging.warning(f"Failed to parse article XML without identifiable PMID in batch starting {pmid_batch[0]}.")

        logging.debug(f"Successfully fetched and parsed batch starting {pmid_batch[0]}")
        return parsed_data

    except requests.exceptions.Timeout:
        logging.warning(f"Batch starting {pmid_batch[0]}: Request timed out (attempt {attempt}).")
        if attempt < FETCH_MAX_RETRIES:
            wait_time = FETCH_BACKOFF_FACTOR * attempt
            logging.warning(f"Retrying batch starting {pmid_batch[0]} after {wait_time}s.")
            time.sleep(wait_time)
            return _fetch_pubmed_batch(pmid_batch, api_key, email, attempt + 1)
        else:
            logging.error(f"Batch starting {pmid_batch[0]}: Max retries exceeded due to timeout.")
            return []
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        logging.warning(f"Batch starting {pmid_batch[0]}: HTTP error {status_code} (attempt {attempt}).")
        if status_code in {429, 500, 502, 503, 504} and attempt < FETCH_MAX_RETRIES:
             wait_time = FETCH_BACKOFF_FACTOR * attempt
             logging.warning(f"Retrying batch starting {pmid_batch[0]} after {wait_time}s.")
             time.sleep(wait_time)
             return _fetch_pubmed_batch(pmid_batch, api_key, email, attempt + 1)
        else:
            logging.error(f"Batch starting {pmid_batch[0]}: Non-retryable HTTP error {status_code} or max retries exceeded.")
            logging.error(f"Response text: {e.response.text}")
            return []
    except requests.exceptions.RequestException as e:
        logging.warning(f"Batch starting {pmid_batch[0]}: Request exception: {e} (attempt {attempt}).")
        if attempt < FETCH_MAX_RETRIES:
            wait_time = FETCH_BACKOFF_FACTOR * attempt
            logging.warning(f"Retrying batch starting {pmid_batch[0]} after {wait_time}s.")
            time.sleep(wait_time)
            return _fetch_pubmed_batch(pmid_batch, api_key, email, attempt + 1)
        else:
            logging.error(f"Batch starting {pmid_batch[0]}: Max retries exceeded due to request exception.")
            return []
    except ET.ParseError as e:
         logging.error(f"Batch starting {pmid_batch[0]}: Failed to parse XML: {e}", exc_info=False)
         return []
    except Exception as e:
        logging.error(f"Batch starting {pmid_batch[0]}: An unexpected error occurred during fetch/parse: {e}", exc_info=False)
        return []


def get_pubmed_data_bulk(pmids: List[str], api_key: Optional[str]) -> pd.DataFrame:
    """
    Fetches PubMed data for a list of PMIDs in batches using multiple threads.

    Args:
        pmids: A list of PMID strings.
        api_key: The NCBI API key.

    Returns:
        A pandas DataFrame containing the fetched and partially parsed data.
        Returns an empty DataFrame if no data is fetched.
    """
    if not pmids:
        logging.info("No PMIDs provided for bulk fetching.")
        return pd.DataFrame()

    # Ensure PMIDs are strings
    pmids = [str(p) for p in pmids]

    logging.info(f"Starting bulk fetch for {len(pmids)} PMIDs in batches of {FETCH_BATCH_SIZE} using {FETCH_MAX_WORKERS} workers.")

    pmid_batches = [pmids[i:i + FETCH_BATCH_SIZE] for i in range(0, len(pmids), FETCH_BATCH_SIZE)]
    all_articles_data: List[Dict[str, Any]] = []
    futures = []

    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        for batch in pmid_batches:
            futures.append(executor.submit(_fetch_pubmed_batch, batch, api_key, NCBI_EMAIL))

        # Use tqdm to show progress of completed futures
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching PubMed batches"):
            try:
                batch_results = future.result()
                if batch_results:
                    all_articles_data.extend(batch_results)
            except Exception as exc:
                # This catches exceptions that occurred during the execution of _fetch_pubmed_batch
                # that weren't already handled and logged within _fetch_pubmed_batch itself.
                logging.error(f'Batch processing generated an exception: {exc}', exc_info=False)

    if not all_articles_data:
        logging.warning("No PubMed data was successfully fetched for any PMID batch.")
        return pd.DataFrame()

    # Create DataFrame from the collected data
    # The columns will be the union of all keys found in the dictionaries
    raw_df = pd.DataFrame(all_articles_data)

    # Ensure PMID column exists and is the first column for clarity
    if 'PMID' in raw_df.columns:
         # Reorder columns to put PMID first
         cols = raw_df.columns.tolist()
         cols.remove('PMID')
         raw_df = raw_df[['PMID'] + cols]
    else:
         logging.warning("PMID column not found in fetched data after parsing.")


    logging.info(f"Finished bulk fetch. Successfully processed data for {len(raw_df)} articles.")
    # Note: len(raw_df) might be less than len(pmids) due to failed fetches or parsing errors

    return raw_df

def log_missing_pmids(requested_pmids: List[str], result_df: pd.DataFrame):
    """Helper function to log which requested PMIDs were not found in the results."""
    found_pmids = set(result_df['PMID']) if not result_df.empty else set()
    missing_pmids = set(requested_pmids) - found_pmids
    if missing_pmids:
        logging.warning(f"Data for the following PMIDs was not found in the response: {', '.join(missing_pmids)}")

def transform_pubmed_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw PubMed DataFrame by creating a 'text' column
    and selecting/renaming columns for the final output.

    Args:
        raw_df: DataFrame containing raw fetched and parsed data
                (expected columns: 'PMID', 'Year', 'ArticleTitle', 'Abstract').

    Returns:
        DataFrame with 'PMID', 'Year', 'text' columns (or others as defined).
    """
    logging.info("Starting data transformation...")
    start_transform_time = time.time()

    if raw_df.empty:
        logging.warning("Raw DataFrame for transformation is empty.")
        # Return an empty DataFrame with the *expected* final columns
        return pd.DataFrame(columns=['PMID', 'Year', 'text'])

    # Check if essential columns exist, add them if not (e.g., due to total parsing failure)
    required_cols = ['PMID', 'Year', 'ArticleTitle', 'Abstract']
    for col in required_cols:
        if col not in raw_df.columns:
            logging.warning(f"Column '{col}' missing in raw data. Adding with None values.")
            raw_df[col] = None

    # Ensure Title and Abstract are strings before concatenation
    # Use .fillna('') to safely handle missing or None values
    raw_df['ArticleTitle'] = raw_df['ArticleTitle'].fillna('').astype(str)
    raw_df['Abstract'] = raw_df['Abstract'].fillna('').astype(str)

    # Create the 'text' column using the correct column names
    # Corrected "Tittle" to "Title" here
    raw_df['text'] = ("# Title: " + raw_df['ArticleTitle'] +
                      "\n# Abstract: " + raw_df['Abstract'])

    # Ensure Year is string/object type for consistent output if mixed types occurred
    raw_df['Year'] = raw_df['Year'].astype(str).replace('None', pd.NA) # Replace 'None' string with actual NA

    # Select and reorder final columns
    # Use .copy() to avoid potential SettingWithCopyWarning
    final_columns = ['PMID', 'Year', 'text']

    # Ensure selected columns exist in the DataFrame before selecting
    # (Handles cases where even PMID might be missing from raw_df somehow)
    cols_to_select = [col for col in final_columns if col in raw_df.columns]

    # If essential columns are missing, this might result in an empty or partial df
    if not all(col in raw_df.columns for col in final_columns):
         logging.warning("Some final columns ('PMID', 'Year', 'text') missing after transformation steps.")
         # Decide how to handle: proceed with available, or return empty?
         # Let's proceed, but log columns present
         logging.warning(f"Columns available for final selection: {raw_df.columns.tolist()}")


    # Select the columns, ensuring 'PMID', 'Year', 'text' are present if possible
    # Use a robust selection method
    output_df = pd.DataFrame()
    if 'PMID' in raw_df.columns and 'Year' in raw_df.columns and 'text' in raw_df.columns:
         output_df = raw_df[['PMID', 'Year', 'text']].copy()
    else:
         logging.error("Could not create final DataFrame with 'PMID', 'Year', 'text' columns.")
         # Fallback: try to include any of these that exist
         fallback_cols = [col for col in ['PMID', 'Year', 'text'] if col in raw_df.columns]
         if fallback_cols:
              output_df = raw_df[fallback_cols].copy()
         else:
              output_df = pd.DataFrame(columns=['PMID', 'Year', 'text']) # Return empty with correct columns


    end_transform_time = time.time()
    logging.info(f"Transformation complete in {end_transform_time - start_transform_time:.2f} seconds.")
    return output_df