import time
import requests
import pandas as pd
import lxml.etree as ET
import re
import logging
import os
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_year(pubdate_element: Optional[ET._Element]) -> str:
    """
    Helper function to extract the year from various PubDate structures.
    Prioritizes <Year>, then tries to parse <MedlineDate>.

    Args:
        pubdate_element: The PubDate XML element or None.

    Returns:
        The extracted year as a string, or "No year found".
    """
    if pubdate_element is None:
        return "No year found"

    year_element = pubdate_element.find("Year")
    if year_element is not None and year_element.text and year_element.text.isdigit():
        return year_element.text

    medline_date_element = pubdate_element.find("MedlineDate")
    if medline_date_element is not None and medline_date_element.text:
        # Extract first 4 digits if they look like a year
        match = re.match(r"^\d{4}", medline_date_element.text)
        if match:
            return match.group(0)

    return "No year found"

def get_pubmed_data_bulk(pmid_list: List[str], api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch Title, Year, and Abstract in bulk using epost + efetch given a list of PMIDs.

    Args:
        pmid_list: A list of PubMed IDs (strings).
        api_key: Optional NCBI API key for higher rate limits.

    Returns:
        pandas.DataFrame: DataFrame with columns "PMID", "Year", "Title", "Abstract".
                         Returns an empty DataFrame if pmid_list is empty or no data found.
    """
    if not pmid_list:
        logging.warning("Input PMID list is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=["PMID", "Year", "Title", "Abstract"])

    # Ensure all PMIDs are strings
    pmid_list = [str(p) for p in pmid_list]
    logging.info(f"Attempting to fetch data for {len(pmid_list)} PMIDs...")

    # --- epost to store PMIDs ---
    epost_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi"
    epost_params: Dict[str, str] = {"db": "pubmed", "id": ",".join(pmid_list)}
    if api_key:
         epost_params["api_key"] = api_key

    try:
        epost_response = requests.post(epost_url, data=epost_params, timeout=30) # Increased timeout
        epost_response.raise_for_status()

        epost_root = ET.fromstring(epost_response.content)
        webenv_element = epost_root.find(".//WebEnv")
        query_key_element = epost_root.find(".//QueryKey")

        if webenv_element is None or query_key_element is None:
            logging.error("Could not find WebEnv or QueryKey in epost response.")
            logging.error(f"Epost response content: {epost_response.text}")
            return pd.DataFrame(columns=["PMID", "Year", "Title", "Abstract"])

        webenv = webenv_element.text
        query_key = query_key_element.text
        logging.info("Epost successful, obtained WebEnv and QueryKey.")

    except requests.RequestException as e:
        logging.error(f"Error during epost request: {e}")
        return pd.DataFrame(columns=["PMID", "Year", "Title", "Abstract"])
    except ET.XMLSyntaxError as e:
        logging.error(f"Error parsing epost XML response: {e}")
        logging.error(f"Epost response content: {epost_response.text}")
        return pd.DataFrame(columns=["PMID", "Year", "Title", "Abstract"])

    # --- efetch to retrieve full records ---
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params: Dict[str, str] = {
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retmode": "xml",
        "rettype": "abstract"
    }
    if api_key:
        efetch_params["api_key"] = api_key

    results_list: List[Dict[str, str]] = [] # Store results as a list of dictionaries

    try:
        # Adjust retmax if needed, though history server should handle large numbers
        # Add batching logic here if dealing with >10,000 PMIDs at once
        logging.info("Sending efetch request...")
        efetch_response = requests.get(efetch_url, params=efetch_params, timeout=90) # Increased timeout
        efetch_response.raise_for_status()
        logging.info("Efetch request successful.")

        efetch_root = ET.fromstring(efetch_response.content)
        articles_found_count = 0

        for article in efetch_root.xpath(".//PubmedArticle"):
            articles_found_count += 1
            pmid = "Not found"
            title = "No title found"
            year = "No year found"
            abstract_text = "No abstract found"

            pmid_element = article.find(".//PMID")
            if pmid_element is not None and pmid_element.text:
                pmid = pmid_element.text
            else:
                 logging.warning("Found an article structure with no PMID element.")
                 continue # Skip article if PMID is missing

            title_element = article.find(".//ArticleTitle")
            if title_element is not None:
                 # Handle potential complex title structures (e.g., with HTML tags)
                 title = ET.tostring(title_element, method='text', encoding='unicode').strip()
                 if not title:
                      title = "No title found" # Handle empty title tags

            # Look for PubDate in multiple potential locations
            pubdate_element = article.find(".//Article/Journal/JournalIssue/PubDate")
            if pubdate_element is None:
                 pubdate_element = article.find(".//Article/PubDate") # Check Article level too
            year = extract_year(pubdate_element)

            # Handle structured abstracts correctly
            abstract_parts = article.xpath(".//Abstract/AbstractText")
            if abstract_parts:
                processed_parts = []
                for part in abstract_parts:
                    # Get text content robustly
                    part_text = ET.tostring(part, method='text', encoding='unicode').strip()
                    label = part.get("Label")
                    if label and part_text: # Only add label if text exists
                        processed_parts.append(f"{label.upper()}: {part_text}")
                    elif part_text: # Add part if text exists, even without label
                        processed_parts.append(part_text)
                if processed_parts:
                    abstract_text = "\n".join(processed_parts) # Use newline as separator
                # If after processing, it's empty, keep default "No abstract found"
                if not abstract_text:
                    abstract_text = "No abstract found"

            results_list.append({
                "PMID": pmid,
                "Year": year,
                "Title": title,
                "Abstract": abstract_text
            })

        logging.info(f"Processed {articles_found_count} article structures found in efetch response.")
        if articles_found_count < len(pmid_list):
             logging.warning(f"Requested {len(pmid_list)} PMIDs, but received data structures for {articles_found_count}.")
             # It's possible some PMIDs are invalid or suppressed

    except requests.RequestException as e:
        logging.error(f"Error during efetch request: {e}")
        if results_list:
            logging.warning("Returning partial results due to network error.")
            # Convert partial results to DataFrame before returning
            df = pd.DataFrame(results_list, columns=["PMID", "Year", "Title", "Abstract"])
            log_missing_pmids(pmid_list, df)
            return df
        else:
            return pd.DataFrame(columns=["PMID", "Year", "Title", "Abstract"])
    except ET.XMLSyntaxError as e:
        logging.error(f"Error parsing efetch XML response: {e}")
        logging.error(f"Efetch response content (first 500 chars): {efetch_response.text[:500]}")
        if results_list:
             logging.warning("Returning partial results due to XML parsing error.")
             df = pd.DataFrame(results_list, columns=["PMID", "Year", "Title", "Abstract"])
             log_missing_pmids(pmid_list, df)
             return df
        else:
             return pd.DataFrame(columns=["PMID", "Year", "Title", "Abstract"])

    if results_list:
        df = pd.DataFrame(results_list, columns=["PMID", "Year", "Title", "Abstract"])
        log_missing_pmids(pmid_list, df) # Log missing PMIDs after successful parsing
        logging.info(f"Successfully fetched and parsed data for {len(df)} PMIDs.")
    else:
        logging.warning("Efetch successful, but no article data could be extracted.")
        df = pd.DataFrame(columns=["PMID", "Year", "Title", "Abstract"])

    return df

def log_missing_pmids(requested_pmids: List[str], result_df: pd.DataFrame):
    """Helper function to log which requested PMIDs were not found in the results."""
    found_pmids = set(result_df['PMID']) if not result_df.empty else set()
    missing_pmids = set(requested_pmids) - found_pmids
    if missing_pmids:
        logging.warning(f"Data for the following PMIDs was not found in the response: {', '.join(missing_pmids)}")

def transform_pubmed_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw PubMed data DataFrame by creating a 'text' column.

    Args:
        input_df: DataFrame with 'PMID', 'Year', 'Title', 'Abstract' columns.

    Returns:
        DataFrame with 'PMID', 'Year', 'text' columns.
    """
    if input_df.empty:
        logging.warning("Input DataFrame for transformation is empty.")
        return pd.DataFrame(columns=['PMID', 'Year', 'text'])

    logging.info("Transforming fetched data...")
    start_transform_time = time.time()

    # Ensure Title and Abstract are strings before concatenation
    input_df['Title'] = input_df['Title'].astype(str)
    input_df['Abstract'] = input_df['Abstract'].astype(str)

    # Corrected "Tittle" to "Title"
    input_df['text'] = ("# Title: " + input_df['Title'] +
                        "\n# Abstract: " + input_df['Abstract'])

    # Select and reorder final columns, using .copy() to avoid warnings
    final_df = input_df[['PMID', 'Year', 'text']].copy()

    end_transform_time = time.time()
    logging.info(f"Transformation complete in {end_transform_time - start_transform_time:.2f} seconds.")
    return final_df