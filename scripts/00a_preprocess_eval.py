import os
import csv
from lxml import etree
import logging
from tqdm import tqdm

# --- Configuration ---
DATA_FOLDER = 'data_pubmed'
OUTPUT_CSV = 'meta_analysis_summary.csv'
DEBUG_MODE = False  # Set this to False for production mode (tqdm, errors only)
                    # Set this to True for debug mode (detailed logs, no tqdm)
DEBUG_LIMIT = 5
CSV_HEADER = [
    'Number',
    'Meta Analysis Name',
    'URL',
    'PMID',
    'Category',
    'Conclusion',
    'Date (Year)',
    'References', # Adjusted based on realistic extraction
    'Number of Refs'
]

# --- Conditional Logging Setup ---
if DEBUG_MODE:
    log_level = logging.INFO
    log_format = '%(levelname)s:%(name)s:%(message)s' # Include logger name for clarity
    # In debug mode, we might want debug messages from specific parts
    logging.getLogger().setLevel(logging.DEBUG) # Set root logger level
    # Keep specific loggers less verbose if needed, e.g.:
    # logging.getLogger('lxml').setLevel(logging.INFO)
else:
    # Production mode: Only log errors
    log_level = logging.ERROR
    log_format = '%(levelname)s: %(message)s'

logging.basicConfig(level=log_level, format=log_format)
# Get a specific logger for our application messages if needed
logger = logging.getLogger('MetaAnalysisScript')
# Set the level for our specific logger based on the global setting
logger.setLevel(log_level if not DEBUG_MODE else logging.INFO)
# If you want debug messages from this script in debug mode:
if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)


# --- Helper Functions ---
# (Keep helper functions as they are, logging calls inside will respect the configured level)
# Minor change: Use the specific logger instance for better context control
def safe_xpath_get_text(element, path, default=''):
    """Safely extracts text using XPath, returning default if not found."""
    try:
        result = element.xpath(path)
        if result:
            return ' '.join(node.strip() for node in result if isinstance(node, str) or hasattr(node, 'strip')).strip()
        else:
            return default
    except Exception as e:
        logger.warning(f"XPath error for path '{path}': {e}") # Will only show if level allows WARNING
        return default

def get_conclusion(root):
    """Attempts to find the conclusion text using different common structures."""
    conclusion = safe_xpath_get_text(root, ".//sec[@sec-type='conclusions']//p//text()")
    if conclusion:
        logger.debug("Conclusion found using sec[@sec-type='conclusions']") # Debug level
        return conclusion

    conclusion = safe_xpath_get_text(root, ".//sec[./title[contains(translate(normalize-space(.), 'CONCLUSION', 'conclusion'), 'conclusion')]]//p//text()")
    if conclusion:
        logger.debug("Conclusion found using //body//sec//title") # Debug level
        return conclusion

    conclusion = safe_xpath_get_text(root, ".//abstract//sec[./title[contains(translate(text(), 'CONCLUSIONS', 'conclusions'), 'conclusions')]]//p//text()")
    if conclusion:
        logger.debug("Conclusion found using abstract//sec//title") # Debug level
        return conclusion

    # conclusion = safe_xpath_get_text(root, ".//abstract//p[last()]//text()")
    # if conclusion:
    #     logger.debug("Conclusion found using abstract//p[last()] fallback") # Debug level
    #     return conclusion

    # logger.warning("Conclusion not found using standard methods.") # Warning level
    return ''

def get_year(root):
    """Attempts to find the publication year, preferring print pub date."""
    year = safe_xpath_get_text(root, ".//pub-date[@pub-type='ppub']/year/text()")
    if year: return year
    year = safe_xpath_get_text(root, ".//pub-date[@pub-type='epub']/year/text()")
    if year: return year
    year = safe_xpath_get_text(root, ".//pub-date/year/text()")
    if year: return year
    logger.warning("Publication year not found.") # Warning level
    return ''

def get_table_references(root):
    """
    Finds the first table whose caption contains 'characteristics'.
    Extracts PMIDs of references cited (<xref>) within that table.
    Returns a tuple: (pmids_string_or_flag, count).
    Handles mismatch logging/flagging based on DEBUG_MODE.
    """
    target_rids = set()
    found_target_table = False
    num_xrefs_in_table = 0

    try:
        table_wrappers = root.xpath(".//table-wrap")
        logger.debug(f"Found {len(table_wrappers)} table-wrap elements.") # Debug level

        for table_wrap in table_wrappers:
            label_text = safe_xpath_get_text(table_wrap, "./label/text()").lower()
            caption_text = safe_xpath_get_text(table_wrap, "./caption//text()").lower()
            logger.debug(f"Checking table '{label_text}'. Caption: '{caption_text[:100]}...'") # Debug level

            if "characteristics" in caption_text:
                logger.info(f"Found target table with 'characteristics' in caption (label: '{label_text}').") # Info level
                found_target_table = True
                refs_in_table = table_wrap.xpath(".//table//xref[@ref-type='bibr']")
                num_xrefs_in_table = len(refs_in_table)
                logger.debug(f"Found {num_xrefs_in_table} xrefs with ref-type='bibr' in the target table.") # Debug level

                for xref in refs_in_table:
                    rid = xref.get('rid')
                    if rid:
                        target_rids.add(rid)
                    else:
                        logger.warning(f"Found xref in target table without 'rid' attribute: {etree.tostring(xref, encoding='unicode').strip()}") # Warning level

                break # Stop after finding the first matching table

        if not found_target_table:
            logger.warning("No table found with 'characteristics' in its caption.") # Warning level
            return "", 0

        ref_pmids = []
        if target_rids:
            logger.debug(f"Looking up PMIDs for {len(target_rids)} unique rids: {target_rids}") # Debug level
            for rid in sorted(list(target_rids)):
                ref_element = root.xpath(f".//ref-list//ref[@id='{rid}']")
                if ref_element:
                    pmid = safe_xpath_get_text(ref_element[0], ".//pub-id[@pub-id-type='pmid']/text()")
                    if pmid and pmid.isdigit():
                        ref_pmids.append(pmid)
                        logger.debug(f"Found PMID {pmid} for rid='{rid}'") # Debug level
                    else:
                         logger.warning(f"Ref element found for rid='{rid}', but no valid PMID found within it.") # Warning level
                         # logger.debug(f"Content of ref[@id='{rid}']: {etree.tostring(ref_element[0], encoding='unicode', pretty_print=True)}") # Debug level
                else:
                    logger.warning(f"Could not find reference element for rid='{rid}' in ref-list.") # Warning level
        else:
             if num_xrefs_in_table > 0:
                 logger.warning(f"Target table found, but no valid 'rid' attributes found in its {num_xrefs_in_table} xrefs.") # Warning level
             else:
                 logger.warning("Target table found, but it contained no <xref> citations with ref-type='bibr'.") # Warning level

        num_pmids_found = len(ref_pmids)
        if num_xrefs_in_table != num_pmids_found:
            mismatch_msg = f"References len mismatch: Found {num_xrefs_in_table} xrefs in table, but extracted {num_pmids_found} PMIDs."
            logger.warning(mismatch_msg) # Warning level
            if DEBUG_MODE:
                return "References len mismatch", num_xrefs_in_table # Flag in Debug
            else:
                return "", num_xrefs_in_table # Empty string in Production
        else:
            pmids_str = ",".join(ref_pmids)
            count = num_pmids_found
            if count > 0:
                logger.info(f"Successfully extracted {count} PMIDs (match): {pmids_str}") # Info level
            else:
                logger.info("Target table found, 0 xrefs found, 0 PMIDs extracted (match).") # Info level
            return pmids_str, count

    except Exception as e:
        logger.error(f"Error during table reference extraction: {e}", exc_info=True) # Error level (always shows)
        return "", 0

# --- Main Processing ---
extracted_data = []
files_skipped_filter = 0

if not os.path.isdir(DATA_FOLDER):
    logger.error(f"Data folder '{DATA_FOLDER}' not found.") # Error level
    exit()

logger.info(f"Starting processing of XML files in '{DATA_FOLDER}'...") # Info level

# Prepare list of files to process
try:
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.xml')]
except FileNotFoundError:
     logger.error(f"Data folder '{DATA_FOLDER}' not found or inaccessible during file listing.") # Error level
     exit()
except Exception as e:
     logger.error(f"Error listing files in '{DATA_FOLDER}': {e}") # Error level
     exit()


files_to_process = all_files
total_files = len(all_files)

if DEBUG_MODE:
    logger.info(f"DEBUG MODE enabled. Processing max {DEBUG_LIMIT} files if more exist.")
    if total_files > DEBUG_LIMIT:
        logger.info(f"Limiting processing to the first {DEBUG_LIMIT} of {total_files} XML files.")
        files_to_process = all_files[:DEBUG_LIMIT]
    # In Debug mode, we use a standard loop and log each file
    iterable_files = files_to_process
    logger.info(f"Processing {len(files_to_process)} files...")
else:
    # In Production mode, wrap the list with tqdm for a progress bar
    logger.info(f"PRODUCTION MODE enabled. Logging level set to ERROR. Using tqdm for progress.") # This info won't show due to level, but good for code clarity
    iterable_files = tqdm(files_to_process, desc="Processing XML files", unit="file", disable=False) # Ensure tqdm is not disabled

file_counter = 0 # Use this to assign the 'Number' column

for filename in iterable_files:
    file_counter += 1 # Increment counter for each file *attempted*
    file_path = os.path.join(DATA_FOLDER, filename)

    # In DEBUG mode, log the specific file being processed
    if DEBUG_MODE:
        logger.info(f"--- Processing file {file_counter}/{len(files_to_process)}: {filename} ---")
    # In Production mode (not DEBUG_MODE), tqdm updates the progress bar automatically

    try:
        tree = etree.parse(file_path)
        root = tree.getroot()

        # --- Extract Data ---
        pmid = safe_xpath_get_text(root, ".//article-meta/article-id[@pub-id-type='pmid']/text()")
        title = safe_xpath_get_text(root, ".//title-group/article-title//text()")
        doi = safe_xpath_get_text(root, ".//article-meta/article-id[@pub-id-type='doi']/text()")
        url = f"https://doi.org/{doi}" if doi else ''
        category = safe_xpath_get_text(root, ".//article-categories/subj-group[@subj-group-type='heading']/subject/text()")
        conclusion = get_conclusion(root)
        if not conclusion:
            # Log that we are skipping because the conclusion is missing
            logging.warning(f"Skipping file '{filename}': No conclusion text found.")
            # Use 'continue' to skip the rest of the loop for this file
            continue
            
        year = get_year(root)

        ref_pmids_str, num_refs = get_table_references(root)
        logger.debug(f"File {filename}: Extracted PMIDs string: '{ref_pmids_str}', Count: {num_refs}") # Debug level

        # --- Append to results ---
        extracted_data.append([
            file_counter, # Use the running counter
            title,
            url,
            pmid,
            category,
            conclusion,
            year,
            ref_pmids_str,
            num_refs
        ])

    except etree.XMLSyntaxError as e:
        logger.error(f"XML Syntax Error parsing {filename}: {e}") # Error level
        # Optionally add a placeholder row or skip in production? For now, just log.
        if not DEBUG_MODE and isinstance(iterable_files, tqdm):
             iterable_files.set_postfix_str(f"Error in {filename}", refresh=True) # Show error briefly in tqdm bar
    except Exception as e:
        logger.error(f"An unexpected error occurred processing {filename}: {e}", exc_info=DEBUG_MODE) # Show traceback only in debug
        if not DEBUG_MODE and isinstance(iterable_files, tqdm):
             iterable_files.set_postfix_str(f"Error in {filename}", refresh=True) # Show error briefly in tqdm bar

# Ensure tqdm closes properly if it was used
if not DEBUG_MODE and isinstance(iterable_files, tqdm):
    iterable_files.close()

logger.info(f"Finished processing loop. Attempted {file_counter} files.") # Info level

# --- Write to CSV ---
if not extracted_data:
    logger.warning("No data was extracted. CSV file not created.") # Warning level
else:
    # Filtering logic remains the same (based on DEBUG_MODE flag for mismatch string)
    output_data_to_write = []
    if DEBUG_MODE:
        logger.info(f"DEBUG MODE: Preparing all {len(extracted_data)} extracted records for CSV.")
        output_data_to_write = extracted_data
    else:
        logger.info(f"PRODUCTION MODE: Filtering {len(extracted_data)} records to exclude rows where 'References' field is empty (potential mismatches handled in extraction).") # Info level (won't show)
        # Filter based on the actual content of the reference string (index 6)
        # Empty string "" means either no refs found, or a mismatch occurred in production mode.
        output_data_to_write = [row for row in extracted_data if row[7] != ""]
        logger.info(f"{len(output_data_to_write)} records remain after filtering.") # Info level (won't show)
        # Add a print statement for production mode clarity
        print(f"Extracted data for {len(output_data_to_write)} articles after filtering.")


    if output_data_to_write:
        try:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(CSV_HEADER)
                writer.writerows(output_data_to_write)
            logger.info(f"Successfully wrote {len(output_data_to_write)} records to '{OUTPUT_CSV}'") # Info level
            if not DEBUG_MODE:
                print(f"Successfully wrote {len(output_data_to_write)} records to '{OUTPUT_CSV}'") # Explicit print for production
        except IOError as e:
            logger.error(f"Could not write to CSV file '{OUTPUT_CSV}': {e}") # Error level
        except Exception as e:
            logger.error(f"An unexpected error occurred writing the CSV: {e}", exc_info=True) # Error level
    else:
        logger.warning("No data available to write to CSV (either none extracted or all filtered out).") # Warning level
        if not DEBUG_MODE:
            print("No data available to write to CSV (either none extracted or all filtered out).") # Explicit print for production

logger.info("Script finished.") # Info level