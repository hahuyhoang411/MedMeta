import argparse
import logging
import sys
import os
import time
import re
import pandas as pd
from tqdm import tqdm # For progress bar
from typing import List, Set, Dict, Any, Optional

# Ensure the src directory is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Project Modules
from src.config import (
    FETCHED_DATA_CSV_PATH,
    PUBMED25_DATASET_NAME,
    PUBMED25_SPLIT,
    PUBMED25_SUBSET_SIZE,
    CACHE_DIR,
    RETRIEVER_CONFIG,
    LLM_CONFIG,
    DEVICE,
    MEDMETA_CSV_PATH,
    EVALUATION_RESULTS_CSV_PATH,
    EVAL_MAX_ROWS,
    EVAL_WAIT_SECONDS
)
from src.data_processing.dataset_loader import (
    load_and_process_pubmed25,
    load_local_csv_dataset,
    concatenate_hf_datasets
)
from src.langchain_components.document_loaders import CustomHuggingFaceDatasetLoader
from src.langchain_components.retrievers import setup_retrieval_chain
from src.langchain_components.llm_models import get_llm
from src.meta_analysis_graph.graph_builder import build_graph
from langgraph.graph import StateGraph # For type hint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("huggingface_hub").setLevel(logging.WARNING) # Reduce verbosity

def parse_pmids_from_string(references_str: Optional[str]) -> Set[int]:
    """Extracts integer PMIDs from a string."""
    target_pmids: Set[int] = set()
    if references_str and isinstance(references_str, str):
        # Find all sequences of digits
        pmid_matches = re.findall(r'\d+', references_str)
        for pmid_str in pmid_matches:
            try:
                # Basic validation: Check length (usually 7-8 digits, but can vary)
                if 6 <= len(pmid_str) <= 9:
                    target_pmids.add(int(pmid_str))
                else:
                    logging.debug(f"Skipping potential non-PMID number: {pmid_str}")
            except ValueError:
                logging.warning(f"Could not convert found number '{pmid_str}' to integer PMID.")
    elif references_str:
         logging.warning(f"References field is not a string: {type(references_str)} - {references_str}")
    return target_pmids

def get_retrieved_pmids(retrieved_docs: List[Dict[str, Any]]) -> Set[int]:
    """Extracts integer PMIDs from the metadata of retrieved documents."""
    retrieved_pmids_set: Set[int] = set()
    if not retrieved_docs:
        return retrieved_pmids_set

    for doc_idx, doc_data in enumerate(retrieved_docs):
        # Handle both Document objects and potential dict representations
        metadata = None
        pmid = None
        if hasattr(doc_data, 'metadata') and isinstance(getattr(doc_data, 'metadata'), dict):
            metadata = doc_data.metadata
        elif isinstance(doc_data, dict) and 'metadata' in doc_data and isinstance(doc_data['metadata'], dict):
            metadata = doc_data['metadata']

        if metadata:
            pmid_val = metadata.get('PMID')
            if pmid_val is not None:
                try:
                    # Try converting to int after checking type
                    if isinstance(pmid_val, (int, str)):
                         retrieved_pmids_set.add(int(pmid_val))
                    elif isinstance(pmid_val, float): # Handle potential floats if source data is messy
                         retrieved_pmids_set.add(int(pmid_val))
                    else:
                         logging.warning(f"Retrieved PMID has unexpected type {type(pmid_val)}: {pmid_val}. Skipping.")

                except (ValueError, TypeError) as conv_err:
                     logging.warning(f"Could not convert retrieved PMID '{pmid_val}' to int for comparison. Error: {conv_err}")
            # else: logging.debug(f"PMID key missing in metadata for doc {doc_idx}") # Too verbose usually
        # else: logging.debug(f"Metadata missing or not a dict for doc {doc_idx}") # Too verbose

    return retrieved_pmids_set


def setup_pipeline():
    """Consolidated setup function to prepare data, retrievers, LLM, and graph."""
    logging.info("--- Setting up RAG Pipeline for Evaluation ---")
    logging.info(f"Using device: {DEVICE}")

    # --- 1. Load and Prepare Data ---
    ref_dataset = load_local_csv_dataset(FETCHED_DATA_CSV_PATH)
    if ref_dataset is None: raise RuntimeError("Failed to load reference dataset.")

    processed_pubmed_dataset = load_and_process_pubmed25(
        dataset_name=PUBMED25_DATASET_NAME, split=PUBMED25_SPLIT, cache_dir=CACHE_DIR
    )
    datasets_to_combine = [ref_dataset]
    if processed_pubmed_dataset and len(processed_pubmed_dataset) > 0:
        small_pubmed_subset = processed_pubmed_dataset.select(range(min(PUBMED25_SUBSET_SIZE, len(processed_pubmed_dataset))))
        datasets_to_combine.append(small_pubmed_subset)
    rag_dataset = concatenate_hf_datasets(datasets_to_combine)
    if rag_dataset is None: raise RuntimeError("Failed to create combined dataset.")
    logging.info(f"RAG dataset ready with {len(rag_dataset)} documents.")

    # --- 2. Load LangChain Docs ---
    loader = CustomHuggingFaceDatasetLoader(rag_dataset, metadata_columns=["PMID", "Year"])
    docs = loader.load()
    if not docs: raise RuntimeError("No documents loaded into LangChain format.")
    logging.info(f"Loaded {len(docs)} LangChain documents.")

    # --- 3. Setup Retrievers ---
    compression_retriever = setup_retrieval_chain(docs, config=RETRIEVER_CONFIG, device=DEVICE)
    if compression_retriever is None: raise RuntimeError("Failed to setup retrieval chain.")

    # --- 4. Setup LLM ---
    llm = get_llm(config=LLM_CONFIG)
    if llm is None: raise RuntimeError("Failed to setup LLM.")

    # --- 5. Build Graph ---
    app = build_graph(llm=llm, retriever=compression_retriever)
    if app is None: raise RuntimeError("Failed to build LangGraph application.")

    logging.info("--- Pipeline Setup Complete ---")
    return app

def main(eval_file: str, output_file: str, max_rows: Optional[int], wait_time: int):
    """
    Main function to run evaluation on the MedMeta dataset.

    Args:
        eval_file: Path to the MedMeta CSV file.
        output_file: Path to save the evaluation results CSV.
        max_rows: Maximum number of rows to process from eval_file (None for all).
        wait_time: Seconds to wait between processing rows (for API rate limits).
    """
    eval_start_time = time.time()

    # --- Setup the Pipeline ---
    try:
        rag_app = setup_pipeline()
    except RuntimeError as e:
        logging.error(f"Pipeline setup failed: {e}. Exiting.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during setup: {e}", exc_info=True)
        return

    # --- Load Evaluation Data ---
    logging.info(f"Loading evaluation data from: {eval_file}")
    if not os.path.exists(eval_file):
        logging.error(f"Evaluation file not found: {eval_file}. Exiting.")
        return
    try:
        df_eval_input = pd.read_csv(eval_file)
        # Basic validation of expected columns
        required_cols = {'Meta Analysis Name', 'References', 'Number'}
        if not required_cols.issubset(df_eval_input.columns):
            logging.error(f"Evaluation file missing required columns. Found: {df_eval_input.columns}. Required: {required_cols}. Exiting.")
            return
        logging.info(f"Loaded {len(df_eval_input)} rows for evaluation.")
    except Exception as e:
        logging.error(f"Failed to load or validate evaluation CSV: {e}", exc_info=True)
        return

    # Limit rows if specified
    if max_rows is not None and max_rows > 0:
        logging.warning(f"Processing only the first {max_rows} rows for evaluation.")
        df_eval_input = df_eval_input.head(max_rows)
    elif max_rows == 0:
         logging.warning("Max rows set to 0. No evaluation will be performed.")
         return

    num_rows_to_process = len(df_eval_input)
    results_list = []

    # --- Process Each Row ---
    logging.info(f"Starting evaluation loop for {num_rows_to_process} rows...")
    for index, row in tqdm(df_eval_input.iterrows(), total=num_rows_to_process, desc="Evaluating Rows"):
        start_row_time = time.time()
        row_dict = row.to_dict()
        query = row_dict.get('Meta Analysis Name')
        references_str = row_dict.get('References')
        original_number = row_dict.get('Number', f'Index_{index}') # Use index if Number is missing

        logging.info(f"\n--- Processing Row {index+1}/{num_rows_to_process} (Number: {original_number}) ---")
        logging.info(f"Query: {query}")

        generated_conclusion = "Skipped"
        retrieved_docs_list = []
        missing_pmids_list: List[int] = []
        target_pmids: Set[int] = set()
        final_state: Optional[Dict[str, Any]] = None


        if not query or pd.isna(query):
            logging.warning(f"Row {index+1}: 'Meta Analysis Name' is missing or NaN. Skipping RAG query.")
            generated_conclusion = 'Skipped - No Query'
        else:
            target_pmids = parse_pmids_from_string(references_str)
            logging.info(f"Parsed Target PMIDs: {target_pmids if target_pmids else 'None'}")

            inputs = {"research_topic": query}
            try:
                logging.info("Invoking RAG application...")
                # Use invoke for simplicity in evaluation loop
                final_state = rag_app.invoke(inputs)
                logging.info("RAG application invocation complete.")

                if final_state:
                    generated_conclusion = final_state.get('final_conclusion', "Conclusion not found in RAG output")
                    # Extract retrieved docs - handle potential variation in state structure
                    retrieved_docs_list = final_state.get('retrieved_docs', [])

                    logging.info(f"Retrieved {len(retrieved_docs_list)} documents.")

                    retrieved_pmids_set = get_retrieved_pmids(retrieved_docs_list)
                    logging.info(f"Retrieved PMIDs (parsed): {retrieved_pmids_set if retrieved_pmids_set else 'None'}")

                    if target_pmids:
                        missing_pmids_set = target_pmids - retrieved_pmids_set
                        missing_pmids_list = sorted(list(missing_pmids_set))
                    # else: missing_pmids_list = [] # Already initialized

                    logging.info(f"Missing PMIDs: {missing_pmids_list if missing_pmids_list else 'None'}")

                else:
                     generated_conclusion = "Error: RAG invocation returned None"
                     logging.error("RAG invocation returned None state.")

            except Exception as e:
                logging.error(f"Error during RAG invocation or processing for row index {index} (Number: {original_number}): {e}", exc_info=True)
                generated_conclusion = f"Error: {type(e).__name__}" # Store error type
                # If error, assume all target PMIDs are missing
                missing_pmids_list = sorted(list(target_pmids)) if target_pmids else []


        # Store results
        result_row = {
            **row_dict, # Include original columns
            'Generated Conclusion': generated_conclusion,
            'Retrieved PMIDs': sorted(list(get_retrieved_pmids(retrieved_docs_list))), # Store retrieved PMIDs as sorted list
            'Target PMIDs': sorted(list(target_pmids)), # Store target PMIDs as sorted list
            'Missing PMIDs': missing_pmids_list,
            'Num Retrieved': len(retrieved_docs_list),
            'Num Target': len(target_pmids),
            'Num Missing': len(missing_pmids_list)
        }
        # Optionally add details from final_state if needed (e.g., generated plan, queries)
        if final_state:
             result_row['Generated Plan'] = str(final_state.get('research_plan', 'N/A'))
             result_row['Generated Queries'] = str(final_state.get('search_queries', 'N/A'))

        results_list.append(result_row)

        end_row_time = time.time()
        logging.info(f"Row {index+1} processed in {end_row_time - start_row_time:.2f} seconds.")

        # Wait between rows if it's not the last row
        if wait_time > 0 and index < num_rows_to_process - 1:
            logging.info(f"Waiting for {wait_time} seconds before next row...")
            time.sleep(wait_time)

    # --- Save Results ---
    logging.info(f"Evaluation loop finished. Processed {len(results_list)} rows.")
    if results_list:
        df_results = pd.DataFrame(results_list)
        logging.info(f"Saving evaluation results to {output_file}...")
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_results.to_csv(output_file, index=False, encoding='utf-8')
            logging.info("Evaluation results saved successfully.")
            print("\n--- Evaluation Results Sample (First 5 Rows) ---")
            with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 50):
                print(df_results.head())
            print("\n--- Evaluation Summary ---")
            # Calculate basic metrics if desired (e.g., average missing rate)
            if 'Num Target' in df_results.columns and 'Num Missing' in df_results.columns:
                df_results['Recall@k'] = df_results.apply(lambda r: (r['Num Target'] - r['Num Missing']) / r['Num Target'] if r['Num Target'] > 0 else 0, axis=1)
                avg_recall = df_results['Recall@k'].mean()
                print(f"Average Recall@k (Top {RETRIEVER_CONFIG.get('COMPRESSION_TOP_N', 5)}): {avg_recall:.3f}")
                print(f"Total Missing PMIDs across all rows: {df_results['Num Missing'].sum()}")


        except Exception as e:
            logging.error(f"Failed to save evaluation results: {e}", exc_info=True)
    else:
        logging.warning("No results were generated to save.")

    eval_end_time = time.time()
    logging.info(f"Total evaluation script execution time: {eval_end_time - eval_start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the MedMeta dataset.")
    parser.add_argument(
        "--eval_file",
        default=MEDMETA_CSV_PATH,
        help=f"Path to the input MedMeta CSV file (default: {MEDMETA_CSV_PATH})."
    )
    parser.add_argument(
        "--output",
        default=EVALUATION_RESULTS_CSV_PATH,
        help=f"Path to save the evaluation results CSV file (default: {EVALUATION_RESULTS_CSV_PATH})."
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=EVAL_MAX_ROWS, # Use default from config (None means all rows)
        help="Maximum number of rows to process from the evaluation file (for debugging)."
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=EVAL_WAIT_SECONDS, # Use default from config
        help="Seconds to wait between processing rows (for API rate limits)."
    )

    args = parser.parse_args()

    main(
        eval_file=args.eval_file,
        output_file=args.output,
        max_rows=args.max_rows,
        wait_time=args.wait
    )

    # Example Usage:
    # python scripts/03_evaluate_on_medmeta.py
    # python scripts/03_evaluate_on_medmeta.py --max_rows 5 --wait 5