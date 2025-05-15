import argparse
import logging
import sys
import os
import time
import re
import pandas as pd
from tqdm import tqdm
from typing import List, Set, Dict, Any, Optional
import numpy as np

# Ensure to host vLLM server before running this script
# Example command to start vLLM server:
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/QwQ-32B --tensor-parallel-size 4 --port 8001
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-32B --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size 4 --port 8001 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072 --guided-decoding-backend outlines
# vllm serve Qwen/Qwen3-30B-A3B --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size 8 --gpu-memory-utilization 0.8 --port 8001 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 65536 --enable-expert-parallel --guided-decoding-backend outlines
# Then check the config file

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

# Modify this function to return an ordered list
def get_ordered_retrieved_pmids(retrieved_docs: List[Any]) -> List[int]:
    """Extracts integer PMIDs from the metadata of retrieved documents, preserving order."""
    retrieved_pmids_list: List[int] = []
    if not retrieved_docs:
        return retrieved_pmids_list

    for doc_idx, doc_data in enumerate(retrieved_docs):
        # Handle both Document objects and potential dict representations
        metadata = None
        pmid_val = None
        try:
            if hasattr(doc_data, 'metadata') and isinstance(getattr(doc_data, 'metadata'), dict):
                metadata = doc_data.metadata
            elif isinstance(doc_data, dict) and 'metadata' in doc_data and isinstance(doc_data['metadata'], dict):
                metadata = doc_data['metadata']

            if metadata:
                pmid_val = metadata.get('PMID')
                if pmid_val is not None:
                    # Try converting to int after checking type
                    if isinstance(pmid_val, (int, str)):
                         retrieved_pmids_list.append(int(pmid_val))
                    elif isinstance(pmid_val, float): # Handle potential floats
                         retrieved_pmids_list.append(int(pmid_val))
                    else:
                         logging.warning(f"Retrieved PMID has unexpected type {type(pmid_val)}: {pmid_val}. Skipping.")
                # else: logging.debug(f"PMID key missing in metadata for doc {doc_idx}")
            # else: logging.debug(f"Metadata missing or not a dict for doc {doc_idx}")
        except (ValueError, TypeError) as conv_err:
             logging.warning(f"Could not convert retrieved PMID '{pmid_val}' to int for comparison. Error: {conv_err}. Skipping.")
        except Exception as e:
             logging.error(f"Unexpected error processing doc {doc_idx} for PMID: {e}. Skipping.", exc_info=True)


    return retrieved_pmids_list

# --- New Metric Calculation Functions ---

def calculate_hit_at_k(ordered_retrieved_pmids: List[int], target_pmids: Set[int], k: int) -> int:
    """Calculates Hit@k: 1 if any target PMID is in the top k retrieved, 0 otherwise."""
    if not target_pmids:
        return 0 # Or handle as appropriate, maybe 1 if retrieval is also empty? Usually 0.
    top_k_retrieved = set(ordered_retrieved_pmids[:k])
    return 1 if any(pmid in top_k_retrieved for pmid in target_pmids) else 0

def calculate_precision_at_k(ordered_retrieved_pmids: List[int], target_pmids: Set[int], k: int) -> float:
    """Calculates Precision@k: Proportion of relevant items in the top k."""
    if k == 0:
        return 0.0
    top_k_retrieved = ordered_retrieved_pmids[:k]
    relevant_in_top_k = sum(1 for pmid in top_k_retrieved if pmid in target_pmids)
    return relevant_in_top_k / k

def calculate_average_precision_at_k(ordered_retrieved_pmids: List[int], target_pmids: Set[int], k: int) -> float:
    """Calculates Average Precision (AP)@k."""
    if not target_pmids:
        return 0.0

    ap_sum = 0.0
    relevant_hits = 0
    for i, pmid in enumerate(ordered_retrieved_pmids[:k]):
        rank = i + 1
        if pmid in target_pmids:
            relevant_hits += 1
            precision_at_i = relevant_hits / rank
            ap_sum += precision_at_i

    # Normalize by the total number of relevant documents (min(k, len(target_pmids)) is sometimes used, but standard is len(target_pmids))
    return ap_sum / len(target_pmids) if target_pmids else 0.0
    
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
    k_value = RETRIEVER_CONFIG.get('COMPRESSION_TOP_N', 5) # Define K for metrics
    logging.info(f"Calculating metrics using k={k_value}")

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
    all_ap_at_k = [] # Store AP@k for each query
    all_hit_at_k = [] # Store Hit@k for each query

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
        # Initialize variables before the try block to ensure they exist
        ordered_retrieved_pmids: List[int] = []
        retrieved_docs_list = []
        missing_pmids_list: List[int] = []
        target_pmids: Set[int] = set()
        final_state: Optional[Dict[str, Any]] = None
        hit_at_k: int = 0
        ap_at_k: float = 0.0

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
                    retrieved_docs_list = final_state.get('retrieved_docs', []) # Assign here

                    logging.info(f"Retrieved {len(retrieved_docs_list)} documents.")

                    # Get ORDERED retrieved PMIDs - Assign here
                    ordered_retrieved_pmids = get_ordered_retrieved_pmids(retrieved_docs_list)
                    logging.info(f"Retrieved PMIDs (ordered): {ordered_retrieved_pmids if ordered_retrieved_pmids else 'None'}")

                    # Calculate metrics - Assign here
                    hit_at_k = calculate_hit_at_k(ordered_retrieved_pmids, target_pmids, k_value)
                    ap_at_k = calculate_average_precision_at_k(ordered_retrieved_pmids, target_pmids, k_value)
                    all_hit_at_k.append(hit_at_k)
                    all_ap_at_k.append(ap_at_k)
                    logging.info(f"Metrics for this row: Hit@{k_value}={hit_at_k}, AP@{k_value}={ap_at_k:.3f}")


                    if target_pmids:
                        # Fix TypeError: Convert list to set for difference operation
                        missing_pmids_set = target_pmids - set(ordered_retrieved_pmids)
                        missing_pmids_list = sorted(list(missing_pmids_set))
                    # else: missing_pmids_list = [] # Already initialized

                    logging.info(f"Missing PMIDs: {missing_pmids_list if missing_pmids_list else 'None'}")

                else:
                     generated_conclusion = "Error: RAG invocation returned None"
                     logging.error("RAG invocation returned None state.")
                     # Ensure missing PMIDs are calculated even if final_state is None but target_pmids exist
                     missing_pmids_list = sorted(list(target_pmids)) if target_pmids else []


            except Exception as e:
                logging.error(f"Error during RAG invocation or processing for row index {index} (Number: {original_number}): {e}", exc_info=True)
                generated_conclusion = f"Error: {type(e).__name__}" # Store error type
                # If error, assume all target PMIDs are missing
                missing_pmids_list = sorted(list(target_pmids)) if target_pmids else []
                # Metrics remain at their initialized values (0, 0.0)


        # Store results - Now ordered_retrieved_pmids, hit_at_k, ap_at_k are guaranteed to exist
        result_row = {
            **row_dict, # Include original columns
            'Generated Conclusion': generated_conclusion,
            'Retrieved PMIDs Ordered': ordered_retrieved_pmids, # Store ordered list
            'Target PMIDs': sorted(list(target_pmids)),
            'Missing PMIDs': missing_pmids_list,
            'Num Retrieved': len(ordered_retrieved_pmids),
            'Num Target': len(target_pmids),
            'Num Missing': len(missing_pmids_list),
            f'Hit@{k_value}': hit_at_k, # Store per-row metric
            f'AP@{k_value}': ap_at_k,   # Store per-row metric
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
            print("\n--- Evaluation Summary (k={k_value}) ---")
            # Calculate overall metrics
            if all_ap_at_k:
                map_at_k = np.mean(all_ap_at_k)
                print(f"MAP@{k_value}: {map_at_k:.4f}")
            else:
                print(f"MAP@{k_value}: N/A (No results)")

            if all_hit_at_k:
                hit_rate_at_k = np.mean(all_hit_at_k)
                print(f"Hit Rate@{k_value}: {hit_rate_at_k:.4f}")
            else:
                 print(f"Hit Rate@{k_value}: N/A (No results)")

            # Keep existing recall calculation if desired
            if 'Num Target' in df_results.columns and 'Num Missing' in df_results.columns:
                # Ensure division by zero is handled if Num Target is 0
                df_results['Recall@NumRetrieved'] = df_results.apply(
                    lambda r: (r['Num Target'] - r['Num Missing']) / r['Num Target'] if r['Num Target'] > 0 else (1.0 if r['Num Missing'] == 0 else 0.0),
                    axis=1
                )
                avg_recall = df_results['Recall@NumRetrieved'].mean()
                print(f"Average Recall@(Num Retrieved): {avg_recall:.4f}") # Recall over all retrieved docs
                print(f"Total Missing PMIDs across all rows: {df_results['Num Missing'].sum()}")


        except Exception as e:
            logging.error(f"Failed to save evaluation results: {e}", exc_info=True)
    else:
        logging.warning("No results were generated to save.")

    eval_end_time = time.time()
    logging.info(f"Total evaluation script execution time: {eval_end_time - eval_start_time:.2f} seconds.")

# --- Argparse and main call ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the MedMeta dataset.")
    parser.add_argument(
        "--k_metrics",
        type=int,
        default=RETRIEVER_CONFIG.get('COMPRESSION_TOP_N', 5),
        help="Value of K for calculating MAP@k and Hit Rate@k."
    )
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