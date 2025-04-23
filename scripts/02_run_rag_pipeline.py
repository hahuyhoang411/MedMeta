import argparse
import logging
import sys
import os
import time
from itertools import islice

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
    OUTPUT_DIR
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress excessive warnings from libraries like huggingface_hub if needed
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

def main(query: str, visualize: bool):
    """
    Main function to setup and run the RAG pipeline for a single query.

    Args:
        query: The research topic query.
        visualize: Whether to attempt to save a visualization of the graph.
    """
    start_time = time.time()
    logging.info("--- Starting RAG Pipeline ---")
    logging.info(f"Using device: {DEVICE}")

    # --- 1. Load and Prepare Data ---
    logging.info("--- Step 1: Loading and Preparing Data ---")
    data_load_start = time.time()

    # Load reference dataset (fetched data)
    ref_dataset = load_local_csv_dataset(FETCHED_DATA_CSV_PATH)
    if ref_dataset is None:
        logging.error(f"Failed to load reference dataset from {FETCHED_DATA_CSV_PATH}. Exiting.")
        return

    # Load and process a subset of the large pubmed25 dataset
    # This can be slow, consider caching or pre-processing if run frequently
    processed_pubmed_dataset = load_and_process_pubmed25(
        dataset_name=PUBMED25_DATASET_NAME,
        split=PUBMED25_SPLIT,
        cache_dir=CACHE_DIR,
        num_proc=os.cpu_count() # Use multiple cores if available
    )

    datasets_to_combine = [ref_dataset]
    if processed_pubmed_dataset is not None and len(processed_pubmed_dataset) > 0:
        logging.info(f"Selecting {PUBMED25_SUBSET_SIZE} records from processed PubMed dataset...")
        small_pubmed_subset = processed_pubmed_dataset.select(range(min(PUBMED25_SUBSET_SIZE, len(processed_pubmed_dataset))))
        datasets_to_combine.append(small_pubmed_subset)
    else:
        logging.warning("Processed PubMed dataset is empty or failed to load. Using only reference dataset.")

    # Concatenate datasets
    rag_dataset = concatenate_hf_datasets(datasets_to_combine)
    if rag_dataset is None:
        logging.error("Failed to create combined dataset for RAG. Exiting.")
        return

    logging.info(f"Total combined dataset size for RAG: {len(rag_dataset)}")
    logging.info(f"Data loading and preparation took {time.time() - data_load_start:.2f} seconds.")

    # --- 2. Load Data into LangChain Documents ---
    logging.info("--- Step 2: Loading Data into LangChain Documents ---")
    doc_load_start = time.time()

    loader = CustomHuggingFaceDatasetLoader(
        dataset=rag_dataset,
        page_content_column="text",
        metadata_columns=["PMID", "Year"]
    )
    # Load all docs into memory for FAISS/BM25 indexing
    # For very large datasets, consider lazy loading strategies if indexing methods support it
    docs = loader.load() # Use load() instead of islice(lazy_load()) for full indexing

    if not docs:
        logging.error("No documents were loaded into LangChain format. Exiting.")
        return

    logging.info(f"Loaded {len(docs)} documents into LangChain format.")
    logging.info(f"Document loading took {time.time() - doc_load_start:.2f} seconds.")
    # logging.info("Sample document:")
    # print(docs[0]) # Print first doc as a sample check

    # --- 3. Setup Retrieval Chain ---
    logging.info("--- Step 3: Setting up Retrieval Chain ---")
    retriever_setup_start = time.time()

    compression_retriever = setup_retrieval_chain(docs, config=RETRIEVER_CONFIG, device=DEVICE)
    if compression_retriever is None:
        logging.error("Failed to set up the retrieval chain. Exiting.")
        return

    logging.info(f"Retrieval chain setup took {time.time() - retriever_setup_start:.2f} seconds.")

    # --- 4. Setup LLM ---
    logging.info("--- Step 4: Setting up Language Model ---")
    llm_setup_start = time.time()

    llm = get_llm(config=LLM_CONFIG)
    if llm is None:
        logging.error("Failed to set up the Language Model. Exiting.")
        return

    logging.info(f"LLM setup took {time.time() - llm_setup_start:.2f} seconds.")

    # --- 5. Build LangGraph Application ---
    logging.info("--- Step 5: Building LangGraph Application ---")
    graph_build_start = time.time()

    app = build_graph(llm=llm, retriever=compression_retriever)
    if app is None:
        logging.error("Failed to build the LangGraph application. Exiting.")
        return

    logging.info(f"LangGraph build took {time.time() - graph_build_start:.2f} seconds.")

    # --- 6. Visualize Graph (Optional) ---
    if visualize:
        logging.info("--- Step 6: Visualizing Graph ---")
        try:
            output_png_path = os.path.join(OUTPUT_DIR, "meta_analysis_graph.png")
            graph_viz_bytes = app.get_graph().draw_mermaid_png()
            with open(output_png_path, "wb") as f:
                f.write(graph_viz_bytes)
            logging.info(f"Graph visualization saved to {output_png_path}")
            # Optional: Display if in interactive environment (won't work in script)
            # from IPython.display import Image, display
            # display(Image(graph_viz_bytes))
        except ImportError:
            logging.warning("Could not visualize graph: `pygraphviz` or `mermaid` not installed properly. Skipping visualization.")
        except Exception as e:
            logging.error(f"Could not display graph: {e}", exc_info=True)


    # --- 7. Run Inference ---
    logging.info("--- Step 7: Running Inference ---")
    inference_start = time.time()

    inputs = {"research_topic": query}
    logging.info(f"Invoking RAG graph with query: '{query}'")

    # Use stream for progress updates or invoke for final result
    final_state = None
    try:
        # Config can be used to pass runtime parameters if needed, e.g., recursion limits
        # config = {"recursion_limit": 10}
        # final_state = app.invoke(inputs, config=config)

        # Stream events to see progress (optional)
        events = []
        logging.info("Streaming graph execution events...")
        for event in app.stream(inputs, stream_mode="values"):
             events.append(event)
             # Print updates (can be verbose)
             # print(f"Event: {event}")
             # final_state = event # Keep track of the latest state

        # The last event in 'values' mode should be the final state
        if events:
            final_state = events[-1]

        logging.info("Graph execution complete.")

    except Exception as e:
        logging.error(f"Error during graph invocation: {e}", exc_info=True)

    logging.info(f"Inference took {time.time() - inference_start:.2f} seconds.")

    # --- 8. Display Results ---
    logging.info("--- Step 8: Displaying Results ---")
    if final_state:
        print("\n--- Meta-Analysis Assistant Results ---")
        print(f"Initial Research Topic: {final_state.get('research_topic', 'N/A')}")

        plan = final_state.get('research_plan')
        if plan and isinstance(plan, dict): # Pydantic models might get dict representation
            print("\nGenerated Research Plan:")
            print(f"  Background: {plan.get('background', 'N/A')}")
            print(f"  Key Questions: {plan.get('key_questions', 'N/A')}")
            print(f"  Search Strategy Summary: {plan.get('search_strategy_summary', 'N/A')}")
        elif hasattr(plan, 'background'): # Handle case where it's still a Pydantic object
             print("\nGenerated Research Plan:")
             print(f"  Background: {plan.background}")
             print(f"  Key Questions: {plan.key_questions}")
             print(f"  Search Strategy Summary: {plan.search_strategy_summary}")
        else:
            print("\nResearch Plan: Not generated or invalid format.")

        queries = final_state.get('search_queries')
        if queries:
            print(f"\nGenerated Search Queries: {queries}")
        else:
             print("\nSearch Queries: Not generated.")

        retrieved = final_state.get('retrieved_docs', [])
        print(f"\nTotal Retrieved Docs (Aggregated): {len(retrieved)}")
        # You can print details of retrieved docs if needed
        # for i, doc in enumerate(retrieved[:3]): # Print first 3
        #      print(f"  Doc {i+1} PMID: {doc.metadata.get('PMID', 'N/A')}, Year: {doc.metadata.get('Year', 'N/A')}")
        #      print(f"    Content: {doc.page_content[:200]}...")


        conclusion = final_state.get('final_conclusion')
        if conclusion:
            print(f"\nSynthesized Conclusion:\n{conclusion}")
        else:
            print("\nFinal Conclusion: Not generated.")
    else:
        print("\n--- No final state returned from graph execution. ---")

    total_time = time.time() - start_time
    logging.info(f"--- RAG Pipeline Finished ---")
    logging.info(f"Total execution time: {total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Meta-Analysis RAG pipeline for a single query.")
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="The research topic query to process."
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="Attempt to save a visualization of the graph to output/meta_analysis_graph.png."
    )

    args = parser.parse_args()

    main(query=args.query, visualize=args.visualize)

    # Example usage:
    # python scripts/02_run_rag_pipeline.py -q "The Effect of Oral Semaglutide on Cardiovascular Risk Factors" --visualize