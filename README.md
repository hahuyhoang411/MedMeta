# MedMeta

This repository contains a Python application that implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain and LangGraph. The goal is to assist with preliminary steps of meta-analysis research by:

1.  Generating a research plan and search queries based on a topic.
2.  Retrieving relevant scientific abstracts from a pre-built corpus (combining fetched PubMed data and a subset of a larger PubMed dataset).
3.  Synthesizing a conclusion based on the retrieved documents.

The pipeline uses BM25 and FAISS (with BGE embeddings) for retrieval, an Ensemble Retriever to combine them, and a Cross-Encoder (BGE Reranker) for reranking results before feeding them to a Google Gemini LLM for generation and synthesis.

## Repository Structure

```
pubmed_rag_pipeline/
├── src/ # Source code modules
│ ├── data_processing/ # Data loading, fetching, transformation
│ ├── langchain_components/ # LangChain wrappers (loaders, retrievers, LLM)
│ ├── meta_analysis_graph/ # LangGraph state, nodes, graph builder
│ └── config.py # Configuration settings
├── scripts/ # Runnable scripts
│ ├── 01_fetch_pubmed_data.py # Fetch data for specific PMIDs
│ ├── 02_run_rag_pipeline.py # Run the pipeline for a single query
│ └── 03_evaluate_on_medmeta.py# Evaluate pipeline on MedMeta.csv
├── data/ # Input data directory (e.g., MedMeta.csv)
├── output/ # Generated output files (fetched data, results)
├── requirements.txt # Python dependencies
├── .env.example # Example environment variables file
├── .gitignore # Git ignore file
└── README.md # This file
```


## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd pubmed_rag_pipeline
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3.  **Install dependencies:**
    *   Ensure you have CUDA installed if you plan to use `faiss-gpu`. If not, modify `requirements.txt` to use `faiss-cpu`.
    *   Install Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    *   *(Optional)* For graph visualization with Mermaid, install Playwright browsers:
        ```bash
        playwright install --with-deps
        ```

4.  **Set up API Keys:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your Google AI API key:
        ```ini
        GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
        NCBI_API_KEY="" # Optional: Add NCBI key if needed
        ```

5.  **Prepare Input Data:**
    *   Place your `MedMeta.csv` file inside the `data/` directory. It should have columns like `Meta Analysis Name`, `References`, `Number`.

## Usage

### 1. Fetch Initial PubMed Data (Optional but Recommended)

This script fetches data for specific PMIDs (like those in `MedMeta.csv` or other known relevant papers) and saves it to `output/pubmed_data_final.csv`. This file is used as part of the RAG knowledge base.

```bash
# Example: Fetch data for PMIDs from Notebook 1
python scripts/01_fetch_pubmed_data.py --pmids 31091372 38320511 29766772 38324415 39297945 37877587 36765286 35066509 32243865 26387030 23873274 --output output/pubmed_data_final.csv --use-api-key
```
* --pmids: Space-separated list of PubMed IDs.
* --output: Path for the output CSV (defaults to output/pubmed_data_final.csv).
* --use-api-key: Flag to use the NCBI_API_KEY from your .env file.

### 2. Run the RAG Pipeline for a Single Query
This script loads the data (fetched data + subset of PubMed25), sets up the retrievers and LLM, builds the graph, and runs it for a query you provide.

```bash
python scripts/02_run_rag_pipeline.py -q "Your Research Topic Query Here" --visualize
```
* -q or --query: The research topic (required).
* --visualize: (Optional) Tries to save a PNG visualization of the graph to the output/ directory.

### 3. Evaluate the Pipeline on MedMeta.csv
This script runs the RAG pipeline for each query listed in the Meta Analysis Name column of your MedMeta.csv file. It compares the PMIDs retrieved by the RAG process against the target PMIDs listed in the References column and saves detailed results.

```bash
# Run evaluation on all rows
python scripts/03_evaluate_on_medmeta.py

# Run evaluation on the first 5 rows with a 10-second wait between rows
python scripts/03_evaluate_on_medmeta.py --max_rows 5 --wait 10
```

* --eval_file: Path to the input CSV (defaults to data/MedMeta.csv).
* --output: Path for the results CSV (defaults to output/medmeta_evaluation_results.csv).
* --max_rows: (Optional) Process only the first N rows.
* --wait: (Optional) Seconds to wait between processing rows (useful for API rate limits).
The output CSV will contain the original columns plus Generated Conclusion, Retrieved PMIDs, Target PMIDs, Missing PMIDs, counts, and potentially the generated plan/queries.

## Configuration
Key parameters (model names, paths, retriever settings, etc.) can be adjusted in src/config.py.
