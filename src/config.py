# src/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
# Useful for local development
load_dotenv()

# --- API Keys ---
# Load from environment variables for security
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY" # Keep the variable name consistent
NCBI_API_KEY_ENV_VAR = "NCBI_API_KEY"
# Note: NCBI key is optional for basic use, required for higher E-utils rate limits.
# GOOGLE_API_KEY = os.getenv(GOOGLE_API_KEY_ENV_VAR) # Loaded directly in llm_models.py
NCBI_API_KEY = os.getenv(NCBI_API_KEY_ENV_VAR) # Can be passed to pubmed_fetcher

# --- File Paths ---
# Use relative paths from the project root or absolute paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Project root
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
CACHE_DIR = os.path.join(ROOT_DIR, ".cache") # Central cache directory

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

MEDMETA_CSV_PATH = os.path.join(DATA_DIR, "MedMeta.csv")
FETCHED_DATA_CSV_PATH = os.path.join(OUTPUT_DIR, "pubmed_data_final.csv") # Output from script 01
EVALUATION_RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, "medmeta_evaluation_results.csv") # Output from script 03

# For the check abstract fit for recreate conclusion
# MEDMETA_CSV_PATH = "meta_analysis_summary_with_topics.csv"
# FETCHED_DATA_CSV_PATH = "pubmed_data_final.csv" # Output from script 01
# EVALUATION_RESULTS_CSV_PATH = "abstract_check.csv" # Output from script 03

# --- Model Names ---
# Embeddings (Sentence Transformers/HuggingFace)
BGE_MODEL_NAME = "BAAI/bge-m3"
# Reranker (CrossEncoder/HuggingFace)
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# LLM (Google Generative AI)
LLM_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Updated model name

# --- Retriever Settings ---
RETRIEVER_CONFIG = {
    "RETRIEVER_MODE": "ensemble", # Options: 'ensemble', 'dense', 'bm25'
    "BM25_K": 10,
    "DENSE_RETRIEVER_K": 10, # Renamed from FAISS_K
    "ENSEMBLE_WEIGHTS": [0.5, 0.5], # Weights for BM25 and Dense Retriever (only used if RETRIEVER_MODE is 'ensemble')
    "COMPRESSION_TOP_N": 5, # Number of documents after reranking
    "BGE_MODEL_NAME": BGE_MODEL_NAME,
    "RERANKER_MODEL_NAME": RERANKER_MODEL_NAME,
    # FAISS persistence settings
    "FAISS_INDEX_PATH": os.path.join(CACHE_DIR, "faiss_index"), # Path to save/load FAISS index
    "FORCE_REEMBED": False, # Set to True to force re-embedding even if saved index exists
    "SAVE_FAISS_INDEX": True, # Whether to save the FAISS index after creation
}

# --- LLM Settings ---
LLM_CONFIG = {
    "LLM_PROVIDER": "google", # Or "openai" or "local" or "google"
    "LLM_MODEL_NAME": LLM_MODEL_NAME, # Or "gpt-4o-mini" or "Qwen/QwQ-32B" etc.
    "LLM_TEMPERATURE": 0.0,
    "LLM_MAX_TOKENS": 64000, # Let the provider handle default or set specific value
    # Optional Google Gemini inference
    "GOOGLE_API_KEY_ENV_VAR": "GOOGLE_API_KEY", # Keep for Google option
    "LLM_GOOGLE_THINKING_BUDGET": 0, # Example value, can be None or adjusted
    "OPENAI_API_KEY_ENV_VAR": "OPENAI_API_KEY", # Add for OpenAI option
    # Optional OpenAI specific params (add if needed)
    # "OPENAI_BASE_URL": None,
    # "OPENAI_ORGANIZATION": None,
    # "OPENAI_MAX_RETRIES": 2,
    # "OPENAI_TIMEOUT": None,
    # Optional vLLM inference
    "VLLM_OPENAI_API_BASE": "http://localhost:8001/v1", # For local provider
}

# --- Data Processing Settings ---
DEFAULT_PMID_COLUMN = 'References' 
PUBMED25_DATASET_NAME = "HoangHa/pubmed25"
PUBMED25_SPLIT = "train"
PUBMED25_SUBSET_SIZE = 10000 # Number of records to take from pubmed25 for the RAG dataset

# Device Selection ('cuda' or 'cpu')
# Automatically detect CUDA if available, otherwise use CPU
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu" # Default to CPU if torch is not installed or fails


# --- Evaluation Settings ---
EVAL_MAX_ROWS = None # Set to an integer (e.g., 2) to process only the first N rows of MedMeta.csv for debugging. Set to None to process all.
EVAL_WAIT_SECONDS = 10 # Reduced wait time between rows in evaluation script (adjust as needed for API limits)