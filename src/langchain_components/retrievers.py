import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_retrieval_chain(
    docs: List[Document],
    config: Dict[str, Any],
    device: str = 'cuda' # Allow device selection ('cuda' or 'cpu')
) -> Optional[ContextualCompressionRetriever]:
    """
    Sets up the retrieval chain based on the configuration.
    Can use ensemble (BM25 + Dense), single Dense (FAISS), or single BM25,
    followed by a reranker.

    Args:
        docs: List of LangChain Documents to index.
        config: Dictionary containing configuration parameters:
            - RETRIEVER_MODE: 'ensemble', 'dense', or 'bm25'.
            - BGE_MODEL_NAME: Name of the BGE embedding model.
            - RERANKER_MODEL_NAME: Name of the CrossEncoder reranker model.
            - BM25_K: Number of documents for BM25 to retrieve.
            - DENSE_RETRIEVER_K: Number of documents for the dense retriever (FAISS) to retrieve.
            - ENSEMBLE_WEIGHTS: Weights for BM25 and Dense retriever (if mode is 'ensemble').
            - COMPRESSION_TOP_N: Number of documents to return after reranking.
        device: The device to run embedding and reranker models on ('cuda' or 'cpu').

    Returns:
        The configured ContextualCompressionRetriever, or None if setup fails.
    """
    if not docs:
        logging.error("Cannot setup retriever chain: No documents provided.")
        return None

    retriever_mode = config.get('RETRIEVER_MODE', 'ensemble').lower()
    logging.info(f"Setting up retriever in mode: '{retriever_mode}'")

    base_retriever = None
    bm25_retriever = None
    dense_retriever = None # Renamed from faiss_retriever

    try:
        # Initialize retrievers based on mode
        if retriever_mode in ['ensemble', 'bm25']:
            logging.info(f"Initializing BM25Retriever (k={config['BM25_K']})...")
            bm25_retriever = BM25Retriever.from_documents(docs, k=config['BM25_K'])
            logging.info("BM25Retriever initialized.")
            if retriever_mode == 'bm25':
                base_retriever = bm25_retriever

        if retriever_mode in ['ensemble', 'dense']: # Changed from 'faiss'
            logging.info(f"Initializing HuggingFaceBgeEmbeddings ({config['BGE_MODEL_NAME']}) on device '{device}'...")
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=config['BGE_MODEL_NAME'],
                model_kwargs={'device': device, 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True},
                query_instruction="search_query:",
                embed_instruction="search_document:"
            )
            logging.info("Embeddings model loaded. Initializing FAISS vector store for Dense Retriever...")
            # We still use FAISS as the backend, but refer to it as the dense retriever conceptually
            faiss_vectorstore = FAISS.from_documents(docs, embeddings)
            dense_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": config['DENSE_RETRIEVER_K']}) # Use new config key
            logging.info(f"Dense Retriever (FAISS backend) initialized (k={config['DENSE_RETRIEVER_K']}).")
            if retriever_mode == 'dense': # Changed from 'faiss'
                base_retriever = dense_retriever

        # Setup Ensemble if mode is 'ensemble'
        if retriever_mode == 'ensemble':
            if not bm25_retriever or not dense_retriever: # Changed from faiss_retriever
                 logging.error("Both BM25 and Dense retrievers are required for ensemble mode.")
                 return None
            logging.info(f"Initializing EnsembleRetriever with weights {config['ENSEMBLE_WEIGHTS']}...")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, dense_retriever], # Changed from faiss_retriever
                weights=config['ENSEMBLE_WEIGHTS']
            )
            logging.info("EnsembleRetriever initialized.")
            base_retriever = ensemble_retriever

        if base_retriever is None:
            logging.error(f"Invalid retriever mode '{retriever_mode}' or failed to initialize base retriever.")
            return None

        # 4. Initialize Contextual Compression Retriever with CrossEncoder reranker (applies to all modes)
        logging.info(f"Initializing CrossEncoder ({config['RERANKER_MODEL_NAME']}) on device '{device}'...")
        cross_encoder_model = HuggingFaceCrossEncoder(
            model_name=config['RERANKER_MODEL_NAME'],
            model_kwargs={'device': device, 'trust_remote_code': True}
        )
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=config['COMPRESSION_TOP_N'])
        logging.info(f"CrossEncoder loaded. Initializing ContextualCompressionRetriever (top_n={config['COMPRESSION_TOP_N']})...")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever # Use the selected base retriever
        )
        logging.info("ContextualCompressionRetriever initialized successfully.")
        return compression_retriever

    except Exception as e:
        logging.error(f"Failed to setup retrieval chain: {e}", exc_info=True)
        return None