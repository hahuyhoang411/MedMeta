import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
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
    Sets up the ensemble and compression retrieval chain (BM25 + FAISS + Reranker).

    Args:
        docs: List of LangChain Documents to index.
        config: Dictionary containing configuration parameters:
            - BGE_MODEL_NAME: Name of the BGE embedding model.
            - RERANKER_MODEL_NAME: Name of the CrossEncoder reranker model.
            - BM25_K: Number of documents for BM25 to retrieve.
            - FAISS_K: Number of documents for FAISS to retrieve.
            - ENSEMBLE_WEIGHTS: Weights for BM25 and FAISS in the ensemble.
            - COMPRESSION_TOP_N: Number of documents to return after reranking.
        device: The device to run embedding and reranker models on ('cuda' or 'cpu').

    Returns:
        The configured ContextualCompressionRetriever, or None if setup fails.
    """
    if not docs:
        logging.error("Cannot setup retriever chain: No documents provided.")
        return None

    try:
        # 1. Initialize BM25 retriever
        logging.info(f"Initializing BM25Retriever (k={config['BM25_K']})...")
        bm25_retriever = BM25Retriever.from_documents(docs, k=config['BM25_K'])
        logging.info("BM25Retriever initialized.")

        # 2. Initialize FAISS retriever with HuggingFace embeddings
        logging.info(f"Initializing HuggingFaceBgeEmbeddings ({config['BGE_MODEL_NAME']}) on device '{device}'...")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=config['BGE_MODEL_NAME'],
            model_kwargs={'device': device, 'trust_remote_code': True}, # Trust remote code for BGE models
            encode_kwargs={'normalize_embeddings': True}, # Normalize for cosine similarity
            query_instruction="search_query:", # Instructions for BGE M3/Mistral models
            embed_instruction="search_document:"
        )
        logging.info("Embeddings model loaded. Initializing FAISS vector store...")
        # Consider handling FAISS index saving/loading for large datasets
        faiss_vectorstore = FAISS.from_documents(docs, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": config['FAISS_K']})
        logging.info(f"FAISS vector store initialized and retriever created (k={config['FAISS_K']}).")

        # 3. Initialize Ensemble Retriever
        logging.info(f"Initializing EnsembleRetriever with weights {config['ENSEMBLE_WEIGHTS']}...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=config['ENSEMBLE_WEIGHTS']
        )
        logging.info("EnsembleRetriever initialized.")

        # 4. Initialize Contextual Compression Retriever with CrossEncoder reranker
        logging.info(f"Initializing CrossEncoder ({config['RERANKER_MODEL_NAME']}) on device '{device}'...")
        cross_encoder_model = HuggingFaceCrossEncoder(
            model_name=config['RERANKER_MODEL_NAME'],
            model_kwargs={'device': device, 'trust_remote_code': True} # Trust remote code if necessary
        )
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=config['COMPRESSION_TOP_N'])
        logging.info(f"CrossEncoder loaded. Initializing ContextualCompressionRetriever (top_n={config['COMPRESSION_TOP_N']})...")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        logging.info("ContextualCompressionRetriever initialized successfully.")
        return compression_retriever

    except Exception as e:
        logging.error(f"Failed to setup retrieval chain: {e}", exc_info=True)
        return None