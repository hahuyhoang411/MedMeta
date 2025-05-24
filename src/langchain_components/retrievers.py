import logging
import os
import hashlib
import json
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _generate_documents_hash(docs: List[Document]) -> str:
    """
    Generate a hash of the documents to detect changes.
    
    Args:
        docs: List of LangChain Documents
        
    Returns:
        SHA256 hash string representing the documents
    """
    # Create a consistent representation of documents for hashing
    doc_data = []
    for doc in docs:
        doc_dict = {
            'page_content': doc.page_content,
            'metadata': dict(sorted(doc.metadata.items())) if doc.metadata else {}
        }
        doc_data.append(doc_dict)
    
    # Sort documents by page_content for consistent ordering
    doc_data.sort(key=lambda x: x['page_content'])
    
    # Create hash
    doc_string = json.dumps(doc_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(doc_string.encode('utf-8')).hexdigest()

def _save_documents_metadata(faiss_index_path: str, docs_hash: str, num_docs: int):
    """Save metadata about the documents used to create the FAISS index."""
    metadata_path = f"{faiss_index_path}_metadata.json"
    metadata = {
        'documents_hash': docs_hash,
        'num_documents': num_docs,
        'created_at': os.path.getctime(faiss_index_path) if os.path.exists(faiss_index_path) else None
    }
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    except Exception as e:
        logging.warning(f"Failed to save documents metadata: {e}")

def _load_documents_metadata(faiss_index_path: str) -> Dict[str, Any]:
    """Load metadata about the documents used to create the FAISS index."""
    metadata_path = f"{faiss_index_path}_metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load documents metadata: {e}")
    return {}

def _should_recreate_index(faiss_index_path: str, docs: List[Document], force_reembed: bool) -> bool:
    """
    Determine if we should recreate the FAISS index.
    
    Args:
        faiss_index_path: Path to the FAISS index
        docs: Current documents
        force_reembed: Whether to force re-embedding
        
    Returns:
        True if index should be recreated, False otherwise
    """
    if force_reembed:
        logging.info("Force re-embedding is enabled.")
        return True
    
    if not os.path.exists(faiss_index_path):
        logging.info("FAISS index does not exist.")
        return True
    
    # Check if documents have changed
    current_docs_hash = _generate_documents_hash(docs)
    saved_metadata = _load_documents_metadata(faiss_index_path)
    saved_docs_hash = saved_metadata.get('documents_hash')
    saved_num_docs = saved_metadata.get('num_documents', 0)
    
    if saved_docs_hash != current_docs_hash:
        logging.info(f"Documents have changed (hash mismatch). Current docs: {len(docs)}, Saved docs: {saved_num_docs}")
        return True
    
    logging.info(f"Documents unchanged (hash match). Using existing index with {saved_num_docs} documents.")
    return False

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
            logging.info("Embeddings model loaded.")
            
            # FAISS persistence logic
            faiss_index_path = config.get('FAISS_INDEX_PATH')
            force_reembed = config.get('FORCE_REEMBED', False)
            save_faiss_index = config.get('SAVE_FAISS_INDEX', True)
            
            faiss_vectorstore = None
            
            # Determine if we should recreate the index
            should_recreate = _should_recreate_index(faiss_index_path, docs, force_reembed)
            
            # Try to load existing FAISS index if we don't need to recreate
            if faiss_index_path and not should_recreate:
                try:
                    logging.info(f"Loading existing FAISS index from: {faiss_index_path}")
                    faiss_vectorstore = FAISS.load_local(
                        faiss_index_path, 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    logging.info(f"Successfully loaded FAISS index with {faiss_vectorstore.index.ntotal} vectors.")
                except Exception as e:
                    logging.warning(f"Failed to load existing FAISS index: {e}. Creating new index from documents.")
                    faiss_vectorstore = None
            
            # Create new FAISS index if loading failed or we need to recreate
            if faiss_vectorstore is None:
                logging.info("Creating new FAISS vector store from documents...")
                faiss_vectorstore = FAISS.from_documents(docs, embeddings)
                logging.info(f"FAISS vector store created with {faiss_vectorstore.index.ntotal} vectors.")
                
                # Save the FAISS index if requested and path is provided
                if save_faiss_index and faiss_index_path:
                    try:
                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
                        faiss_vectorstore.save_local(faiss_index_path)
                        logging.info(f"FAISS index saved to: {faiss_index_path}")
                        
                        # Save metadata about the documents
                        docs_hash = _generate_documents_hash(docs)
                        _save_documents_metadata(faiss_index_path, docs_hash, len(docs))
                        logging.info(f"Documents metadata saved (hash: {docs_hash[:8]}..., count: {len(docs)})")
                    except Exception as e:
                        logging.warning(f"Failed to save FAISS index to {faiss_index_path}: {e}")
            
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