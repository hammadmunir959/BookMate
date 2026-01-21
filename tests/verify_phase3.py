"""
Verification script for Phase 3: Advanced Retrieval
Tests Hybrid Search (RRF) and Cross-Encoder Reranking
"""

import os
import sys
import logging
import time
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import config
from src.core.document_models import RetrievalQuery, QueryType, DocumentChunk, DocumentMetadata
from src.storage.metadata_store import SQLiteStorage
from src.storage.vector_store import ChromaDatabase
from src.processors.retrieval.hybrid_retriever import HybridRetriever
from src.processors.retrieval.keyword_retriever import KeywordRetriever
from src.processors.retrieval.semantic_retriever import SemanticRetriever
from src.processors.retrieval.reranker import Reranker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_test_data(metadata_store, vector_store):
    """Setup test data in DB and Vector Store"""
    logger.info("Setting up test data...")
    
    doc_id = "verification_doc_1"
    
    # Create chunks with varied content to test hybrid vs semantic
    chunks = [
        DocumentChunk(
            chunk_id="c1", document_id=doc_id, chunk_index=0, start_position=0, end_position=100, token_count=20,
            content="The quick brown fox jumps over the lazy dog. Agile animals are fast.",
            metadata=DocumentMetadata(
                filename="test.txt", title="Test Doc", 
                file_path="/tmp/test.txt", file_size=100, file_extension=".txt", mime_type="text/plain"
            )
        ),
        DocumentChunk(
            chunk_id="c2", document_id=doc_id, chunk_index=1, start_position=100, end_position=200, token_count=20,
            content="Python is a programming language. It is great for data science and AI.",
            metadata=DocumentMetadata(
                filename="test.txt", title="Test Doc", 
                file_path="/tmp/test.txt", file_size=100, file_extension=".txt", mime_type="text/plain"
            )
        ),
        DocumentChunk(
            chunk_id="c3", document_id=doc_id, chunk_index=2, start_position=200, end_position=300, token_count=20,
            content="Apples are red or green fruits. They are healthy and delicious.",
            metadata=DocumentMetadata(
                filename="test.txt", title="Test Doc", 
                file_path="/tmp/test.txt", file_size=100, file_extension=".txt", mime_type="text/plain"
            )
        ),
        DocumentChunk(
            chunk_id="c4", document_id=doc_id, chunk_index=3, start_position=300, end_position=400, token_count=20,
            content="Keyword match: special_term_xyz. But semantically unrelated to query.",
            metadata=DocumentMetadata(
                filename="test.txt", title="Test Doc", 
                file_path="/tmp/test.txt", file_size=100, file_extension=".txt", mime_type="text/plain"
            )
        ),
        DocumentChunk(
            chunk_id="c5", document_id=doc_id, chunk_index=4, start_position=400, end_position=500, token_count=20,
            content="Advanced retrieval systems use hybrid search combining keyword and vector methods.",
            metadata=DocumentMetadata(
                filename="test.txt", title="Test Doc", 
                file_path="/tmp/test.txt", file_size=100, file_extension=".txt", mime_type="text/plain"
            )
        )
    ]
    
    # Add to SQLite (Metadata Store)
    for chunk in chunks:
        # We need to manually insert into chunks table for testing keyword search
        with sqlite3.connect(metadata_store.db_path) as conn:
            cursor = conn.cursor()
            # Ensure document exists
            cursor.execute("INSERT OR IGNORE INTO documents (document_id, filename, title, created_at) VALUES (?, ?, ?, ?)",
                          (doc_id, "test.txt", "Test Doc", "2023-01-01"))
            # Insert chunk
            cursor.execute("""
                INSERT OR REPLACE INTO chunks (chunk_id, document_id, content, chunk_index, token_count, start_position, end_position)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chunk.chunk_id, doc_id, chunk.content, chunk.chunk_index, chunk.token_count, chunk.start_position, chunk.end_position))
    
    # Add embeddings (mock or real)
    # Using SemanticRetriever to generate embeddings if possible, or just mock
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.model.embedding_model)
        for chunk in chunks:
            chunk.embedding = model.encode(chunk.content).tolist()
    except Exception as e:
        logger.warning(f"Could not load embedding model for setup, using random embeddings: {e}")
        import random
        for chunk in chunks:
            chunk.embedding = [random.random() for _ in range(384)]

    # Add to Vector Store
    vector_store.add_chunks(chunks)
    logger.info("Test data setup complete.")

import sqlite3

def test_keyword_search():
    logger.info(">>> Testing Keyword Retriever (BM25)")
    kr = KeywordRetriever()
    
    # Test Exact Match
    query = RetrievalQuery(query="special_term_xyz", query_type=QueryType.KEYWORD, top_k=3)
    results = kr.retrieve(query)
    
    if results and results[0].chunk_id == "c4":
        logger.info(f"✅ Exact keyword match found: {results[0].content[:50]}...")
        logger.info(f"   Score: {results[0].keyword_score}")
    else:
        logger.error("❌ Exact keyword match failed")
        if results: logger.info(f"Top result: {results[0].content}")

    # Test Partial Match
    query = RetrievalQuery(query="hybrid search", query_type=QueryType.KEYWORD, top_k=3)
    results = kr.retrieve(query)
    if results:
         logger.info(f"✅ Partial match found results: {len(results)}")
         for r in results:
             logger.info(f"   - {r.chunk_id}: Score={r.keyword_score:.4f}")
    else:
         logger.warning("⚠️ No results for partial match")

def test_semantic_search():
    logger.info(">>> Testing Semantic Retriever")
    sr = SemanticRetriever()
    
    # Test Semantic Match (no keyword overlap)
    # "fox" chunk should match "quick animal"
    query = RetrievalQuery(query="fast moving animal", query_type=QueryType.SEMANTIC, top_k=3)
    results = sr.retrieve(query)
    
    if results:
        logger.info(f"✅ Semantic results found: {len(results)}")
        logger.info(f"   Top result: {results[0].content[:50]}... (ID: {results[0].chunk_id})")
        logger.info(f"   Score: {results[0].semantic_score}")
    else:
        logger.error("❌ Semantic search failed")

def test_hybrid_search():
    logger.info(">>> Testing Hybrid Retriever (RRF)")
    hr = HybridRetriever()
    
    # "hybrid search" appears in c5. "search" might appear elsewhere.
    # We want to see if ranks are combined.
    query = RetrievalQuery(query="hybrid retrieval systems", query_type=QueryType.HYBRID, top_k=3)
    results = hr.retrieve(query)
    
    if results:
        logger.info(f"✅ Hybrid results found: {len(results)}")
        for r in results:
            logger.info(f"   - {r.chunk_id} ({r.retrieval_method}): FinalScore={r.final_score:.4f}")
            if hasattr(r, 'semantic_score'): logger.info(f"     Sem={r.semantic_score:.4f}, Kw={r.keyword_score:.4f}")
    else:
        logger.error("❌ Hybrid search failed")

def test_reranker():
    logger.info(">>> Testing Reranker Integration")
    # Enable reranking in config temporarily if not on
    orig_enable = config.retrieval.enable_reranking
    config.retrieval.enable_reranking = True
    
    try:
        hr = HybridRetriever()
        
        # Query that might be ambiguous
        # "python" -> c2 (prog lang) vs c1 (animal? no)
        query = RetrievalQuery(query="What gets used for AI?", query_type=QueryType.HYBRID, top_k=3, enable_reranking=True)
        
        results = hr.retrieve(query)
        
        if results:
            logger.info(f"✅ Reranked results found: {len(results)}")
            for r in results:
                logger.info(f"   - {r.chunk_id}: {r.content[:50]}...")
                logger.info(f"     Final Score: {r.final_score:.4f} (Method: {r.retrieval_method})")
                
            # Check if retrieval method indicates reranking
            if results[0].retrieval_method == "hybrid_reranked":
                logger.info("✅ Reranking was applied successfully")
            else:
                logger.warning(f"⚠️ Reranking tag missing, method is: {results[0].retrieval_method}")
        else:
            logger.error("❌ Reranking test returned no results")
            
    finally:
        config.retrieval.enable_reranking = orig_enable

def main():
    try:
        # Initialize stores
        ms = SQLiteStorage("/tmp/test_bookmate_phase3.db")
        vs = ChromaDatabase("/tmp/test_chroma_phase3")
        
        # Setup
        setup_test_data(ms, vs)
        
        # Point global instances to test stores (Monkey patch implies we need to be careful, 
        # but here we can just rely on the fact that retrieving instances usually imports them. 
        # Actually better to instantiate Retrievers with these stores if possible, but they import globals.
        # We will monkey patch the modules.)
        
        import src.processors.retrieval.keyword_retriever as kr_mod
        kr_mod.metadata_store = ms
        
        import src.storage.vector_store as vs_mod
        vs_mod.chroma_db = vs
        import src.processors.retrieval.semantic_retriever as sr_mod
        sr_mod.chroma_db = vs
        
        # Run tests
        test_keyword_search()
        test_semantic_search()
        test_hybrid_search()
        test_reranker()
        
        # Cleanup
        if os.path.exists("/tmp/test_bookmate_phase3.db"):
            os.remove("/tmp/test_bookmate_phase3.db")
        # Cleaning up chroma dir is harder, leaving it for now
        
        logger.info("\n✅ Phase 3 Verification Complete!")
        
    except Exception as e:
        logger.error(f"\n❌ Phase 3 Verification Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
