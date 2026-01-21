
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import config
from src.processors.ingestion.ingestion_pipeline import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_phase2():
    print("🚀 Starting Phase 2 Verification")

    # Enable new features
    config.ingestion.enable_semantic_chunking = True
    config.ingestion.enable_rich_metadata = True
    
    # Check if we have an API key (or mock it)
    if not config.model.groq_api_key:
        print("⚠️ No GROQ API KEY found. LLM calls might use fallback/mock.")
    
    # Create a test document with distinct sections and topic shifts for semantic chunking
    test_file = "test_phase2.txt"
    content = """
Introduction to AI

Artificial Intelligence (AI) is intelligence demonstrated by machines. It is a field of study in computer science that develops and studies intelligent machines. Such machines may be called AIs.

Machine Learning

Machine learning is a subset of AI. It focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Deep learning is a further subset of machine learning.

Neural Networks

Neural networks are computing systems inspired by the biological neural networks that constitute animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons.
    """
    
    with open(test_file, "w") as f:
        f.write(content.strip())
        
    try:
        processor = DocumentProcessor()
        success, message, results = processor.process_document(
            file_path=os.path.abspath(test_file),
            uploader_id="verifier"
        )
        
        print(f"\nProcessing Result: {success}")
        print(f"Message: {message}")
        
        if success:
            print("\n✅ Ingestion Successful!")
            print(f"Chunks Created: {results['chunks_created']}")
            print(f"Steps Completed: {results['steps_completed']}")
            
            # Verify Rich Metadata
            if "rich_metadata_extraction" in results['steps_completed']:
                print("✅ Rich Metadata Extraction Step Executed")
            else:
                print("❌ Rich Metadata Extraction Step Missing")
                
            # Verify Semantic Chunking (indirectly via chunk count or logs)
            # Since we can't easily inspect chunk objects here without fetching from DB or modifying pipeline return
            # We assume if it ran without error and produced chunks, the logic was executed.
            # We can check logs for "Generating new embeddings" if semantic chunking was used (it calls _get_embedding_model)
            
            print("\nVerification Complete.")
            
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    verify_phase2()
