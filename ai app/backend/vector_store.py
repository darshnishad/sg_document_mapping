import os
import faiss
import numpy as np
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pathlib import Path
import time

# Import our local embedder
try:
    from .embedder import get_embedding, get_embeddings_batch, initialize_embedder, embedder
    LOCAL_EMBEDDER_AVAILABLE = True
except ImportError:
    try:
        # If running as standalone, try direct import
        from embedder import get_embedding, get_embeddings_batch, initialize_embedder, embedder
        LOCAL_EMBEDDER_AVAILABLE = True
    except ImportError:
        LOCAL_EMBEDDER_AVAILABLE = False
        print("⚠️ Local embedder not available. Please ensure embedder.py is accessible.")

# Load .env from: ai app/other-files/.env
load_dotenv(dotenv_path=r"C:\Users\Darsh J Nishad\Desktop\synergiz\ai app\other-files\.env")

# Read environment variables with defaults for local embedding models
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Default sentence transformer model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)

# Get embedding dimension based on model
MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-distilroberta-v1": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "all-MiniLM-L12-v2": 384,
    "text-embedding-3-small": 1536,  # For backward compatibility
    "text-embedding-ada-002": 1536   # For backward compatibility
}

EMBEDDING_DIM = os.getenv("EMBEDDING_DIM")
if EMBEDDING_DIM is None:
    # Auto-detect dimension based on model
    EMBEDDING_DIM = MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 384)
    print(f"🔍 Auto-detected embedding dimension: {EMBEDDING_DIM} for model: {EMBEDDING_MODEL}")
else:
    EMBEDDING_DIM = int(EMBEDDING_DIM)
    print(f"📋 Using configured embedding dimension: {EMBEDDING_DIM}")

VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "data/processed/faiss_index.index")
VECTOR_META_PATH = os.getenv("VECTOR_META_PATH", "data/processed/faiss_metadata.json")

# Global FAISS index and metadata store
index = None
metadata_store = []

def initialize_index():
    """Initialize a new FAISS index"""
    global index, metadata_store
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata_store = []  # Reset metadata store
    print(f"✅ Initialized FAISS index with dimension {EMBEDDING_DIM}")

def ensure_embedder_initialized():
    """Ensure the local embedder is initialized"""
    if LOCAL_EMBEDDER_AVAILABLE:
        try:
            # Check if embedder is already initialized
            if embedder is None:
                initialize_embedder(EMBEDDING_MODEL)
            elif embedder.model_name != EMBEDDING_MODEL:
                # Reinitialize if model changed
                initialize_embedder(EMBEDDING_MODEL)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize embedder: {e}")
            return False
    else:
        print("❌ Local embedder not available")
        return False

def add_to_index(embedding: List[float], metadata: Dict):
    """Add a vector and its metadata to the FAISS index"""
    global index, metadata_store
    
    # Validate inputs
    if index is None:
        print("⚠️ Index not initialized. Initializing now...")
        initialize_index()
    
    if not embedding:
        raise ValueError("❌ Embedding cannot be empty")
    
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(f"❌ Embedding dimension mismatch. Expected {EMBEDDING_DIM}, got {len(embedding)}")
    
    try:
        vector = np.array([embedding], dtype='float32')
        index.add(vector)
        metadata_store.append(metadata)
        print(f"➕ Added vector. Total vectors: {len(metadata_store)}")
    except Exception as e:
        print(f"❌ Error adding vector to index: {e}")
        raise e

def add_multiple_to_index(embeddings: List[List[float]], metadatas: List[Dict]):
    """Add multiple vectors and their metadata to the FAISS index (batch operation)"""
    global index, metadata_store
    
    if index is None:
        print("⚠️ Index not initialized. Initializing now...")
        initialize_index()
    
    if len(embeddings) != len(metadatas):
        raise ValueError("❌ Number of embeddings and metadata must match")
    
    if not embeddings:
        print("⚠️ No embeddings to add")
        return
    
    try:
        # Validate all embeddings have correct dimension
        for i, embedding in enumerate(embeddings):
            if len(embedding) != EMBEDDING_DIM:
                raise ValueError(f"❌ Embedding {i} dimension mismatch. Expected {EMBEDDING_DIM}, got {len(embedding)}")
        
        # Convert to numpy array and add to index
        vectors = np.array(embeddings, dtype='float32')
        index.add(vectors)
        metadata_store.extend(metadatas)
        
        print(f"➕ Added {len(embeddings)} vectors. Total vectors: {len(metadata_store)}")
        
    except Exception as e:
        print(f"❌ Error adding multiple vectors to index: {e}")
        raise e

def add_chunks_to_index(chunk_ids: List[str], embeddings: List[List[float]]):
    """
    Add chunks with their IDs to the FAISS index
    This is a wrapper around add_multiple_to_index with proper metadata formatting
    """
    if not chunk_ids or not embeddings:
        raise ValueError("❌ chunk_ids and embeddings cannot be empty")
    
    if len(chunk_ids) != len(embeddings):
        raise ValueError(f"❌ Mismatch: {len(chunk_ids)} chunk_ids vs {len(embeddings)} embeddings")
    
    # Create metadata for each chunk
    metadatas = []
    for chunk_id in chunk_ids:
        metadata = {
            "chunk_id": chunk_id,
            "type": "text_chunk",
            "embedding_model": EMBEDDING_MODEL,
            "created_at": time.time()
        }
        metadatas.append(metadata)
    
    # Use the existing add_multiple_to_index function
    add_multiple_to_index(embeddings, metadatas)
    print(f"✅ Added {len(chunk_ids)} chunks to index")

def add_texts_to_index(texts: List[str], metadatas: Optional[List[Dict]] = None, 
                      chunk_ids: Optional[List[str]] = None):
    """
    Add texts directly to the index by generating embeddings locally
    
    Args:
        texts (List[str]): List of texts to embed and add
        metadatas (List[Dict], optional): Metadata for each text
        chunk_ids (List[str], optional): IDs for each text chunk
    """
    if not texts:
        raise ValueError("❌ texts cannot be empty")
    
    if not ensure_embedder_initialized():
        raise RuntimeError("❌ Cannot initialize local embedder")
    
    print(f"🔄 Generating embeddings for {len(texts)} texts using local model...")
    
    try:
        # Generate embeddings using local model
        embeddings = get_embeddings_batch(texts, model=EMBEDDING_MODEL)
        
        if not embeddings:
            raise ValueError("❌ Failed to generate embeddings")
        
        # Create metadata if not provided
        if metadatas is None:
            metadatas = []
            for i, text in enumerate(texts):
                chunk_id = chunk_ids[i] if chunk_ids else f"chunk_{i}_{int(time.time())}"
                metadata = {
                    "chunk_id": chunk_id,
                    "type": "text_chunk",
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "text_length": len(text),
                    "embedding_model": EMBEDDING_MODEL,
                    "created_at": time.time()
                }
                metadatas.append(metadata)
        
        # Add to index
        add_multiple_to_index(embeddings, metadatas)
        print(f"✅ Successfully added {len(texts)} texts to index")
        
    except Exception as e:
        print(f"❌ Error adding texts to index: {e}")
        raise e

def save_index():
    """Save the FAISS index and metadata to disk"""
    global index, metadata_store
    
    if index is None:
        print("⚠️ No index to save")
        return
    
    if len(metadata_store) == 0:
        print("⚠️ No vectors in index to save")
        return
    
    try:
        # Create directories if they don't exist
        Path(VECTOR_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(VECTOR_META_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(VECTOR_INDEX_PATH))
        
        # Save metadata with additional info
        metadata_to_save = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "created_at": time.time(),
            "total_vectors": len(metadata_store),
            "metadata": metadata_store
        }
        
        with open(VECTOR_META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"💾 FAISS index saved to {VECTOR_INDEX_PATH}")
        print(f"💾 Metadata saved to {VECTOR_META_PATH}")
        print(f"💾 Saved {len(metadata_store)} vectors")
        
    except Exception as e:
        print(f"❌ Error saving index: {e}")
        raise e

def load_index():
    """Load the FAISS index and metadata from disk"""
    global index, metadata_store
    
    # Check if files exist
    if not os.path.exists(VECTOR_INDEX_PATH):
        print(f"⚠️ FAISS index not found at {VECTOR_INDEX_PATH}. Creating new index...")
        initialize_index()
        return
    
    if not os.path.exists(VECTOR_META_PATH):
        print(f"⚠️ Metadata file not found at {VECTOR_META_PATH}. Creating new index...")
        initialize_index()
        return
    
    try:
        # Load FAISS index
        index = faiss.read_index(str(VECTOR_INDEX_PATH))
        
        # Load metadata
        with open(VECTOR_META_PATH, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        
        # Handle both old and new metadata formats
        if isinstance(saved_data, list):
            # Old format: just the metadata list
            metadata_store = saved_data
            saved_model = "unknown"
            saved_dim = EMBEDDING_DIM
        else:
            # New format: structured data
            metadata_store = saved_data.get("metadata", [])
            saved_model = saved_data.get("embedding_model", "unknown")
            saved_dim = saved_data.get("embedding_dimension", EMBEDDING_DIM)
        
        # Validate consistency
        if index.ntotal != len(metadata_store):
            print(f"⚠️ Warning: Index has {index.ntotal} vectors but metadata has {len(metadata_store)} entries")
        
        # Check model compatibility
        if saved_model != EMBEDDING_MODEL and saved_model != "unknown":
            print(f"⚠️ Warning: Saved model ({saved_model}) differs from current model ({EMBEDDING_MODEL})")
        
        # Check dimension compatibility
        if saved_dim != EMBEDDING_DIM:
            print(f"⚠️ Warning: Saved dimension ({saved_dim}) differs from current dimension ({EMBEDDING_DIM})")
        
        print(f"✅ FAISS index and metadata loaded. Total vectors: {len(metadata_store)}")
        print(f"📊 Loaded model: {saved_model}, dimension: {saved_dim}")
        
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        print("🔄 Creating new index...")
        initialize_index()
        raise e

def search_index(query_embedding: List[float], top_k: int = 5, return_scores: bool = False) -> List[Dict]:
    """Search the FAISS index using the provided query embedding"""
    global index, metadata_store
    
    if index is None:
        print("⚠️ Index not loaded. Attempting to load...")
        load_index()
    
    if index.ntotal == 0:
        print("⚠️ Index is empty")
        return []
    
    # Validate query embedding
    if len(query_embedding) != EMBEDDING_DIM:
        raise ValueError(f"❌ Query embedding dimension mismatch. Expected {EMBEDDING_DIM}, got {len(query_embedding)}")
    
    try:
        query = np.array([query_embedding], dtype='float32')
        distances, indices = index.search(query, min(top_k, index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata_store) and idx != -1:  # -1 indicates no match found
                result = metadata_store[idx].copy()
                if return_scores:
                    result['distance'] = float(distances[0][i])
                    result['similarity_score'] = 1 / (1 + float(distances[0][i]))  # Convert distance to similarity
                results.append(result)
        
        print(f"🔍 Found {len(results)} results for query")
        return results
        
    except Exception as e:
        print(f"❌ Error searching index: {e}")
        raise e

def search_by_text(query_text: str, top_k: int = 5, return_scores: bool = False) -> List[Dict]:
    """
    FIXED: Search using get_similar_chunks for consistency
    """
    if not query_text.strip():
        raise ValueError("❌ Query text cannot be empty")
    
    # Just use get_similar_chunks
    results = get_similar_chunks(query_text, top_k=top_k, min_similarity=0.0)
    
    # Ensure scores are included if requested
    if return_scores:
        for result in results:
            if 'similarity_score' not in result and 'distance' in result:
                result['similarity_score'] = max(0.0, 1.0 - result['distance'])
    
    return results

def get_index_stats() -> Dict:
    """Get statistics about the current index"""
    global index, metadata_store
    
    stats = {
        "total_vectors": len(metadata_store) if metadata_store else 0,
        "index_dimension": EMBEDDING_DIM,
        "embedding_model": EMBEDDING_MODEL,
        "index_type": type(index).__name__ if index else "None",
        "index_file_exists": os.path.exists(VECTOR_INDEX_PATH),
        "metadata_file_exists": os.path.exists(VECTOR_META_PATH),
        "index_ntotal": index.ntotal if index else 0,
        "local_embedder_available": LOCAL_EMBEDDER_AVAILABLE
    }
    
    if metadata_store:
        # Get source document statistics
        doc_types = {}
        source_docs = {}
        embedding_models = {}
        
        for meta in metadata_store:
            doc_type = meta.get('doc_type', meta.get('type', 'unknown'))
            source_doc = meta.get('source_doc', 'unknown')
            emb_model = meta.get('embedding_model', 'unknown')
            
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            source_docs[source_doc] = source_docs.get(source_doc, 0) + 1
            embedding_models[emb_model] = embedding_models.get(emb_model, 0) + 1
        
        stats['doc_types'] = doc_types
        stats['source_documents'] = len(source_docs)
        stats['embedding_models_used'] = embedding_models
        stats['top_documents'] = dict(sorted(source_docs.items(), key=lambda x: x[1], reverse=True)[:5])
    
    return stats

def clear_index():
    """Clear the current index and metadata"""
    global index, metadata_store
    
    print("🗑️ Clearing index and metadata...")
    initialize_index()
    
    # Optionally remove files from disk
    try:
        if os.path.exists(VECTOR_INDEX_PATH):
            os.remove(VECTOR_INDEX_PATH)
            print(f"🗑️ Removed {VECTOR_INDEX_PATH}")
        
        if os.path.exists(VECTOR_META_PATH):
            os.remove(VECTOR_META_PATH)
            print(f"🗑️ Removed {VECTOR_META_PATH}")
            
    except Exception as e:
        print(f"⚠️ Error removing files: {e}")

def rebuild_index_from_texts(texts: List[str], metadatas: Optional[List[Dict]] = None):
    """
    Rebuild the entire index from a list of texts (generates new embeddings)
    
    Args:
        texts (List[str]): List of texts to rebuild index from
        metadatas (List[Dict], optional): Metadata for each text
    """
    print("🔧 Rebuilding index from texts...")
    
    # Clear existing index
    clear_index()
    
    # Add texts with new embeddings
    add_texts_to_index(texts, metadatas)
    
    # Save the rebuilt index
    save_index()
    
    print(f"✅ Index rebuilt with {len(texts)} texts")

def get_embeddings_batch_local(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Get embeddings for multiple texts using local sentence transformers.
    This replaces the OpenAI API version.
    
    Args:
        texts (List[str]): List of texts to embed
        batch_size (int): Number of texts to process per batch
    
    Returns:
        List[List[float]]: List of embedding vectors
    """
    if not ensure_embedder_initialized():
        raise RuntimeError("❌ Cannot initialize local embedder")
    
    return get_embeddings_batch(texts, model=EMBEDDING_MODEL, batch_size=batch_size)

# Backward compatibility - alias to local version
get_embeddings_batch = get_embeddings_batch_local

def test_vector_store():
    """Test the vector store functionality"""
    print("🧪 Testing vector store...")
    
    if not ensure_embedder_initialized():
        print("❌ Cannot test without local embedder")
        return False
    
    try:
        # Initialize
        initialize_index()
        
        # Test with actual text embedding
        test_text = "This is a test document for vector search."
        test_metadata = {"test": "data", "chunk_id": "test_123", "doc_type": "test"}
        
        # Add text directly (will generate embedding)
        add_texts_to_index([test_text], [test_metadata])
        
        # Test search by text
        results = search_by_text("test document", top_k=1, return_scores=True)
        
        if results and results[0]["chunk_id"] == "test_123":
            print("✅ Vector store test passed!")
            print(f"   - Similarity score: {results[0].get('similarity_score', 'N/A')}")
            
            # Clean up test data
            clear_index()
            return True
        else:
            print("❌ Vector store test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Vector store test error: {e}")
        return False

def get_similar_chunks(query_text: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
    """
    WORKING VERSION: Get similar chunks with proper file handling
    """
    try:
        print(f"🔍 get_similar_chunks called with query: '{query_text}'")
        
        # Use the same paths as your VECTOR_INDEX_PATH and VECTOR_META_PATH
        index_path = VECTOR_INDEX_PATH  # Should be "data/processed/faiss_index.index"
        metadata_path = VECTOR_META_PATH  # Should be "data/processed/faiss_metadata.json"
        
        print(f"Looking for files:")
        print(f"  Index: {index_path}")
        print(f"  Metadata: {metadata_path}")
        
        # Check if files exist
        if not os.path.exists(index_path):
            print(f"❌ Index file not found: {index_path}")
            return []
            
        if not os.path.exists(metadata_path):
            print(f"❌ Metadata file not found: {metadata_path}")
            return []
        
        print("✅ Both files exist, proceeding...")
        
        # Load FAISS index
        import faiss
        index = faiss.read_index(index_path)
        print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
        
        if index.ntotal == 0:
            print("❌ Index has 0 vectors")
            return []
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_content = json.load(f)
        
        # Handle different metadata formats
        if isinstance(metadata_content, list):
            # Direct list format
            metadata_list = metadata_content
            print(f"✅ Loaded metadata list with {len(metadata_list)} entries")
        elif isinstance(metadata_content, dict):
            if 'metadata' in metadata_content:
                # Wrapped format
                metadata_list = metadata_content['metadata']
                print(f"✅ Loaded wrapped metadata with {len(metadata_list)} entries")
            else:
                # Dict with string keys (index -> metadata)
                metadata_list = []
                for i in range(index.ntotal):
                    if str(i) in metadata_content:
                        metadata_list.append(metadata_content[str(i)])
                    else:
                        metadata_list.append({'chunk_id': f'chunk_{i}', 'content_preview': 'No content'})
                print(f"✅ Converted dict metadata to list with {len(metadata_list)} entries")
        else:
            print(f"❌ Unknown metadata format: {type(metadata_content)}")
            return []
        
        # Ensure embedder is available
        if not ensure_embedder_initialized():
            print("❌ Embedder not initialized")
            return []
        
        # Generate embedding for query
        query_embedding = get_embedding(query_text, model=EMBEDDING_MODEL)
        if query_embedding is None:
            print("❌ Failed to generate embedding")
            return []
        
        print(f"✅ Generated embedding with {len(query_embedding)} dimensions")
        
        # Prepare query vector
        import numpy as np
        if isinstance(query_embedding, list):
            query_vector = np.array([query_embedding], dtype=np.float32)
        else:
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Perform FAISS search
        distances, indices = index.search(query_vector, k=min(top_k, index.ntotal))
        print(f"✅ FAISS search completed")
        print(f"   Distances: {distances[0]}")
        print(f"   Indices: {indices[0]}")
        
        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            idx = int(idx)
            distance = float(distances[0][i])
            similarity = max(0.0, 1.0 - distance)  # Convert distance to similarity
            
            print(f"Processing result {i}: idx={idx}, distance={distance:.4f}, similarity={similarity:.4f}")
            
            # Check similarity threshold
            if similarity < min_similarity:
                print(f"  Skipping due to low similarity ({similarity:.4f} < {min_similarity})")
                continue
            
            # Get metadata
            if idx < len(metadata_list):
                meta = metadata_list[idx]
            else:
                print(f"  ⚠️ Index {idx} out of range (metadata has {len(metadata_list)} entries)")
                meta = {'chunk_id': f'chunk_{idx}', 'content_preview': 'Metadata missing'}
            
            # Build result - try multiple field names for content
            content_text = (
                meta.get('content_preview') or 
                meta.get('text_preview') or 
                meta.get('text') or 
                meta.get('content') or 
                'No content available'
            )
            
            result = {
                'similarity_score': round(similarity, 4),
                'chunk_id': meta.get('chunk_id', f'chunk_{idx}'),
                'source_doc': meta.get('source_doc', meta.get('filename', 'Unknown')),
                'source_table': meta.get('source_table', meta.get('doc_type', 'Unknown')),
                'text': content_text,
                'content_preview': content_text,
                'filename': meta.get('source_doc', meta.get('filename', 'Unknown')),
                'page': meta.get('page', 'N/A'),
                'chunk_text': content_text,
                'distance': distance
            }
            
            results.append(result)
            print(f"  ✅ Added result: {result['source_doc']} (similarity: {similarity:.4f})")
        
        print(f"✅ Returning {len(results)} results")
        return results
        
    except Exception as e:
        print(f"❌ Error in get_similar_chunks: {e}")
        import traceback
        traceback.print_exc()
        return []