import os
import time
from typing import List, Union, Optional
from dotenv import load_dotenv
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ sentence-transformers package is available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")

# Load .env from the specific path you mentioned
load_dotenv(dotenv_path=r"C:\Users\Darsh J Nishad\Desktop\synergiz\ai app\other-files\.env")

# Configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast model
# Alternative models you can use:
# "all-mpnet-base-v2" - Higher quality, larger size
# "all-distilroberta-v1" - Good balance
# "paraphrase-MiniLM-L6-v2" - Optimized for paraphrase detection

class LocalEmbedder:
    """Local sentence transformer embedder class"""
    
    def __init__(self, model_name: str = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package is required. Install with: pip install sentence-transformers")
        
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"🔄 Loading model: {self.model_name}")
            
            # Add device configuration for better performance
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🖥️ Using device: {device}")
            
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Get embedding dimension by encoding a test sentence
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            
            print(f"✅ Model loaded successfully!")
            print(f"   - Model: {self.model_name}")
            print(f"   - Embedding dimension: {self.embedding_dim}")
            print(f"   - Device: {device}")
            
        except Exception as e:
            print(f"❌ Failed to load model {self.model_name}: {e}")
            raise e
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Encode text(s) into embeddings"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Ensure consistent output format
        result = self.model.encode(texts, convert_to_numpy=True, **kwargs)
        
        # Convert to list format for consistency
        if isinstance(result, np.ndarray):
            if result.ndim == 1:  # Single embedding
                return result.tolist()
            else:  # Multiple embeddings
                return [emb.tolist() for emb in result]
        
        return result

# Initialize global embedder instance
embedder = None

def initialize_embedder(model_name: str = None) -> bool:
    """Initialize the global embedder instance"""
    global embedder
    try:
        print("🚀 Initializing local embedder...")
        embedder = LocalEmbedder(model_name)
        print("✅ Embedder initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize embedder: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def get_embedding(text: str, model: str = None, max_retries: int = 3) -> Optional[List[float]]:
    """
    Generate embedding for given text using local sentence transformer
    
    Args:
        text (str): Text to embed
        model (str): Model to use (will reinitialize if different from current)
        max_retries (int): Maximum number of retry attempts (kept for compatibility)
    
    Returns:
        List[float]: Embedding vector or None if failed
    """
    global embedder
    
    # Initialize embedder if not already done or if model changed
    if embedder is None or (model and model != embedder.model_name):
        print(f"🔄 Initializing embedder with model: {model or 'default'}")
        if not initialize_embedder(model):
            print("❌ Failed to initialize embedder")
            return None
    
    # Validate input
    if not text or not text.strip():
        print("⚠️ Warning: Empty text provided for embedding")
        return None
    
    # Clean the text
    cleaned_text = text.replace("\n", " ").replace("\r", " ").strip()
    
    # Handle very long texts
    if len(cleaned_text) > 5000:
        print(f"⚠️ Warning: Text length ({len(cleaned_text)}) is quite long, truncating to 5000 chars")
        cleaned_text = cleaned_text[:5000]
    
    try:
        # Generate embedding with retry logic
        for attempt in range(max_retries):
            try:
                embedding = embedder.encode(cleaned_text)
                
                # Validate embedding
                if embedding is None:
                    raise ValueError("Embedding returned None")
                
                if not isinstance(embedding, list):
                    raise ValueError(f"Embedding should be a list, got {type(embedding)}")
                
                if len(embedding) == 0:
                    raise ValueError("Embedding is empty")
                
                # Check for invalid values
                if not all(isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x) for x in embedding):
                    raise ValueError("Embedding contains invalid values")
                
                print(f"✅ Generated embedding with {len(embedding)} dimensions")
                return embedding
                
            except Exception as e:
                print(f"❌ Embedding attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying... ({attempt + 2}/{max_retries})")
                    time.sleep(1)
                else:
                    print(f"❌ All {max_retries} attempts failed")
                    return None
        
    except Exception as e:
        print(f"❌ Embedding generation failed completely: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return None

def get_embeddings_batch(texts: List[str], model: str = None, batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches
    
    Args:
        texts (List[str]): List of texts to embed
        model (str): Model to use
        batch_size (int): Number of texts to process at once
    
    Returns:
        List[List[float]]: List of embedding vectors
    """
    global embedder
    
    # Initialize embedder if not already done or if model changed
    if embedder is None or (model and model != embedder.model_name):
        if not initialize_embedder(model):
            print("❌ Failed to initialize embedder for batch processing")
            return []
    
    if not texts:
        print("⚠️ No texts provided for batch embedding")
        return []
    
    all_embeddings = []
    
    # Clean texts first
    cleaned_texts = []
    for i, text in enumerate(texts):
        if text and text.strip():
            cleaned_text = text.replace("\n", " ").replace("\r", " ").strip()
            # Truncate very long texts
            if len(cleaned_text) > 5000:
                cleaned_text = cleaned_text[:5000]
            cleaned_texts.append(cleaned_text)
        else:
            print(f"⚠️ Warning: Empty text at index {i}, using placeholder")
            cleaned_texts.append("empty text placeholder")
    
    total_batches = (len(cleaned_texts) + batch_size - 1) // batch_size
    print(f"🔢 Processing {len(cleaned_texts)} texts in {total_batches} batches of {batch_size}")
    
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"🔄 Processing batch {batch_num}/{total_batches}: {len(batch)} texts")
        
        try:
            # Generate embeddings for the batch
            batch_embeddings = embedder.encode(batch, batch_size=len(batch), show_progress_bar=False)
            
            # Validate batch embeddings
            if batch_embeddings is None:
                raise ValueError("Batch embeddings returned None")
            
            if not isinstance(batch_embeddings, list):
                raise ValueError(f"Batch embeddings should be a list, got {type(batch_embeddings)}")
            
            if len(batch_embeddings) != len(batch):
                raise ValueError(f"Batch size mismatch: expected {len(batch)}, got {len(batch_embeddings)}")
            
            all_embeddings.extend(batch_embeddings)
            print(f"✅ Batch {batch_num} complete: {len(batch_embeddings)} embeddings")
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            # Fallback: process individually
            print("🔄 Falling back to individual processing...")
            for j, text in enumerate(batch):
                try:
                    embedding = get_embedding(text, model)
                    if embedding:
                        all_embeddings.append(embedding)
                    else:
                        # Add zero vector as placeholder
                        print(f"⚠️ Using zero vector for failed text at batch {batch_num}, item {j}")
                        all_embeddings.append([0.0] * embedder.embedding_dim)
                except Exception as individual_error:
                    print(f"❌ Failed to embed individual text: {individual_error}")
                    # Add zero vector as placeholder
                    all_embeddings.append([0.0] * embedder.embedding_dim)
    
    print(f"✅ Total embeddings generated: {len(all_embeddings)}")
    return all_embeddings

def test_embedding_connection():
    """Test the embedding connection with a simple example"""
    try:
        print("🧪 Testing embedding connection...")
        test_text = "This is a test sentence for embedding."
        embedding = get_embedding(test_text)
        
        if embedding is None:
            print("❌ Test failed: get_embedding returned None")
            return False
        
        if not isinstance(embedding, list):
            print(f"❌ Test failed: expected list, got {type(embedding)}")
            return False
        
        if len(embedding) == 0:
            print("❌ Test failed: embedding is empty")
            return False
        
        print(f"✅ Test successful!")
        print(f"   - Text: '{test_text}'")
        print(f"   - Embedding dimension: {len(embedding)}")
        print(f"   - First 5 values: {embedding[:5]}")
        print(f"   - Model used: {embedder.model_name if embedder else 'None'}")
        
        # Test batch processing too
        print("🧪 Testing batch embedding...")
        batch_results = get_embeddings_batch([test_text, "Another test sentence"], batch_size=2)
        
        if batch_results and len(batch_results) == 2:
            print("✅ Batch test successful!")
        else:
            print(f"❌ Batch test failed: expected 2 embeddings, got {len(batch_results) if batch_results else 0}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def get_embedding_info():
    """Get information about the current embedding configuration"""
    global embedder
    
    if embedder is None:
        # Try to initialize with default model
        print("🔄 No embedder found, initializing default...")
        if not initialize_embedder():
            print("❌ No embedder initialized and failed to initialize default")
            return None
    
    try:
        model_name = embedder.model_name
        dimension = embedder.embedding_dim
        
        # Check if CUDA is available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device = embedder.model.device if hasattr(embedder.model, 'device') else 'unknown'
        except:
            cuda_available = False
            device = 'unknown'
        
        info = {
            "model": model_name,
            "dimension": dimension,
            "backend": "sentence-transformers",
            "model_loaded": embedder.model is not None,
            "cuda_available": cuda_available,
            "device": str(device)
        }
        
        print("📊 Embedding Configuration:")
        print(f"   - Model: {model_name}")
        print(f"   - Dimension: {dimension}")
        print(f"   - Backend: Local Sentence Transformers")
        print(f"   - Model loaded: {'Yes' if embedder.model else 'No'}")
        print(f"   - CUDA available: {'Yes' if cuda_available else 'No'}")
        print(f"   - Device: {device}")
        
        return info
        
    except Exception as e:
        print(f"❌ Error getting embedding info: {e}")
        return None

def list_available_models():
    """List some popular sentence transformer models"""
    models = {
        "all-MiniLM-L6-v2": "Fast, lightweight (80MB), 384 dimensions",
        "all-mpnet-base-v2": "High quality (420MB), 768 dimensions", 
        "all-distilroberta-v1": "Good balance (290MB), 768 dimensions",
        "paraphrase-MiniLM-L6-v2": "Optimized for paraphrases (80MB), 384 dimensions",
        "multi-qa-MiniLM-L6-cos-v1": "Question-answering optimized (80MB), 384 dimensions",
        "all-MiniLM-L12-v2": "Better quality than L6 (120MB), 384 dimensions"
    }
    
    print("📋 Available Sentence Transformer Models:")
    for model, description in models.items():
        print(f"   - {model}: {description}")
    
    return models

def change_model(model_name: str):
    """Change the current model"""
    global embedder
    try:
        print(f"🔄 Changing model to: {model_name}")
        embedder = LocalEmbedder(model_name)
        print(f"✅ Successfully changed to model: {model_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to change model: {e}")
        return False

def force_reinitialize():
    """Force reinitialize the embedder (useful for debugging)"""
    global embedder
    embedder = None
    return initialize_embedder()

# Initialize on module load
if __name__ == "__main__":
    print("🚀 Testing local embedder module...")
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        list_available_models()
        print("\n" + "="*50 + "\n")
        info = get_embedding_info()
        if info:
            test_embedding_connection()
        else:
            print("❌ Failed to get embedding info")
    else:
        print("❌ Please install sentence-transformers: pip install sentence-transformers")
else:
    # Quick initialization when imported
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            print(f"📦 Local embedder module loaded successfully")
            # Don't auto-initialize to save memory - initialize on first use
        except Exception as e:
            print(f"⚠️ Warning: Local embedder module loaded with issues: {e}")
    else:
        print("⚠️ sentence-transformers not available. Install with: pip install sentence-transformers")