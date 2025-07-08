import sys
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import streamlit as st
import urllib.parse
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker
import numpy as np

# Load environment variables
load_dotenv(dotenv_path=r"C:\Users\Darsh J Nishad\Desktop\synergiz\ai app\other-files\.env")

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ================== IMPORTS AND INITIALIZATION ==================

# Import backend modules
try:
    from backend.embedder import (
        get_embedding, get_embeddings_batch, initialize_embedder,
        get_embedding_info, test_embedding_connection
    )
    from backend.storage import test_storage_connection
    from backend.vector_store import (
        save_index, load_index, get_index_stats, 
        add_to_index, initialize_index
    )
    from backend.query_handler import enhanced_answer_query
    
    LOCAL_MODULES_AVAILABLE = True
    print("✅ All backend modules imported successfully")
    
    # Initialize modules
    initialize_embedder()
    initialize_index()
    
except ImportError as e:
    LOCAL_MODULES_AVAILABLE = False
    print(f"❌ Backend modules not available: {e}")

# Check OpenAI availability
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False

# Set availability flags
EMBEDDER_AVAILABLE = LOCAL_MODULES_AVAILABLE
EMBEDDER_TYPE = "Local Sentence Transformers" if EMBEDDER_AVAILABLE else "Not Available"

# ================== SMART SEARCH SYSTEM ==================

class DynamicQueryExpander:
    """Smart query expansion system that learns from your documents"""
    
    def __init__(self):
        self.data_dir = Path("data/processed")
        self.cache_file = self.data_dir / "query_expansions.json"
        self.term_associations = defaultdict(set)
        self.document_vocabulary = set()
        self.context_patterns = {}
        self.is_initialized = False
        
        self._ensure_directories()
        self._safe_load_expansions()
    
    def _ensure_directories(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ Could not create directory {self.data_dir}: {e}")
    
    def _safe_load_expansions(self):
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.document_vocabulary = set(data.get('document_vocabulary', []))
                for k, v in data.get('term_associations', {}).items():
                    self.term_associations[k] = set(v)
                self.context_patterns = data.get('context_patterns', {})
                self.is_initialized = len(self.document_vocabulary) > 0
                print(f"✅ Loaded smart search vocabulary: {len(self.document_vocabulary)} terms")
            else:
                self.is_initialized = False
                print("📝 No smart search vocabulary found")
        except Exception as e:
            print(f"⚠️ Could not load smart search vocabulary: {e}")
            self.is_initialized = False
    
    def _safe_save_expansions(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            data = {
                'document_vocabulary': list(self.document_vocabulary),
                'term_associations': {k: list(v) for k, v in self.term_associations.items()},
                'context_patterns': self.context_patterns,
                'timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.cache_file)
            
            print(f"✅ Smart search vocabulary saved: {len(self.document_vocabulary)} terms")
            self.is_initialized = True
        except Exception as e:
            print(f"❌ Could not save smart search vocabulary: {e}")
    
    def analyze_document_content(self, force_rebuild=False):
        if not force_rebuild and self.is_initialized:
            if self.cache_file.exists():
                cache_age = datetime.now().timestamp() - self.cache_file.stat().st_mtime
                if cache_age < 86400:
                    st.info("📚 Using recent smart search vocabulary")
                    return True
        
        st.info("🧠 Building smart search vocabulary...")
        progress_bar = st.progress(0)
        
        try:
            engine = get_database_connection()
            if not engine:
                st.error("❌ Database connection failed")
                return False
            
            progress_bar.progress(10)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            all_text_content = []
            tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
            
            for i, table in enumerate(tables):
                try:
                    query = sql_text(f"""
                        SELECT chunk_text 
                        FROM {table} 
                        WHERE chunk_text IS NOT NULL 
                        AND LEN(chunk_text) > 50
                        AND chunk_text != 'No content available'
                    """)
                    
                    result = session.execute(query)
                    for row in result:
                        text = row[0]
                        if text and len(text.strip()) > 50:
                            all_text_content.append(text.lower())
                    
                    progress_bar.progress(10 + (i + 1) * 15)
                except Exception as e:
                    continue
            
            session.close()
            
            if not all_text_content:
                st.error("❌ No valid content found")
                return False
            
            progress_bar.progress(60)
            self._build_vocabulary(all_text_content)
            progress_bar.progress(80)
            self._find_term_associations(all_text_content)
            progress_bar.progress(90)
            self._identify_context_patterns(all_text_content)
            progress_bar.progress(100)
            self._safe_save_expansions()
            
            st.balloons()
            st.success(f"🧠 Smart Search Ready! Built vocabulary with {len(self.document_vocabulary)} terms")
            return True
            
        except Exception as e:
            st.error(f"❌ Smart search analysis failed: {e}")
            return False
        finally:
            progress_bar.empty()
    
    def _build_vocabulary(self, texts):
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'
        }
        
        word_freq = Counter()
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            for word in words:
                if word not in common_words and len(word) > 2:
                    word_freq[word] += 1
        
        self.document_vocabulary = {term for term, freq in word_freq.items() if freq >= 2}
    
    def _find_term_associations(self, texts):
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()) 
                        if w in self.document_vocabulary]
                
                for i, word1 in enumerate(words):
                    for word2 in words[i+1:i+6]:
                        if word1 != word2:
                            self.term_associations[word1].add(word2)
                            self.term_associations[word2].add(word1)
    
    def _identify_context_patterns(self, texts):
        patterns = {
            'payment_terms': r'payment.*?(?:terms|due|amount|schedule)',
            'termination': r'terminat(?:e|ion).*?(?:clause|condition|notice)',
            'liability': r'liabilit(?:y|ies).*?(?:limit|exclude|include)',
            'delivery': r'deliver(?:y|ies).*?(?:date|schedule|requirement)'
        }
        
        for pattern_name, pattern in patterns.items():
            matches = []
            for text in texts:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            if matches:
                self.context_patterns[pattern_name] = Counter(matches).most_common(3)
    
    def expand_query(self, query_text, max_expansions=5):
        if not self.is_initialized:
            return query_text, query_text, []
        
        query_lower = query_text.lower()
        original_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
        expanded_terms = set(original_words)
        
        for word in original_words:
            if word in self.term_associations:
                associations = list(self.term_associations[word])[:max_expansions]
                expanded_terms.update(associations)
        
        expansion_words = expanded_terms - original_words
        
        if expansion_words:
            expansion_words = list(expansion_words)[:max_expansions]
            expanded_query = query_text + " " + " ".join(expansion_words)
        else:
            expanded_query = query_text
        
        return query_text, expanded_query, expansion_words
    
    def get_expansion_stats(self):
        return {
            'vocabulary_size': len(self.document_vocabulary),
            'associations_count': len(self.term_associations),
            'patterns_identified': len(self.context_patterns),
            'is_initialized': self.is_initialized
        }

@st.cache_resource
def get_smart_search_expander():
    return DynamicQueryExpander()

# ================== CORE UTILITY FUNCTIONS ==================

def get_database_connection():
    """Database connection function"""
    try:
        params = urllib.parse.quote_plus(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=SZLP112;'
            'DATABASE=Clause;'
            'Trusted_Connection=yes;'
        )
        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
        return engine
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def get_vector_index_paths():
    """Get paths for vector index files"""
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return (
        str(data_dir / "faiss_index.index"),
        str(data_dir / "faiss_metadata.json"),
        str(data_dir)
    )

def get_available_documents():
    """Get all available documents from database"""
    try:
        engine = get_database_connection()
        if not engine:
            return []
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        all_docs = []
        tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
        
        for table in tables:
            try:
                query = sql_text(f"SELECT DISTINCT source_doc FROM {table}")
                result = session.execute(query)
                for row in result:
                    all_docs.append({
                        'filename': row[0],
                        'table': table,
                        'doc_type': table.replace('chunks_', '')
                    })
            except:
                continue
        
        session.close()
        return all_docs
    except Exception as e:
        st.error(f"Error getting documents: {e}")
        return []

def get_system_status():
    """Get system status"""
    status = {
        'database': False,
        'embedder': False,
        'vector_store': False,
        'total_vectors': 0,
        'errors': []
    }
    
    # Test database
    try:
        engine = get_database_connection()
        status['database'] = engine is not None
    except Exception as e:
        status['errors'].append(f"Database: {e}")
    
    # Test embedder
    if EMBEDDER_AVAILABLE:
        try:
            test_embedding = get_embedding("test")
            status['embedder'] = test_embedding is not None
        except Exception as e:
            status['errors'].append(f"Embedder: {e}")
    
    # Test vector store
    try:
        index_path, _, _ = get_vector_index_paths()
        if os.path.exists(index_path):
            import faiss
            index = faiss.read_index(index_path)
            status['total_vectors'] = index.ntotal
            status['vector_store'] = True
    except Exception as e:
        status['errors'].append(f"Vector store: {e}")
    
    return status

# ================== MAIN SEARCH FUNCTION ==================

def search_documents(query_text, top_k=5, use_smart_search=False, max_expansions=5):
    """Main search function with smart search integration"""
    
    if not EMBEDDER_AVAILABLE:
        st.error("❌ Embedder not available")
        return []
    
    # Smart search expansion
    final_query = query_text
    expansion_words = []
    
    if use_smart_search:
        try:
            expander = get_smart_search_expander()
            if expander.is_initialized:
                original_query, expanded_query, expansion_words = expander.expand_query(query_text, max_expansions)
                if expansion_words:
                    final_query = expanded_query
                    st.info(f"🧠 **Smart Search:** Added terms: {', '.join(expansion_words[:5])}")
        except Exception as e:
            st.warning(f"Smart search failed: {e}")
    
    try:
        # Get file paths
        index_path, metadata_path, _ = get_vector_index_paths()
        
        # Check files exist
        if not os.path.exists(index_path):
            st.error(f"❌ Index file not found: {index_path}")
            return []
        
        if not os.path.exists(metadata_path):
            st.error(f"❌ Metadata file not found: {metadata_path}")
            return []
        
        # Load files
        import faiss
        index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r', encoding='utf-8', errors='ignore') as f:
            metadata_raw = json.load(f)
        
        # Handle metadata format
        if isinstance(metadata_raw, dict) and 'metadata' in metadata_raw:
            metadata = metadata_raw['metadata']
        elif isinstance(metadata_raw, list):
            metadata = metadata_raw
        else:
            metadata = list(metadata_raw.values())
        
        # Generate embedding for final query
        query_embedding = get_embedding(final_query)
        if not query_embedding:
            st.error("❌ Failed to generate embedding")
            return []
        
        # Search
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = index.search(query_vector, min(top_k, index.ntotal))
        
        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
                
            meta = metadata[idx]
            similarity = float(1 - distances[0][i])
            
            # Get content - try multiple fields
            content = None
            for field in ['text', 'chunk_text', 'content', 'full_text']:
                if field in meta and meta[field]:
                    content = str(meta[field]).strip()
                    if content and content not in ['No content available', 'Unable to extract']:
                        break
            
            if not content:
                content = f"[Content from {meta.get('source_doc', 'unknown')}]"
            
            result = {
                'similarity_score': round(similarity, 4),
                'chunk_id': meta.get('chunk_id', f'chunk_{idx}'),
                'source_doc': meta.get('source_doc', 'Unknown'),
                'filename': meta.get('source_doc', 'Unknown'),
                'text': content,
                'content_length': len(content),
                'expansion_words': expansion_words,
                'used_smart_search': use_smart_search and len(expansion_words) > 0
            }
            
            results.append(result)
        
        return results
        
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        return []

def get_content_from_database(chunk_id, source_doc):
    """Fallback function to get content directly from database"""
    try:
        engine = get_database_connection()
        if not engine:
            return None
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
        
        for table in tables:
            try:
                query = sql_text(f"""
                    SELECT chunk_text 
                    FROM {table} 
                    WHERE chunk_id = :chunk_id 
                    AND source_doc = :source_doc
                    AND chunk_text IS NOT NULL
                    AND LEN(chunk_text) > 10
                """)
                
                result = session.execute(query, {"chunk_id": chunk_id, "source_doc": source_doc})
                row = result.fetchone()
                
                if row and row[0] and row[0].strip():
                    content = row[0].strip()
                    if content not in ['No content available', 'Unable to extract']:
                        session.close()
                        return content
                        
            except Exception:
                continue
        
        session.close()
        return None
        
    except Exception:
        return None

# ================== VECTORIZATION FUNCTIONS ==================

def vectorize_document(filename):
    """Vectorize a single document"""
    
    if not EMBEDDER_AVAILABLE:
        st.error("❌ Embedder not available")
        return False
    
    try:
        st.info(f"🔄 Vectorizing {filename}...")
        
        # Get content from database
        engine = get_database_connection()
        if not engine:
            return False
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        all_chunks = []
        tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
        
        for table in tables:
            try:
                query = sql_text(f"""
                    SELECT chunk_id, source_doc, chunk_text
                    FROM {table} 
                    WHERE source_doc = :filename
                    AND chunk_text IS NOT NULL 
                    AND LEN(chunk_text) > 20
                """)
                
                result = session.execute(query, {"filename": filename})
                
                for row in result:
                    chunk_id, source_doc, chunk_text = row
                    clean_text = chunk_text.strip()
                    
                    if len(clean_text) > 20:
                        all_chunks.append({
                            'chunk_id': chunk_id,
                            'filename': source_doc,
                            'text': clean_text,
                            'source_table': table
                        })
            except:
                continue
        
        session.close()
        
        if not all_chunks:
            st.error(f"❌ No valid content found for {filename}")
            return False
        
        st.success(f"✅ Found {len(all_chunks)} chunks")
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = get_embeddings_batch(texts, batch_size=16)
        
        if not embeddings or len(embeddings) != len(texts):
            st.error("❌ Embedding generation failed")
            return False
        
        # Add to index
        for embedding, chunk in zip(embeddings, all_chunks):
            metadata = {
                "chunk_id": str(chunk['chunk_id']),
                "source_doc": chunk['filename'],
                "source_table": chunk['source_table'],
                "text": chunk['text'],
                "chunk_text": chunk['text'],
                "filename": chunk['filename']
            }
            add_to_index(embedding, metadata)
        
        # Save index
        save_index()
        
        st.success(f"✅ {filename} vectorized successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Vectorization failed: {e}")
        return False

def vectorize_all_documents():
    """Vectorize all documents"""
    
    available_docs = get_available_documents()
    unique_docs = list(set([doc['filename'] for doc in available_docs]))
    
    if not unique_docs:
        st.error("❌ No documents found")
        return False
    
    st.info(f"🎯 Vectorizing {len(unique_docs)} documents...")
    
    success_count = 0
    progress_bar = st.progress(0)
    
    for i, doc in enumerate(unique_docs):
        if vectorize_document(doc):
            success_count += 1
        progress_bar.progress((i + 1) / len(unique_docs))
    
    if success_count == len(unique_docs):
        st.balloons()
        st.success(f"🎉 All {success_count} documents vectorized!")
        return True
    else:
        st.warning(f"⚠️ {success_count}/{len(unique_docs)} documents completed")
        return False

# ================== STREAMLIT APP ==================

st.set_page_config(
    page_title="AI Contract Assistant", 
    layout="wide",
    page_icon="📄"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-good { background-color: #d4edda; padding: 0.5rem; border-radius: 0.5rem; }
    .status-warning { background-color: #fff3cd; padding: 0.5rem; border-radius: 0.5rem; }
    .status-error { background-color: #f8d7da; padding: 0.5rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

# Get document info
available_docs = get_available_documents()
doc_count = len(set([doc['filename'] for doc in available_docs])) if available_docs else 0

# ================== SIDEBAR ==================

with st.sidebar:
    st.header("🔧 System Status")
    
    # Get system status
    status = get_system_status()
    
    # Embedder status
    if EMBEDDER_AVAILABLE:
        try:
            embedding_info = get_embedding_info()
            model_name = embedding_info.get('model', 'Unknown') if embedding_info else 'Unknown'
            st.markdown(f'<div class="status-good">✅ <strong>Embedder:</strong> {model_name}</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="status-warning">⚠️ Embedder: Loading...</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">❌ Embedder: Not Available</div>', unsafe_allow_html=True)
    
    # Database status
    if status['database']:
        st.markdown('<div class="status-good">✅ Database: Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">❌ Database: Failed</div>', unsafe_allow_html=True)
    
    # Vector status
    vector_count = status['total_vectors']
    if vector_count > 0:
        st.markdown(f'<div class="status-good">✅ Vectors: {vector_count}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">⚠️ Vectors: None</div>', unsafe_allow_html=True)
    
    # OpenAI status
    if OPENAI_AVAILABLE:
        st.markdown('<div class="status-good">✅ OpenAI: Available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">⚠️ OpenAI: Not Configured</div>', unsafe_allow_html=True)
    
    st.metric("📄 Documents", doc_count)
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 🔧 Quick Actions")
    
    if st.button("🧪 Test Search", use_container_width=True):
        if vector_count > 0:
            test_results = search_documents("contract", top_k=3)
            if test_results:
                st.success(f"✅ Found {len(test_results)} results")
            else:
                st.error("❌ No results found")
        else:
            st.warning("⚠️ No vectors available")
    
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
        # Smart Search Section
    st.markdown("---")
    st.markdown("### 🧠 Smart Search System")
    
    try:
        expander = get_smart_search_expander()
        if expander.is_initialized:
            stats = expander.get_expansion_stats()
            st.success("✅ Smart Search: Ready")
            st.caption(f"📚 {stats['vocabulary_size']} terms | 🔗 {stats['associations_count']} associations")
            
            if st.button("📊 Show Stats", key="show_stats"):
                st.write(f"**Vocabulary:** {stats['vocabulary_size']} terms")
                st.write(f"**Associations:** {stats['associations_count']}")
                st.write(f"**Patterns:** {stats['patterns_identified']}")
            
            if st.button("🔄 Retrain", key="retrain"):
                expander.analyze_document_content(force_rebuild=True)
                st.rerun()
        else:
            st.warning("🧠 Smart Search: Not Ready")
            if st.button("🚀 Build Smart Search", key="build_smart"):
                expander.analyze_document_content(force_rebuild=True)
                st.rerun()
        
        if st.button("🗑️ Clear Cache", key="clear_cache"):
            try:
                cache_file = Path("data/processed/query_expansions.json")
                if cache_file.exists():
                    cache_file.unlink()
                    st.success("✅ Cache cleared")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
    except Exception as e:
        st.error(f"❌ Smart Search Error: {e}")

# ================== MAIN CONTENT ==================

st.markdown('<h1 class="main-header">📄 AI Contract Assistant</h1>', unsafe_allow_html=True)
st.markdown("Ask questions about your contracts using local AI embeddings.")

# Show current status
if EMBEDDER_AVAILABLE:
    st.success("🤖 **Local embedder active** - 100% private processing")
else:
    st.error("❌ **Local embedder not available** - Please check setup")

# Document vectorization section
st.markdown("---")
st.markdown("## 🧮 Document Vectorization")

if available_docs:
    st.success(f"✅ Found {doc_count} documents in database")
    
    if vector_count == 0:
        st.warning("⚠️ Documents need vectorization")
        
        if EMBEDDER_AVAILABLE:
            if st.button("🚀 Vectorize All Documents", type="primary", use_container_width=True):
                vectorize_all_documents()
                st.rerun()
        else:
            st.error("❌ Cannot vectorize - embedder not available")
    
    else:
        st.info(f"📊 {vector_count} vectors available")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Re-vectorize All"):
                if EMBEDDER_AVAILABLE:
                    vectorize_all_documents()
                    st.rerun()
        
        with col2:
            if st.button("🧪 Test Quality"):
                test_results = search_documents("contract payment terms", top_k=3)
                if test_results:
                    st.success("✅ Search quality good")
                    for result in test_results[:2]:
                        st.write(f"- {result['filename']}: {result['text'][:100]}...")
                else:
                    st.error("❌ Search quality poor")

else:
    st.warning("⚠️ No documents found. Please run document ingestion first.")

# Question answering section
st.markdown("---")
st.markdown("## 🤖 Ask Questions")
# Question answering section with Smart Search
st.markdown("---")
st.subheader("🧠 Ask a Question About Your Contracts")

# Smart search controls
try:
    expander = get_smart_search_expander()
    smart_search_ready = expander.is_initialized
except:
    smart_search_ready = False

col1, col2 = st.columns([3, 1])
with col1:
    if smart_search_ready:
        use_smart_search = st.checkbox("🧠 Use Smart Search (Intelligent Query Expansion)", 
                                     value=True, 
                                     help="Automatically adds relevant terms based on your document content")
    else:
        st.info("🧠 **Smart Search Available:** Build the smart search system in the sidebar!")
        use_smart_search = False

with col2:
    if smart_search_ready and use_smart_search:
        expansion_level = st.selectbox("Expansion Level", 
                                     ["Light (3 terms)", "Medium (5 terms)", "Heavy (8 terms)"], 
                                     index=1)
        max_expansions = {"Light (3 terms)": 3, "Medium (5 terms)": 5, "Heavy (8 terms)": 8}[expansion_level]
    else:
        max_expansions = 0

# Smart search help
with st.expander("❓ How Smart Search Works", expanded=False):
    st.markdown("""
    ### 🧠 Smart Search System
    
    **What it does:**
    - **Learns from your documents** to understand related terms
    - **Automatically expands your queries** with relevant keywords
    - **Improves search results** by finding more relevant content
    
    **Examples:**
    - Search for "payment" → automatically adds "terms", "due", "amount"
    - Search for "termination" → adds "clause", "notice", "period"
    
    **How to use:**
    1. **Build the system:** Click "🚀 Build Smart Search" in the sidebar
    2. **Enable smart search:** Check the "🧠 Use Smart Search" box
    3. **Ask your questions:** The system will automatically improve your searches
    """)

# Input area
col1, col2 = st.columns([3, 1])

with col1:
    user_question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What are the payment terms in the contracts?"
    )

with col2:
    st.markdown("**Settings:**")
    top_k = st.number_input("Results", min_value=3, max_value=20, value=10)
    
    if OPENAI_AVAILABLE:
        model_choice = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])
        use_openai = st.checkbox("Use ChatGPT", value=True)
    else:
        model_choice = "context-only"
        use_openai = False
        st.info("💡 OpenAI not configured")

# Example questions
if doc_count > 0:
    with st.expander("💡 Example Questions", expanded=False):
        examples = [
            "What are the payment terms?",
            "Who are the contracting parties?",
            "What are the termination clauses?",
            "Compare liability provisions",
            "What are the key deliverables?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            if cols[i % 2].button(f"📝 {example}", key=f"ex_{i}"):
                user_question = example

# Search button
if st.button("🚀 Search & Analyze", type="primary", use_container_width=True):
    if not user_question.strip():
        st.warning("⚠️ Please enter a question")
    elif not status['database']:
        st.error("❌ Database not connected")
    elif vector_count == 0:
        st.error("❌ No vectors found - please vectorize documents first")
    elif not EMBEDDER_AVAILABLE:
        st.error("❌ Embedder not available")
    else:
        start_time = time.time()
        
        with st.spinner("🔍 Searching documents..."):
            # Perform search
            search_results = search_documents(
                user_question, 
                top_k, 
                use_smart_search=use_smart_search and smart_search_ready, 
                max_expansions=max_expansions
            )
            
            if not search_results:
                st.error("❌ No results found")
            else:
                st.success(f"✅ Found {len(search_results)} relevant chunks")
                
                # Try OpenAI analysis if available
                if use_openai and OPENAI_AVAILABLE:
                    try:
                        st.info("🤖 Sending chunks to ChatGPT for intelligent analysis...")
                        
                        # Prepare context from search results with better formatting
                        context_sections = []
                        for i, chunk in enumerate(search_results[:7], 1):  # Use top 7 chunks for more context
                            context_sections.append(f"""
=== DOCUMENT SECTION {i} ===
SOURCE: {chunk['filename']}
RELEVANCE: {chunk['similarity_score']:.3f}
TEXT:
{chunk['text']}
=== END SECTION {i} ===
""")
                        
                        full_context = "\n".join(context_sections)
                        
                        # Create a very aggressive and specific prompt
                        prompt = f"""You are an expert contract analyst. I am providing you with ACTUAL TEXT EXCERPTS from contract documents. Your job is to thoroughly analyze these excerpts and provide detailed, specific answers.

QUESTION: {user_question}

CONTRACT TEXT EXCERPTS:
{full_context}

CRITICAL INSTRUCTIONS:
1. These text excerpts are REAL contract content - treat them as authoritative sources
2. CAREFULLY read through each section above
3. EXTRACT ALL relevant information that relates to the question
4. QUOTE specific phrases and clauses from the contract text
5. Provide detailed explanations of what you find
6. If you see dates, amounts, terms, conditions, parties, etc. - LIST THEM ALL
7. Do NOT be conservative - if you see relevant information, report it fully
8. Organize your answer with clear headings and bullet points
9. Reference which document sections contain each piece of information

REQUIRED ANSWER FORMAT:

## SUMMARY
[Provide a direct answer to the question]

## DETAILED FINDINGS
[List all relevant information found, with quotes from the contract text]

## SPECIFIC DETAILS
[Extract specific numbers, dates, names, terms, conditions, etc.]

## SOURCE REFERENCES
[Mention which document sections contained the information]

## ADDITIONAL CONTEXT
[Any other relevant information from the contract excerpts]

YOU MUST FIND AND REPORT INFORMATION IF IT EXISTS IN THE PROVIDED TEXT. Do not be overly cautious - analyze thoroughly and report all findings."""

                        # Call OpenAI directly with more generous parameters
                        from openai import OpenAI
                        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        
                        # Show what we're sending to ChatGPT (for debugging)
                        with st.expander("🔍 Debug: Content being sent to ChatGPT", expanded=False):
                            st.write("**Question:**", user_question)
                            st.write("**Number of chunks:**", len(search_results[:7]))
                            st.write("**Total characters being sent:**", len(full_context))
                            
                            for i, chunk in enumerate(search_results[:7], 1):
                                st.write(f"**Chunk {i} from {chunk['filename']}:**")
                                st.write(f"Relevance: {chunk['similarity_score']:.3f}")
                                st.write(f"Character count: {len(chunk['text'])}")
                                
                                # Show more of the content
                                if len(chunk['text']) > 500:
                                    st.write(f"Preview: {chunk['text'][:500]}...")
                                    if st.button(f"Show full chunk {i}", key=f"debug_chunk_{i}"):
                                        st.text_area(f"Full content of chunk {i}:", chunk['text'], height=200)
                                else:
                                    st.write(f"Full content: {chunk['text']}")
                                st.markdown("---")
                        
                        response = client.chat.completions.create(
                            model=model_choice,
                            messages=[
                                {"role": "system", "content": "You are an expert contract analyst. You MUST thoroughly analyze the provided contract excerpts and extract ALL relevant information. Do not be conservative - if information exists in the text, you must find it and report it in detail. Always quote specific text from the contracts."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=2000,  # Increased for much more detailed answers
                            temperature=0.0   # Zero temperature for maximum consistency
                        )
                        
                        ai_answer = response.choices[0].message.content
                        tokens_used = response.usage.total_tokens
                        
                        # Calculate approximate cost
                        if model_choice == "gpt-4":
                            cost = (tokens_used / 1000) * 0.03
                        elif model_choice == "gpt-4o":
                            cost = (tokens_used / 1000) * 0.015
                        else:
                            cost = (tokens_used / 1000) * 0.002
                        
                        # Display AI Analysis
                        st.balloons()
                        st.markdown("### 🎯 Detailed AI Analysis:")
                        st.markdown(ai_answer)
                        
                        # Show comprehensive metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("🤖 Model", model_choice)
                        with col2:
                            st.metric("🔢 Tokens", tokens_used)
                        with col3:
                            st.metric("💰 Cost", f"${cost:.4f}")
                        with col4:
                            st.metric("📄 Chunks", len(search_results[:7]))
                        with col5:
                            st.metric("📊 Chars Sent", len(full_context))
                        
                        # Show what chunks were used with more details
                        with st.expander("📚 Documents Analyzed by AI", expanded=True):
                            st.write("**The AI analyzed content from these contract sections:**")
                            for i, chunk in enumerate(search_results[:7], 1):
                                st.write(f"**{i}.** {chunk['filename']} (Relevance: {chunk['similarity_score']:.3f}) - {len(chunk['text'])} characters")
                        
                        # Set chunks_used for source display
                        chunks_used = search_results
                        ai_analysis_success = True
                        
                    except Exception as e:
                        st.error(f"❌ OpenAI analysis failed: {e}")
                        st.info("📋 Falling back to showing raw search results...")
                        import traceback
                        st.error(f"Full error: {traceback.format_exc()}")
                        chunks_used = search_results
                        ai_analysis_success = False
                        use_openai = False
                else:
                    chunks_used = search_results
                    ai_analysis_success = False
                
                # Show search results (always show, but less prominent if AI worked)
                if ai_analysis_success:
                    st.markdown("### 📚 Source Documents Used by AI:")
                    st.info(f"The AI analysis above was based on content from {len(set([c['filename'] for c in search_results[:5]]))} documents:")
                    for i, chunk in enumerate(search_results[:5], 1):
                        st.write(f"**{i}.** {chunk['filename']} (Relevance: {chunk['similarity_score']:.3f})")
                else:
                    st.markdown("### 📋 Raw Search Results:")
                
                # Group by document for summary
                docs_found = {}
                for chunk in chunks_used:
                    doc_name = chunk.get('filename', 'Unknown')
                    if doc_name not in docs_found:
                        docs_found[doc_name] = []
                    docs_found[doc_name].append(chunk)
                
                if not ai_analysis_success:
                    st.write(f"**Found information in {len(docs_found)} documents:**")
                    for doc_name, doc_chunks in docs_found.items():
                        st.write(f"- **{doc_name}**: {len(doc_chunks)} relevant sections")
                    
                    # Show top 3 results prominently
                    st.markdown("#### 📄 Top Results:")
                    for i, result in enumerate(chunks_used[:3], 1):
                        with st.container():
                            st.markdown(f"**Result {i}** (Relevance: {result['similarity_score']:.3f})")
                            st.markdown(f"**Document:** {result['filename']}")
                            
                            content = result.get('text', 'No content available')
                            if len(content) > 300:
                                preview = content[:300] + "..."
                                st.success(f"**Content:** {preview}")
                                
                                if st.button(f"📖 Show Full Content for Result {i}", key=f"show_full_{i}"):
                                    st.info(f"**Full Content:**")
                                    st.write(content)
                            else:
                                st.success(f"**Content:** {content}")
                            
                            st.markdown("---")
                
                # Show detailed results
                with st.expander(f"🔍 All Results ({len(search_results)} chunks)", expanded=False):
                    for i, result in enumerate(search_results, 1):
                        st.markdown(f"### Result {i}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Document", result['filename'])
                        with col2:
                            st.metric("Relevance", f"{result['similarity_score']:.3f}")
                        with col3:
                            st.metric("Length", f"{result['content_length']} chars")
                        
                        # Show content directly without nested expander
                        st.markdown(f"**Content:**")
                        content_preview = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                        st.info(content_preview)
                        
                        if len(result['text']) > 500:
                            if st.button(f"📖 Show Full Content", key=f"full_content_{i}"):
                                st.write(result['text'])
                        
                        st.markdown("---")
                
                # Save to history
                processing_time = time.time() - start_time
                st.success(f"✅ Analysis completed in {processing_time:.2f}s")
                
                st.session_state['query_history'].append({
                    'timestamp': datetime.now(),
                    'question': user_question,
                    'results_count': len(search_results),
                    'processing_time': processing_time,
                    'used_openai': use_openai,
                    'used_smart_search': use_smart_search and smart_search_ready,
                    'expansion_words': search_results[0].get('expansion_words', []) if search_results else []
                })

# Query history
if st.session_state['query_history']:
    st.markdown("---")
    with st.expander(f"📚 Query History ({len(st.session_state['query_history'])} queries)", expanded=False):
        for i, query in enumerate(reversed(st.session_state['query_history'][-5:]), 1):
            st.markdown(f"**{i}.** {query['question']}")
            st.caption(f"Results: {query['results_count']} | Time: {query['processing_time']:.2f}s | {query['timestamp'].strftime('%H:%M:%S')}")
            
            if st.button(f"🔄 Re-ask", key=f"reask_{i}"):
                user_question = query['question']
                st.rerun()
            
            st.markdown("---")
        
        if st.button("🗑️ Clear History"):
            st.session_state['query_history'] = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        🔒 Local AI Processing • {EMBEDDER_TYPE} • 📊 Vector Search<br>
        Contract Assistant - {f"Analyzing {doc_count} Documents" if doc_count > 0 else "Ready for Documents"}
    </div>
    """, 
    unsafe_allow_html=True
)
