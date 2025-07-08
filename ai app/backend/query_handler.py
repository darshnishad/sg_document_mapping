import os
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
import urllib
from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker

# Load .env from the specific path
load_dotenv(dotenv_path=r"C:\Users\Darsh J Nishad\Desktop\synergiz\ai app\other-files\.env")

# FIXED: Import local embedder instead of using OpenAI for embeddings
from backend.vector_store import load_index, search_index, get_index_stats
from backend.embedder import get_embedding  # This uses LOCAL sentence transformers

# Initialize OpenAI client ONLY for intelligent question answering
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_AVAILABLE = True
    print("✅ OpenAI client initialized for intelligent responses")
except Exception as e:
    OPENAI_AVAILABLE = False
    client = None
    print(f"⚠️ OpenAI not available: {e}")

def get_database_connection():
    """Get database connection using environment variables"""
    try:
        db_driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
        db_server = os.getenv("DB_SERVER", "SZLP112")
        db_database = os.getenv("DB_DATABASE", "Clause")
        db_trusted = os.getenv("DB_TRUSTED_CONNECTION", "yes")
        
        params = urllib.parse.quote_plus(
            f'DRIVER={{{db_driver}}};'
            f'SERVER={db_server};'
            f'DATABASE={db_database};'
            f'Trusted_Connection={db_trusted};'
        )
        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        
        print(f"✅ Database connection successful: {db_server}/{db_database}")
        return engine
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def get_available_documents():
    """Get list of all available documents from all tables"""
    engine = get_database_connection()
    if not engine:
        return []
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    documents = []
    tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
    
    try:
        for table in tables:
            try:
                # Check if table exists first
                check_query = sql_text(f"""
                    SELECT COUNT(*) as count 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_NAME = :table_name
                """)
                result = session.execute(check_query, {"table_name": table})
                if result.fetchone()[0] == 0:
                    continue
                
                # Get documents from table
                doc_query = sql_text(f"SELECT DISTINCT source_doc FROM {table}")
                result = session.execute(doc_query)
                
                for row in result:
                    documents.append({
                        'filename': row[0],
                        'source_table': table
                    })
                    
            except Exception as e:
                print(f"Error querying {table}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in get_available_documents: {e}")
    finally:
        session.close()
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc in documents:
        key = (doc['filename'], doc['source_table'])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    
    print(f"📄 Found {len(unique_docs)} unique documents across {len(tables)} tables")
    return unique_docs

def get_latest_uploaded_document():
    """Get the most recently uploaded document based on file modification time or upload timestamp"""
    engine = get_database_connection()
    if not engine:
        return None
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
    latest_doc = None
    latest_time = None
    
    try:
        for table in tables:
            try:
                # Check if table exists first
                check_query = sql_text(f"""
                    SELECT COUNT(*) as count 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_NAME = :table_name
                """)
                result = session.execute(check_query, {"table_name": table})
                if result.fetchone()[0] == 0:
                    continue
                
                # Get the most recently added document from this table
                query = sql_text(f"""
                    SELECT TOP 1 source_doc, MAX(chunk_id) as latest_chunk_id
                    FROM {table} 
                    GROUP BY source_doc
                    ORDER BY MAX(chunk_id) DESC
                """)
                
                result = session.execute(query)
                row = result.fetchone()
                
                if row:
                    doc_name = row[0]
                    chunk_id = row[1]
                    
                    # Extract timestamp from chunk_id or use the chunk_id itself as ordering
                    if not latest_doc or chunk_id > latest_time:
                        latest_doc = doc_name
                        latest_time = chunk_id
                        
            except Exception as e:
                print(f"Error querying {table}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in get_latest_uploaded_document: {e}")
    finally:
        session.close()
    
    if latest_doc:
        print(f"📄 Latest uploaded document: {latest_doc}")
    else:
        print("❌ No documents found")
    
    return latest_doc

def get_chunk_details(chunk_ids: List[str]) -> List[Dict]:
    """Get detailed information about chunks from database with enhanced content extraction"""
    if not chunk_ids:
        return []
    
    engine = get_database_connection()
    if not engine:
        return []
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    chunk_details = []
    tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
    
    try:
        for table in tables:
            if not chunk_ids:  # Skip if no more chunk_ids to search for
                break
                
            try:
                # Check if table exists
                check_query = sql_text(f"""
                    SELECT COUNT(*) as count 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_NAME = :table_name
                """)
                result = session.execute(check_query, {"table_name": table})
                if result.fetchone()[0] == 0:
                    continue
                
                # Build query for this table with proper parameterization
                placeholders = ','.join([f"'{chunk_id}'" for chunk_id in chunk_ids])
                
                # Enhanced query to get more metadata
                if table == 'tables_pdf':
                    text_column = 'table_text'
                else:
                    text_column = 'chunk_text'
                
                query = sql_text(f"""
                    SELECT chunk_id, source_doc, {text_column}, 
                           file_path, created_date
                    FROM {table} 
                    WHERE chunk_id IN ({placeholders})
                """)
                
                result = session.execute(query)
                found_chunks = []
                
                for row in result:
                    # Enhanced content validation
                    chunk_text = row[2]
                    if not chunk_text or chunk_text.strip() == '' or chunk_text == 'No content available':
                        print(f"⚠️ Empty or invalid content for chunk {row[0]} in {table}")
                        continue
                    
                    chunk_detail = {
                        'chunk_id': row[0],
                        'filename': row[1],
                        'text': chunk_text.strip(),  # Clean the text
                        'page': 'N/A',  # Default since not all tables have page info
                        'source_table': table,
                        'file_path': row[3] if len(row) > 3 else None,
                        'created_date': row[4] if len(row) > 4 else None,
                        'content_length': len(chunk_text.strip()) if chunk_text else 0
                    }
                    chunk_details.append(chunk_detail)
                    found_chunks.append(row[0])
                
                # Remove found chunks from search list for efficiency
                chunk_ids = [cid for cid in chunk_ids if cid not in found_chunks]
                
            except Exception as e:
                print(f"Error querying {table}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in get_chunk_details: {e}")
    finally:
        session.close()
    
    # Filter out chunks with no content
    valid_chunks = [chunk for chunk in chunk_details if chunk['content_length'] > 10]
    print(f"📝 Retrieved {len(valid_chunks)} valid chunk details (filtered from {len(chunk_details)} total)")
    
    return valid_chunks

def enhanced_answer_query(user_question: str, top_k: int = 10, model: str = "gpt-4o", use_openai: bool = True, max_retries: int = 3) -> Dict:
    """
    ENHANCED hybrid approach: Local embeddings for search + OpenAI for intelligent answers
    
    Args:
        user_question: The user's question
        top_k: Number of chunks to retrieve and analyze
        model: OpenAI model to use for answer generation
        use_openai: Whether to use OpenAI for intelligent responses (False = context only)
        max_retries: Number of retry attempts for OpenAI API
    """
    
    # Validate inputs
    if not user_question or not user_question.strip():
        return {
            "answer": "Please provide a valid question.",
            "chunks_used": [],
            "method": "error",
            "error": "Empty question"
        }
    
    print(f"🤔 Processing question: {user_question[:100]}...")
    print(f"🎯 Approach: Local embeddings + {'OpenAI intelligence' if use_openai and OPENAI_AVAILABLE else 'Context-only'}")
    
    try:
        # Step 1: Check vector index status
        index_stats = get_index_stats()
        if index_stats['total_vectors'] == 0:
            return {
                "answer": "No documents have been vectorized yet. Please upload and vectorize some documents first.",
                "chunks_used": [],
                "method": "error",
                "error": "Empty vector index"
            }
        
        print(f"📊 Vector index contains {index_stats['total_vectors']} vectors")
        
        # Step 2: Load vector index
        load_index()
        
        # Step 3: Generate query embedding using LOCAL embedder
        print("🔍 Generating question embedding using LOCAL embedder...")
        question_embedding = get_embedding(user_question)  # Uses local sentence transformers
        
        if not question_embedding:
            return {
                "answer": "Failed to generate embedding for the question.",
                "chunks_used": [],
                "method": "error", 
                "error": "Embedding generation failed"
            }
        
        # Step 4: Search for similar chunks using local vector search
        search_top_k = max(top_k * 2, 20)  # Search for more to ensure diversity
        print(f"🔎 Searching for top {search_top_k} similar chunks using local vector search...")
        
        search_results = search_index(question_embedding, top_k=search_top_k, return_scores=True)
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents for your question.",
                "chunks_used": [],
                "method": "no_results",
                "error": "No search results"
            }
        
        print(f"✅ Found {len(search_results)} relevant chunks")
        
        # Step 5: Get detailed chunk information from database
        chunk_ids = [result['chunk_id'] for result in search_results]
        print(f"📄 Retrieving chunk details from database...")
        chunk_details = get_chunk_details(chunk_ids)
        
        if not chunk_details:
            return {
                "answer": "Found relevant chunks but couldn't retrieve their content from the database. This may indicate a content extraction issue.",
                "chunks_used": [],
                "method": "error",
                "error": "Chunk content retrieval failed"
            }
        
        # Step 6: Add similarity scores and enhance chunk information
        score_map = {result['chunk_id']: result.get('similarity_score', 0) for result in search_results}
        for chunk in chunk_details:
            chunk['similarity_score'] = score_map.get(chunk['chunk_id'], 0)
        
        # Step 7: Smart chunk selection for diversity
        chunk_details.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Diversify by document to get broader coverage
        final_chunks = []
        used_docs = set()
        
        # First pass: get best chunk from each document
        for chunk in chunk_details:
            if chunk['filename'] not in used_docs and len(final_chunks) < top_k:
                final_chunks.append(chunk)
                used_docs.add(chunk['filename'])
        
        # Second pass: fill remaining slots with best remaining chunks
        for chunk in chunk_details:
            if len(final_chunks) >= top_k:
                break
            if chunk not in final_chunks:
                final_chunks.append(chunk)
        
        final_chunks = final_chunks[:top_k]
        
        # Document distribution analysis
        doc_distribution = {}
        for chunk in final_chunks:
            doc_distribution[chunk['filename']] = doc_distribution.get(chunk['filename'], 0) + 1
        
        print(f"📊 Final selection: {len(final_chunks)} chunks from {len(doc_distribution)} documents")
        print(f"📊 Document distribution: {doc_distribution}")
        
        # Step 8: Generate intelligent answer
        if use_openai and OPENAI_AVAILABLE:
            return _generate_openai_answer(user_question, final_chunks, model, max_retries)
        else:
            return _generate_context_answer(user_question, final_chunks)
        
    except Exception as e:
        error_msg = f"Error in enhanced_answer_query: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}",
            "chunks_used": [],
            "method": "error",
            "error": error_msg
        }

def _generate_openai_answer(user_question: str, chunks: List[Dict], model: str, max_retries: int) -> Dict:
    """Generate intelligent answer using OpenAI API"""
    
    # Prepare context for OpenAI
    context_parts = []
    total_context_length = 0
    max_context_length = 15000  # Conservative limit for context
    
    for i, chunk in enumerate(chunks):
        doc_name = chunk['filename']
        content = chunk['text']
        similarity = chunk.get('similarity_score', 0)
        table = chunk.get('source_table', 'unknown')
        
        chunk_context = f"""
Document: {doc_name}
Source: {table}
Relevance Score: {similarity:.3f}
Content: {content}
"""
        
        # Check context length limits
        if total_context_length + len(chunk_context) > max_context_length:
            print(f"⚠️ Context limit reached, using top {i} chunks")
            break
            
        context_parts.append(chunk_context)
        total_context_length += len(chunk_context)
    
    context = "\n" + "="*50 + "\n".join(context_parts)
    
    # Enhanced prompt for better contract analysis
    system_prompt = """You are an expert contract analysis AI assistant. Your role is to:

1. Analyze contract documents thoroughly and provide accurate, detailed answers
2. Compare multiple contracts when relevant information spans across documents  
3. Extract key information like terms, clauses, parties, dates, obligations, etc.
4. Provide structured responses with clear sections and bullet points when appropriate
5. Be specific and factual - quote relevant sections when needed
6. Highlight important differences when comparing contracts
7. Note missing information if the question can't be fully answered

Guidelines:
- Always base your answers on the provided document context
- If information is missing, clearly state what's not available
- Use clear headings and structure for complex answers
- Quote specific contract language when relevant
- For comparisons, create clear tables or structured comparisons
- Be concise but comprehensive"""

    user_prompt = f"""Based on the following contract document sections, please answer the user's question comprehensively.

QUESTION: {user_question}

RETRIEVED DOCUMENT SECTIONS:
{context}

Please provide a detailed answer that:
1. Directly addresses the question
2. References specific information from the documents
3. Organizes the response clearly with headings if needed
4. Notes any limitations or missing information
5. For comparisons, highlights key differences and similarities

ANSWER:"""
    
    print(f"🤖 Generating intelligent answer using {model}...")
    
    # Generate response with retry logic
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=2000
            )
            
            processing_time = time.time() - start_time
            answer = response.choices[0].message.content
            
            print(f"✅ OpenAI answer generated successfully in {processing_time:.2f}s")
            
            # Calculate cost estimate
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            cost_estimate = _estimate_cost(total_tokens, model)
            
            return {
                "answer": answer,
                "chunks_used": chunks,
                "method": "openai_hybrid",
                "model_used": model,
                "search_scope": "all_documents",
                "processing_time": processing_time,
                "total_tokens": total_tokens,
                "cost_estimate": cost_estimate,
                "context_length": total_context_length,
                "documents_analyzed": len(set([c['filename'] for c in chunks]))
            }
            
        except Exception as e:
            print(f"❌ OpenAI attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"🔄 Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("❌ All OpenAI attempts failed, falling back to context-only response")
                return _generate_context_answer(user_question, chunks, openai_error=str(e))

def _generate_context_answer(user_question: str, chunks: List[Dict], openai_error: str = None) -> Dict:
    """Generate context-based answer when OpenAI is not available"""
    
    if not chunks:
        return {
            "answer": "❌ No relevant information found in the documents.",
            "chunks_used": [],
            "method": "context_only",
            "error": "No chunks available"
        }
    
    # Group chunks by document
    docs_dict = {}
    for chunk in chunks:
        doc_name = chunk['filename']
        if doc_name not in docs_dict:
            docs_dict[doc_name] = []
        docs_dict[doc_name].append(chunk)
    
    # Generate structured context-based response
    response_parts = [
        f"# 📋 Analysis Results for: {user_question}",
        "",
        f"**📊 Search Summary:**",
        f"- Found information in **{len(docs_dict)} document(s)**",
        f"- Retrieved **{len(chunks)} relevant sections**",
        f"- Search method: Local vector similarity using sentence transformers",
        ""
    ]
    
    if openai_error:
        response_parts.extend([
            f"⚠️ **Note:** OpenAI API unavailable ({openai_error}), showing structured search results.",
            ""
        ])
    
    response_parts.append("---")
    response_parts.append("")
    
    # Add content organized by document
    for doc_idx, (doc_name, doc_chunks) in enumerate(docs_dict.items(), 1):
        response_parts.append(f"## 📄 Document {doc_idx}: {doc_name}")
        response_parts.append("")
        
        for chunk_idx, chunk in enumerate(doc_chunks, 1):
            similarity = chunk.get('similarity_score', 0)
            content = chunk['text']
            
            response_parts.append(f"### Relevant Section {chunk_idx} (Relevance: {similarity:.3f})")
            
            # Truncate very long content for readability
            if len(content) > 1000:
                content = content[:1000] + "... [Content truncated for display]"
            
            response_parts.append(content)
            response_parts.append("")
        
        response_parts.append("---")
        response_parts.append("")
    
    # Add helpful footer
    response_parts.extend([
        "💡 **For intelligent AI analysis:** Configure OpenAI API key to get synthesized answers, comparisons, and insights.",
        "",
        "🔧 **Current Setup:** Using 100% local embeddings for privacy + basic context display"
    ])
    
    return {
        "answer": "\n".join(response_parts),
        "chunks_used": chunks,
        "method": "context_only",
        "search_scope": "all_documents", 
        "documents_analyzed": len(docs_dict),
        "note": "OpenAI not available - showing structured search results"
    }

def _estimate_cost(total_tokens: int, model: str) -> float:
    """Estimate cost based on token usage and model"""
    # Updated OpenAI pricing (check current rates)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
    }
    
    model_pricing = pricing.get(model, pricing["gpt-4"])
    
    # Rough estimate: 70% input, 30% output
    input_tokens = total_tokens * 0.7
    output_tokens = total_tokens * 0.3
    
    cost = (input_tokens / 1000 * model_pricing["input"]) + (output_tokens / 1000 * model_pricing["output"])
    return round(cost, 6)

# Maintain backwards compatibility
def answer_query(user_question: str, top_k: int = 5, model: str = "gpt-4o", max_retries: int = 3) -> Dict:
    """Backwards compatible function - now uses hybrid approach"""
    return enhanced_answer_query(user_question, top_k, model, use_openai=True, max_retries=max_retries)

# Debug and utility functions (keeping your existing ones)
def debug_vectorization_status():
    """Debug function to compare database chunks vs vector index"""
    print("🔍 Debugging vectorization status...")
    
    engine = get_database_connection()
    if not engine:
        return {'database_chunks': 0, 'vector_chunks': 0, 'coverage_percentage': 0}
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    total_db_chunks = 0
    tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
    
    try:
        for table in tables:
            try:
                check_query = sql_text(f"""
                    SELECT COUNT(*) as count 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_NAME = :table_name
                """)
                result = session.execute(check_query, {"table_name": table})
                if result.fetchone()[0] == 0:
                    continue
                
                query = sql_text(f"SELECT COUNT(*) FROM {table}")
                result = session.execute(query)
                count = result.fetchone()[0]
                total_db_chunks += count
                print(f"   {table}: {count} chunks")
            except Exception as e:
                print(f"   {table}: Error - {e}")
    finally:
        session.close()
    
    print(f"📊 Total database chunks: {total_db_chunks}")
    
    # Get vector index stats
    try:
        vector_stats = get_index_stats()
        print(f"📊 Vector index stats: {vector_stats}")
    except Exception as e:
        print(f"❌ Vector stats error: {e}")
        vector_stats = {'total_vectors': 0}
    
    vector_count = vector_stats.get('total_vectors', 0)
    if total_db_chunks > 0 and vector_count > 0:
        percentage = (vector_count / total_db_chunks) * 100
        print(f"📈 Vectorization coverage: {percentage:.1f}% ({vector_count}/{total_db_chunks})")
        
        if percentage < 95:
            print("⚠️ Low vectorization coverage detected!")
    
    return {
        'database_chunks': total_db_chunks,
        'vector_chunks': vector_count,
        'coverage_percentage': (vector_count / total_db_chunks * 100) if total_db_chunks > 0 else 0
    }

def get_system_status():
    """Get status of all system components"""
    status = {
        "database": False,
        "vector_index": False,
        "local_embedder": False,
        "openai": False,
        "total_documents": 0,
        "total_vectors": 0,
        "errors": []
    }
    
    # Test database connection
    try:
        engine = get_database_connection()
        if engine:
            status["database"] = True
            docs = get_available_documents()
            status["total_documents"] = len(docs)
        else:
            status["errors"].append("Database connection failed")
    except Exception as e:
        status["errors"].append(f"Database error: {e}")
    
    # Test vector index
    try:
        index_stats = get_index_stats()
        status["total_vectors"] = index_stats["total_vectors"]
        if index_stats["total_vectors"] > 0:
            status["vector_index"] = True
        else:
            status["errors"].append("Vector index is empty")
    except Exception as e:
        status["errors"].append(f"Vector index error: {e}")
    
    # Test local embedder
    try:
        test_embedding = get_embedding("test")  # Uses local embedder
        if test_embedding:
            status["local_embedder"] = True
    except Exception as e:
        status["errors"].append(f"Local embedder error: {e}")
    
    # Test OpenAI
    try:
        if OPENAI_AVAILABLE and client:
            # Quick test
            test_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            status["openai"] = True
    except Exception as e:
        status["errors"].append(f"OpenAI error: {e}")
    
    return status

# Quick test when module loads
if __name__ == "__main__":
    print("🚀 Testing hybrid query handler...")
    try:
        status = get_system_status()
        print(f"📊 System Status: {status}")
        
        # Test enhanced query function
        test_question = "What are the main contract terms?"
        print(f"\n🧪 Testing with question: {test_question}")
        result = enhanced_answer_query(test_question, top_k=3, use_openai=OPENAI_AVAILABLE)
        print(f"🎯 Result method: {result.get('method', 'unknown')}")
        
    except Exception as e:
        print(f"⚠️ Error during testing: {e}")
else:
    print("📦 Hybrid query handler module loaded")
    print(f"   - Local embedder: {'✅ Available' if True else '❌ Not available'}")
    print(f"   - OpenAI: {'✅ Available' if OPENAI_AVAILABLE else '❌ Not configured'}")
    print(f"   - Architecture: Local embeddings + {'OpenAI intelligence' if OPENAI_AVAILABLE else 'Context-only'}")