import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import datetime
import uuid
import json
from langdetect import detect
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Text
from sqlalchemy import text as sql_text  # Use alias to avoid naming conflicts
from sqlalchemy.orm import sessionmaker
import camelot
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path  # Import pdf2image for PDF to image conversion
import xml.etree.ElementTree as ET
import tiktoken  # for token-based chunking

# Import your local modules with detailed error handling
LOCAL_MODULES_AVAILABLE = False
import_errors = []

# Test each module individually to identify the specific issue
try:
    from backend.embedder import (
        get_embedding, get_embeddings_batch, initialize_embedder,
        get_embedding_info, test_embedding_connection
    )
    print("✅ Embedder module imported successfully")
except ImportError as e:
    import_errors.append(f"Embedder import error: {e}")

try:
    from backend.storage import (
        upload_file_to_storage, upload_data_to_storage,
        get_file_data_from_storage, file_exists_in_storage,
        delete_file_from_storage, get_storage_public_url,
        test_storage_connection
    )
    print("✅ Storage module imported successfully")
except ImportError as e:
    import_errors.append(f"Storage import error: {e}")

try:
    # Try importing vector_store functions one by one to identify missing functions
    from backend.vector_store import save_index, load_index, get_index_stats
    print("✅ Basic vector_store functions imported")
    
    try:
        from backend.vector_store import initialize_index
        print("✅ initialize_index imported")
    except ImportError:
        print("⚠️ initialize_index not found, will create fallback")
        def initialize_index():
            """Fallback initialize_index function"""
            try:
                load_index()
                print("✅ Existing index loaded")
            except:
                print("📝 No existing index found, will create new one when needed")
    
    try:
        from backend.vector_store import add_to_index
        print("✅ add_to_index imported")
    except ImportError:
        print("⚠️ add_to_index not found, will use alternative")
        def add_to_index(embedding, metadata):
            """Fallback add_to_index function"""
            from backend.vector_store import add_multiple_to_index
            add_multiple_to_index([embedding], [metadata])
    
    try:
        from backend.vector_store import add_chunks_to_index, add_texts_to_index
        print("✅ Batch functions imported")
    except ImportError as e:
        print(f"⚠️ Some batch functions missing: {e}")
        # Define fallback functions
        def add_chunks_to_index(chunk_ids, embeddings):
            """Fallback function"""
            from backend.vector_store import add_multiple_to_index
            metadatas = [{"chunk_id": cid} for cid in chunk_ids]
            add_multiple_to_index(embeddings, metadatas)
        
        def add_texts_to_index(texts, metadatas, chunk_ids=None):
            """Fallback function"""
            try:
                from backend.embedder import get_embeddings_batch
                from backend.vector_store import add_multiple_to_index
                embeddings = get_embeddings_batch(texts)  # ✅ Fixed: Remove model parameter
                if embeddings:
                    add_multiple_to_index(embeddings, metadatas)
            except Exception as e:
                print(f"Fallback add_texts_to_index error: {e}")
                # Try adding one by one
                for text, metadata in zip(texts, metadatas):
                    try:
                        from backend.embedder import get_embedding
                        embedding = get_embedding(text)
                        if embedding:
                            add_to_index(embedding, metadata)
                    except Exception as inner_e:
                        print(f"Individual add error: {inner_e}")
    
    try:
        from backend.vector_store import search_by_text, get_similar_chunks
        print("✅ Search functions imported")
    except ImportError:
        print("⚠️ Search functions not found, will create basic versions")
        def search_by_text(query, top_k=5, return_scores=False):
            """Fallback search function"""
            try:
                from backend.vector_store import search_index
                from backend.embedder import get_embedding
                query_embedding = get_embedding(query)
                return search_index(query_embedding, top_k, return_scores)
            except Exception as e:
                print(f"Search error: {e}")
                return []
        
        def get_similar_chunks(query, top_k=5, min_similarity=0.1):
            """Fallback similar chunks function"""
            results = search_by_text(query, top_k, return_scores=True)
            return [r for r in results if r.get('similarity_score', 0) >= min_similarity]
    
    LOCAL_MODULES_AVAILABLE = True
    print("✅ Vector store module functions available (with fallbacks if needed)")
    
except ImportError as e:
    import_errors.append(f"Vector store import error: {e}")

# Show final status
if LOCAL_MODULES_AVAILABLE:
    print("✅ All local modules imported successfully")
else:
    print("❌ Some local modules failed to import:")
    for error in import_errors:
        print(f"  - {error}")
    
    # Show in Streamlit
    st.error("❌ Local modules not available")
    st.error("Import errors:")
    for error in import_errors:
        st.error(f"• {error}")
    st.info("Expected structure: backend/embedder.py, backend/storage.py, backend/vector_store.py")
    
    # Check if files exist
    backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
    if os.path.exists(backend_path):
        st.info(f"✅ Backend folder found: {backend_path}")
        files = [f for f in os.listdir(backend_path) if f.endswith('.py')]
        st.write("Python files in backend folder:", files)
    else:
        st.error(f"❌ Backend folder not found: {backend_path}")

# Initialize vector store only if modules are available
if LOCAL_MODULES_AVAILABLE:
    try:
        # Initialize embedder first
        print("🔄 Initializing embedder...")
        if not initialize_embedder():
            st.warning("⚠️ Failed to initialize embedder, but continuing...")
        else:
            print("✅ Embedder initialized")
        
        # Initialize vector index
        print("🔄 Initializing vector index...")
        try:
            initialize_index()
            print("✅ Vector index initialized")
        except Exception as init_error:
            print(f"⚠️ Vector index initialization issue: {init_error}")
            # Try alternative initialization
            try:
                load_index()
                print("✅ Existing vector index loaded instead")
            except:
                print("📝 No existing index, will create when needed")
                
    except Exception as e:
        st.warning(f"⚠️ Module initialization issues: {e}")
        st.info("Will attempt to continue with available functionality")
else:
    st.error("❌ Cannot initialize modules - please check backend folder setup")
    st.stop()

# ================== DATABASE CONNECTIONS ==================

def get_database_connection():
    """Get connection to the main Clause database"""
    import urllib
    try:
        params = urllib.parse.quote_plus(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=SZLP112;'
            'DATABASE=Clause;'
            'Trusted_Connection=yes;'
        )
        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
        metadata = MetaData()

        tables = {}

        tables['chunks_pdf'] = Table('chunks_pdf', metadata,
            Column('chunk_id', String(100), primary_key=True),
            Column('source_doc', String(255)),
            Column('file_path', String(500)),
            Column('chunk_text', Text),
            Column('created_date', String(50))
        )

        tables['tables_pdf'] = Table('tables_pdf', metadata,
            Column('table_id', String(100), primary_key=True),
            Column('source_doc', String(255)),
            Column('file_path', String(500)),
            Column('table_text', Text),
            Column('created_date', String(50))
        )

        tables['chunks_docx'] = Table('chunks_docx', metadata,
            Column('chunk_id', String(100), primary_key=True),
            Column('source_doc', String(255)),
            Column('file_path', String(500)),
            Column('chunk_text', Text),
            Column('created_date', String(50))
        )

        tables['chunks_excel'] = Table('chunks_excel', metadata,
            Column('chunk_id', String(100), primary_key=True),
            Column('source_doc', String(255)),
            Column('file_path', String(500)),
            Column('chunk_text', Text),
            Column('created_date', String(50))
        )

        tables['chunks_image'] = Table('chunks_image', metadata,
            Column('chunk_id', String(100), primary_key=True),
            Column('source_doc', String(255)),
            Column('file_path', String(500)),
            Column('chunk_text', Text),
            Column('created_date', String(50))
        )

        tables['chunks_xml'] = Table('chunks_xml', metadata,
            Column('chunk_id', String(100), primary_key=True),
            Column('source_doc', String(255)),
            Column('file_path', String(500)),
            Column('chunk_text', Text),
            Column('created_date', String(50))
        )

        tables['xer_json'] = Table("xer_json", metadata,
            Column('id', String(100), primary_key=True),
            Column('source_doc', String(255)),
            Column('file_path', String(500)),
            Column('language', String(10)),
            Column('json_text', Text),
            Column('created_date', String(50))
        )

        tables['chunks_xer'] = Table('chunks_xer', metadata,
            Column('chunk_id', String(100), primary_key=True),
            Column('source_doc', String(255)),
            Column('file_path', String(500)),
            Column('chunk_text', Text),
            Column('created_date', String(50))
        )

        metadata.create_all(engine)  # Create tables if not exists
        return engine, tables
    except Exception as db_error:
        st.error(f"Database connection error: {db_error}")
        return None, None

def get_xml_database_connection():
    """Get connection to the XML database for Primavera view"""
    import urllib
    try:
        params = urllib.parse.quote_plus(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=SZLP112;'
            'DATABASE=Xml;'
            'Trusted_Connection=yes;'
        )
        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
        return engine
    except Exception as db_error:
        st.error(f"XML Database connection error: {db_error}")
        return None

# ================== XML PRIMAVERA PROCESSING ==================

def process_xml_primavera_view(filename, vectorize=True):
    """
    Process XML by directly vectorizing data from the Primavera view in XML database
    """
    if not LOCAL_MODULES_AVAILABLE:
        st.error("❌ Local modules not available for vectorization")
        return 0
    
    xml_engine = get_xml_database_connection()
    if xml_engine is None:
        st.error("❌ Cannot connect to XML database")
        return 0
    
    try:
        st.info("🔍 Fetching data from Primavera view in XML database...")
        
        with xml_engine.connect() as connection:
            # Check if view exists
            check_view_query = sql_text("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.VIEWS 
                WHERE TABLE_NAME = 'primavera' AND TABLE_SCHEMA = 'dbo'
            """)
            result = connection.execute(check_view_query)
            view_exists = result.fetchone()[0] > 0
            
            if not view_exists:
                st.error("❌ 'primavera' view not found in XML database")
                return 0
            
            # Fetch data from the view
            data_query = sql_text("SELECT * FROM primavera")
            result = connection.execute(data_query)
            rows = result.fetchall()
            columns = result.keys()
            
            if not rows:
                st.warning("⚠️ No data found in Primavera view")
                return 0
            
            st.success(f"✅ Found {len(rows)} records in Primavera view")
            
            # Convert to list of dictionaries
            primavera_data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                primavera_data.append(row_dict)
        
        # Vectorize the data
        if vectorize and primavera_data:
            return vectorize_primavera_data(primavera_data, filename)
        else:
            st.info("📊 Data fetched but vectorization disabled")
            return len(primavera_data)
            
    except Exception as e:
        st.error(f"❌ Error processing Primavera view: {e}")
        return 0

def vectorize_primavera_data(primavera_data, filename):
    """Vectorize the Primavera data records"""
    try:
        st.info(f"🧮 Starting vectorization of {len(primavera_data)} Primavera records...")
        
        vectorization_data = []
        now = datetime.datetime.now().isoformat()
        
        # Prepare data for vectorization
        for i, record in enumerate(primavera_data):
            try:
                # Convert record to text for vectorization
                text_parts = []
                serializable_record = {}
                
                for key, value in record.items():
                    if value is not None:
                        # Convert datetime objects to strings for JSON serialization
                        if isinstance(value, datetime.datetime):
                            str_value = value.strftime('%Y-%m-%d %H:%M:%S')
                            serializable_record[key] = str_value
                        elif isinstance(value, datetime.date):
                            str_value = value.strftime('%Y-%m-%d')
                            serializable_record[key] = str_value
                        else:
                            str_value = str(value)
                            serializable_record[key] = str_value
                        
                        # Add to text if not empty
                        if str_value.strip():
                            text_parts.append(f"{key}: {str_value.strip()}")
                
                record_text = " | ".join(text_parts)
                
                if record_text and len(record_text.strip()) > 10:
                    record_id = f"primavera_{filename}_{i}_{abs(hash(record_text)) % 9999}"
                    
                    vectorization_data.append({
                        'record_id': record_id,
                        'text': record_text,
                        'metadata': {
                            'chunk_id': record_id,
                            'source_doc': filename,
                            'file_path': f"xml_primavera/{filename}",
                            'record_index': i,
                            'doc_type': 'xml_primavera',
                            'created_date': now,
                            'table_name': 'primavera_view',
                            # Store content in multiple field names for compatibility
                            'text': record_text,
                            'chunk_text': record_text,
                            'content': record_text,
                            'full_text': record_text,
                            'extracted_text': record_text,
                            'document_text': record_text,
                            'raw_text': record_text,
                            # Primavera specific metadata
                            'source_view': 'primavera',
                            'source_database': 'Xml',
                            'record_data': serializable_record,  # Use serializable version
                            'text_length': len(record_text),
                            'filename': filename,
                            'processing_method': 'primavera_view_direct'
                        }
                    })
                        
            except Exception as record_error:
                st.warning(f"⚠️ Error processing record {i}: {record_error}")
                continue
        
        if not vectorization_data:
            st.error("❌ No valid records prepared for vectorization")
            return 0
        
        st.info(f"📋 Processing {len(vectorization_data)} valid Primavera records...")
        
        # Generate embeddings
        texts = [item['text'] for item in vectorization_data]
        embeddings = get_embeddings_batch(texts, batch_size=16)
        
        if embeddings and len(embeddings) > 0:
            # Add to vector index
            successful_adds = 0
            valid_embeddings = embeddings[:min(len(embeddings), len(vectorization_data))]
            valid_metadata_list = [item['metadata'] for item in vectorization_data[:len(valid_embeddings)]]
            
            for i, (embedding, metadata) in enumerate(zip(valid_embeddings, valid_metadata_list)):
                try:
                    add_to_index(embedding, metadata)
                    successful_adds += 1
                except Exception as add_error:
                    st.warning(f"Failed to add embedding {i}: {add_error}")
            
            # Save the updated index
            save_index()
            
            if successful_adds == len(valid_embeddings):
                st.success(f"🎉 Successfully vectorized {successful_adds} Primavera records!")
            else:
                st.warning(f"⚠️ Partial success: {successful_adds}/{len(valid_embeddings)} records vectorized")
            
            return successful_adds
        else:
            st.error("❌ No embeddings were generated")
            return 0
            
    except Exception as vectorization_error:
        st.error(f"❌ Primavera vectorization error: {vectorization_error}")
        return 0

# --------- SYSTEM STATUS CHECK -----------

def get_system_status():
    """Get comprehensive system status"""
    status = {
        'database': False,
        'xml_database': False,
        'local_storage': False,
        'local_embedder': False,
        'vector_store': False,
        'total_vectors': 0,
        'errors': []
    }
    
    # Test main database
    try:
        engine, _ = get_database_connection()
        if engine:
            status['database'] = True
        else:
            status['errors'].append("Main database connection failed")
    except Exception as e:
        status['errors'].append(f"Main database error: {e}")
    
    # Test XML database
    try:
        xml_engine = get_xml_database_connection()
        if xml_engine:
            status['xml_database'] = True
        else:
            status['errors'].append("XML database connection failed")
    except Exception as e:
        status['errors'].append(f"XML database error: {e}")
    
    # Test local modules
    if LOCAL_MODULES_AVAILABLE:
        try:
            if test_storage_connection():
                status['local_storage'] = True
        except Exception as e:
            status['errors'].append(f"Local storage error: {e}")
        
        try:
            if test_embedding_connection():
                status['local_embedder'] = True
        except Exception as e:
            status['errors'].append(f"Local embedder error: {e}")
        
        try:
            stats = get_index_stats()
            status['vector_store'] = True
            status['total_vectors'] = stats.get('total_vectors', 0)
        except Exception as e:
            status['errors'].append(f"Vector store error: {e}")
    else:
        status['errors'].append("Local modules not available")
    
    return status

# --------- TOKEN-BASED CHUNKER ----------

def chunk_text_tokenwise(content, max_tokens=500, model_name="gpt-3.5-turbo"):
    """Enhanced chunker with fallback to character-based chunking"""
    try:
        enc = tiktoken.encoding_for_model(model_name)
        tokens = enc.encode(content)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i+max_tokens]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks
    except Exception as e:
        # Fallback to character-based chunking
        st.warning(f"Token-based chunking failed, using character-based: {e}")
        chunk_size = max_tokens * 4  # Rough estimate: 4 chars per token
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

# ================== SAVE CHUNKS FUNCTION ==================

def save_chunks_with_vectorization(engine, table, filename, filepath, chunks, vectorize=True):
    """Save chunks function with proper metadata storage"""
    if not LOCAL_MODULES_AVAILABLE:
        st.error("❌ Local modules not available for vectorization")
        return 0
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    now = datetime.datetime.now().isoformat()
    
    try:
        st.info(f"💾 Saving {len(chunks)} chunks to database...")
        
        valid_chunks_saved = 0
        vectorization_data = []
        
        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip() or len(chunk.strip()) < 10:
                continue
            
            clean_chunk = chunk.strip()
            
            # Generate chunk ID
            filename_short = filename.split('.')[0][:8]
            chunk_hash = abs(hash(clean_chunk)) % 999
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
            chunk_id = f"{filename_short}_{i}_{chunk_hash}_{timestamp}"
            
            if len(chunk_id) > 50:
                chunk_id = chunk_id[:50]
            
            # Prepare database data
            if table.name == 'tables_pdf':
                chunk_data = {
                    'table_id': chunk_id,
                    'source_doc': filename,
                    'file_path': filepath,
                    'table_text': clean_chunk,
                    'created_date': now
                }
            else:
                chunk_data = {
                    'chunk_id': chunk_id,
                    'source_doc': filename,
                    'file_path': filepath,
                    'chunk_text': clean_chunk,
                    'created_date': now
                }
            
            # Insert into database
            try:
                session.execute(table.insert().values(**chunk_data))
                valid_chunks_saved += 1
                
                # Prepare for vectorization
                if vectorize and clean_chunk:
                    vectorization_data.append({
                        'chunk_id': chunk_id,
                        'text': clean_chunk,
                        'metadata': {
                            'chunk_id': chunk_id,
                            'source_doc': filename,
                            'file_path': filepath,
                            'chunk_index': i,
                            'doc_type': table.name.replace('chunks_', '').replace('tables_', ''),
                            'created_date': now,
                            'table_name': table.name,
                            'text': clean_chunk,
                            'chunk_text': clean_chunk,
                            'content': clean_chunk,
                            'extracted_text': clean_chunk,
                            'text_length': len(clean_chunk),
                            'filename': filename,
                            'processing_timestamp': now
                        }
                    })
                
            except Exception as db_error:
                st.error(f"❌ Failed to save chunk {i} to database: {db_error}")
                continue
        
        session.commit()
        st.success(f"✅ {valid_chunks_saved} valid chunks saved to database")
        
        # Vectorization
        if vectorize and vectorization_data:
            st.info(f"🧮 Starting vectorization of {len(vectorization_data)} chunks...")
            
            texts = [item['text'] for item in vectorization_data]
            metadatas = [item['metadata'] for item in vectorization_data]
            
            embeddings = get_embeddings_batch(texts, batch_size=16)
            
            if embeddings and len(embeddings) > 0:
                successful_adds = 0
                for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
                    try:
                        add_to_index(embedding, metadata)
                        successful_adds += 1
                    except Exception as add_error:
                        st.warning(f"Failed to add embedding {i}: {add_error}")
                
                save_index()
                st.success(f"🎉 Successfully vectorized {successful_adds} chunks!")
        
        return valid_chunks_saved
        
    except Exception as e:
        session.rollback()
        st.error(f"❌ Database save error: {e}")
        return 0
    finally:
        session.close()

# ================== FILE EXTRACTION FUNCTIONS ==================

def extract_text_from_pdf(filepath):
    """Enhanced PDF text extraction"""
    try:
        reader = PdfReader(filepath)
        texts = []
        
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            
            if page_text and page_text.strip() and len(page_text.strip()) > 10:
                clean_text = page_text.strip()
                clean_text = ' '.join(clean_text.split())
                
                if len(clean_text) > 10:
                    texts.append((i, clean_text))
        
        st.success(f"📄 PDF extraction complete: {len(texts)} pages with valid content")
        return texts
        
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return []

def extract_tables_from_pdf(filepath):
    """Enhanced table extraction"""
    try:
        tables = camelot.read_pdf(filepath, pages='all')
        extracted = []
        
        for i, table in enumerate(tables):
            table_text = table.df.to_string()
            
            if table_text and table_text.strip() and len(table_text.strip()) > 20:
                clean_table_text = table_text.strip()
                extracted.append((table.page, clean_table_text))
        
        st.success(f"📊 Table extraction complete: {len(extracted)} tables")
        return extracted
        
    except Exception as e:
        st.warning(f"Could not extract tables from PDF: {e}")
        return []

def extract_text_from_docx(filepath):
    """Enhanced DOCX extraction"""
    try:
        doc = Document(filepath)
        full_text = []
        
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                full_text.append(para.text.strip())
        
        content = "\n".join(full_text)
        
        if content and len(content.strip()) > 10:
            clean_content = content.strip()
            st.success(f"📄 DOCX extraction complete: {len(clean_content)} characters")
            return clean_content
        else:
            st.warning("⚠️ DOCX: No substantial content extracted")
            return ""
            
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_excel(filepath):
    """Enhanced Excel extraction"""
    try:
        xls = pd.ExcelFile(filepath)
        all_text = []
        
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            
            if not df.empty:
                all_text.append(f"--- Sheet: {sheet_name} ---")
                sheet_text = df.to_csv(index=False)
                
                if sheet_text and len(sheet_text.strip()) > 20:
                    all_text.append(sheet_text.strip())
        
        content = "\n".join(all_text)
        
        if content and len(content.strip()) > 10:
            st.success(f"📊 Excel extraction complete: {len(content)} characters")
            return content
        else:
            st.warning("⚠️ Excel: No substantial content extracted")
            return ""
            
    except Exception as e:
        st.error(f"Error extracting text from Excel: {e}")
        return ""

def extract_text_from_image(filepath):
    """Enhanced image text extraction"""
    try:
        img = Image.open(filepath)
        extracted_text = pytesseract.image_to_string(img)
        
        if extracted_text and extracted_text.strip() and len(extracted_text.strip()) > 10:
            clean_text = extracted_text.strip()
            st.success(f"🖼️ Image OCR complete: {len(clean_text)} characters")
            return clean_text
        else:
            st.warning("⚠️ Image: No substantial text extracted via OCR")
            return ""
            
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

# --------- XER JSON FUNCTIONS -----------

def save_xer_json(engine, table, filename, filepath, json_text, language):
    """Save XER JSON"""
    try:
        if not json_text or not json_text.strip() or len(json_text.strip()) < 10:
            st.warning("⚠️ XER JSON: No substantial content to save")
            return
        
        Session = sessionmaker(bind=engine)
        session = Session()
        now = datetime.datetime.now().isoformat()
        
        try:
            session.execute(table.insert().values(
                id=str(uuid.uuid4()),
                source_doc=filename,
                file_path=filepath,
                language=language,
                json_text=json_text.strip(),
                created_date=now
            ))
            session.commit()
            st.success(f"✅ Saved XER JSON from {filename}")
        except Exception as e:
            st.error(f"Error saving XER JSON: {e}")
            session.rollback()
        finally:
            session.close()
            
    except Exception as e:
        st.error(f"Error in XER JSON save: {e}")

def convert_xer_to_json(file_obj):
    """Convert XER file to JSON format"""
    try:
        content = file_obj.read().decode("utf-8", errors="ignore")
        
        if not content or len(content.strip()) < 10:
            st.warning("⚠️ XER file appears to be empty or too small")
            return {}
        
        lines = content.splitlines()
        data = {}
        current_table = None
        current_columns = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('%T'):
                current_table = line[2:].strip()
                data[current_table] = []
            elif line.startswith('%F'):
                current_columns = line[2:].strip().split('\t')
            elif current_table and current_columns and line:
                values = line.split('\t')
                if len(values) == len(current_columns):
                    row = dict(zip(current_columns, values))
                    data[current_table].append(row)
        
        if data:
            st.success(f"🔄 XER conversion complete: {len(data)} tables found")
        else:
            st.warning("⚠️ No valid data tables found in XER file")
            
        return data
        
    except Exception as e:
        st.error(f"Error converting XER to JSON: {e}")
        return {}

def clean_and_serialize_json(json_data):
    """Clean and serialize JSON data"""
    try:
        if not json_data:
            return ""
        
        serialized = json.dumps(json_data, indent=2, ensure_ascii=False)
        
        if len(serialized) > 10:
            return serialized
        else:
            st.warning("⚠️ Serialized JSON is too small")
            return ""
            
    except Exception as e:
        st.error(f"Error serializing JSON: {e}")
        return ""

def detect_language_safe(content):
    """Safe language detection"""
    try:
        if not content or len(content.strip()) < 10:
            return "unknown"
        return detect(content)
    except:
        return "unknown"

# ================== STREAMLIT APP ==================

def main():
    st.title("📄 Multi-File Uploader with XML Primavera Integration")
    
    # System status check
    if not LOCAL_MODULES_AVAILABLE:
        st.error("❌ Local modules not available. Please check your backend folder setup.")
        st.stop()
    
    # Show system status
    with st.expander("🔧 System Status", expanded=False):
        status = get_system_status()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if status['database']:
                st.success("✅ Main Database")
            else:
                st.error("❌ Main Database")
        
        with col2:
            if status['xml_database']:
                st.success("✅ XML Database")
            else:
                st.error("❌ XML Database")
        
        with col3:
            if status['local_storage']:
                st.success("✅ Local Storage")
            else:
                st.error("❌ Local Storage")
        
        with col4:
            if status['local_embedder']:
                st.success("✅ Local Embedder")
            else:
                st.error("❌ Local Embedder")
        
        st.metric("Vector Count", status['total_vectors'])
        
        if status['errors']:
            st.error("**Errors:**")
            for error in status['errors']:
                st.error(f"• {error}")

    engine, all_tables = get_database_connection()
    if engine is None:
        st.stop()

    # === PROCESSING SECTION ===
    st.subheader("📤 Upload and Process Files")
    
    # Show XML processing change
    with st.expander("🆕 XML Processing Change", expanded=False):
        st.markdown("""
        ### 🔄 **XML File Processing:**
        
        **When you upload XML files, the system will:**
        - Skip chunking the uploaded XML file
        - Connect directly to the **XML database** (Server: SZLP112, Database: Xml)
        - Fetch data from the **primavera view**
        - Vectorize each record from the view directly
        
        **Benefits:**
        - ✅ Uses actual database data instead of uploaded file
        - ✅ No XML parsing errors
        - ✅ Each Primavera record becomes searchable
        """)
    
    # Processing options
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_vectorization = st.checkbox("🧮 Enable Vectorization", value=True)
    with col2:
        max_tokens = st.number_input("Max tokens per chunk", min_value=100, max_value=2000, value=500)
    with col3:
        content_validation = st.checkbox("🔍 Content Validation", value=True)

    uploaded_files = st.file_uploader("Upload files (PDF, DOCX, Excel, Image, XML, XER)", accept_multiple_files=True)

    if uploaded_files:
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        
        total_chunks_processed = 0
        files_processed = 0

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                filename = uploaded_file.name
                filepath = f"uploaded_documents/{filename}"
                os.makedirs("uploaded_documents", exist_ok=True)
                
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                ext = filename.split('.')[-1].lower()
                st.markdown(f"### 📄 Processing file {idx}/{total_files}: {filename}")

                chunks_this_file = 0

                if ext == "pdf":
                    # Extract text
                    st.info("📖 Extracting text from PDF...")
                    pages_text = extract_text_from_pdf(filepath)
                    
                    if pages_text:
                        for page_num, page_content in pages_text:
                            if page_content and len(page_content.strip()) > 10:
                                chunks = chunk_text_tokenwise(page_content, max_tokens=max_tokens)
                                
                                if content_validation:
                                    chunks = [chunk for chunk in chunks if chunk and chunk.strip() and len(chunk.strip()) > 10]
                                
                                if chunks:
                                    chunk_count = save_chunks_with_vectorization(
                                        engine, all_tables["chunks_pdf"], filename, filepath, 
                                        chunks, vectorize=enable_vectorization
                                    )
                                    chunks_this_file += chunk_count
                    
                    # Extract tables
                    st.info("📊 Extracting tables from PDF...")
                    table_results = extract_tables_from_pdf(filepath)
                    if table_results:
                        for page_num, table_content in table_results:
                            if table_content and len(table_content.strip()) > 20:
                                table_chunks = chunk_text_tokenwise(table_content, max_tokens=max_tokens)
                                
                                if content_validation:
                                    table_chunks = [chunk for chunk in table_chunks if chunk and chunk.strip() and len(chunk.strip()) > 10]
                                
                                if table_chunks:
                                    chunk_count = save_chunks_with_vectorization(
                                        engine, all_tables["tables_pdf"], filename, filepath, 
                                        table_chunks, vectorize=enable_vectorization
                                    )
                                    chunks_this_file += chunk_count

                elif ext == "docx":
                    st.info("📖 Extracting text from DOCX...")
                    doc_content = extract_text_from_docx(filepath)
                    if doc_content and len(doc_content.strip()) > 10:
                        chunks = chunk_text_tokenwise(doc_content, max_tokens=max_tokens)
                        
                        if content_validation:
                            chunks = [chunk for chunk in chunks if chunk and chunk.strip() and len(chunk.strip()) > 10]
                        
                        if chunks:
                            chunk_count = save_chunks_with_vectorization(
                                engine, all_tables["chunks_docx"], filename, filepath, 
                                chunks, vectorize=enable_vectorization
                            )
                            chunks_this_file += chunk_count

                elif ext in ("xls", "xlsx"):
                    st.info("📊 Extracting data from Excel...")
                    excel_content = extract_text_from_excel(filepath)
                    if excel_content and len(excel_content.strip()) > 10:
                        chunks = chunk_text_tokenwise(excel_content, max_tokens=max_tokens)
                        
                        if content_validation:
                            chunks = [chunk for chunk in chunks if chunk and chunk.strip() and len(chunk.strip()) > 10]
                        
                        if chunks:
                            chunk_count = save_chunks_with_vectorization(
                                engine, all_tables["chunks_excel"], filename, filepath, 
                                chunks, vectorize=enable_vectorization
                            )
                            chunks_this_file += chunk_count

                elif ext in ("png", "jpeg", "jpg", "bmp", "tiff"):
                    st.info("🖼️ Extracting text from image using OCR...")
                    image_content = extract_text_from_image(filepath)
                    if image_content and len(image_content.strip()) > 10:
                        chunks = chunk_text_tokenwise(image_content, max_tokens=max_tokens)
                        
                        if content_validation:
                            chunks = [chunk for chunk in chunks if chunk and chunk.strip() and len(chunk.strip()) > 10]
                        
                        if chunks:
                            chunk_count = save_chunks_with_vectorization(
                                engine, all_tables["chunks_image"], filename, filepath, 
                                chunks, vectorize=enable_vectorization
                            )
                            chunks_this_file += chunk_count

                elif ext == "xml":
                    # NEW XML PROCESSING - Direct Primavera view access
                    st.info("🔄 Processing XML file using Primavera view...")
                    st.info("📊 Accessing Primavera view in XML database instead of parsing uploaded file...")
                    
                    # Process XML by accessing Primavera view directly
                    chunk_count = process_xml_primavera_view(filename, vectorize=enable_vectorization)
                    chunks_this_file += chunk_count
                    
                    if chunk_count > 0:
                        st.success(f"✅ Successfully processed {chunk_count} Primavera records")
                    else:
                        st.warning("⚠️ No Primavera records were processed")

                elif ext == "xer":
                    st.info("🔄 Converting XER to JSON...")
                    json_data = convert_xer_to_json(uploaded_file)
                    if json_data:
                        json_str = clean_and_serialize_json(json_data)
                        if json_str:
                            language = detect_language_safe(json_str)
                            save_xer_json(engine, all_tables["xer_json"], filename, filepath, json_str, language)

                            # Create searchable chunks from XER data
                            st.info("📋 Creating searchable chunks from XER data...")
                            xer_chunks = chunk_text_tokenwise(json_str, max_tokens=max_tokens)
                            
                            if content_validation:
                                xer_chunks = [chunk for chunk in xer_chunks if chunk and chunk.strip() and len(chunk.strip()) > 10]
                            
                            if xer_chunks:
                                chunk_count = save_chunks_with_vectorization(
                                    engine, all_tables["chunks_xer"], filename, filepath, 
                                    xer_chunks, vectorize=enable_vectorization
                                )
                                chunks_this_file += chunk_count

                else:
                    st.warning(f"⚠️ Unsupported file type: {ext}")

                # Update counters
                if chunks_this_file > 0:
                    files_processed += 1
                    total_chunks_processed += chunks_this_file
                    st.success(f"✅ {filename}: {chunks_this_file} chunks/records processed")
                else:
                    st.warning(f"⚠️ {filename}: No valid chunks/records were created")

            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")

            progress_bar.progress(idx / total_files)

        # === PROCESSING SUMMARY ===
        st.markdown("---")
        st.markdown("## 📊 Processing Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Processed", files_processed)
        with col2:
            st.metric("Total Chunks/Records", total_chunks_processed)
        with col3:
            success_rate = (files_processed / total_files * 100) if total_files > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        if total_chunks_processed > 0:
            st.balloons()
            st.success("🎉 File processing complete!")
        else:
            st.error("❌ No chunks/records were successfully processed.")

    st.markdown("---")

    # === VECTOR SEARCH TESTING ===
    if LOCAL_MODULES_AVAILABLE:
        st.subheader("🔍 Test Vector Search")
        
        query_text = st.text_input("Enter a test query:", 
                                 placeholder="e.g., contract terms, project data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.number_input("Number of results", min_value=1, max_value=20, value=5)
        with col2:
            min_similarity = st.slider("Minimum similarity", min_value=0.0, max_value=1.0, value=0.1)
        with col3:
            show_metadata = st.checkbox("Show metadata", value=False)
        
        if st.button("🔍 Search") and query_text:
            try:
                with st.spinner("Searching..."):
                    results = get_similar_chunks(query_text, top_k=top_k, min_similarity=min_similarity)
                
                if results:
                    st.success(f"Found {len(results)} relevant results")
                    
                    for i, result in enumerate(results, 1):
                        similarity_score = result.get('similarity_score', 0)
                        content = result.get('text', result.get('chunk_text', ''))
                        source_view = result.get('source_view', '')
                        
                        # Add Primavera indicator
                        if source_view == 'primavera':
                            source_indicator = "🏗️ PRIMAVERA"
                        else:
                            source_indicator = "📄 DOCUMENT"
                        
                        with st.expander(f"Result {i} - {source_indicator} - Similarity: {similarity_score:.3f}", expanded=(i==1)):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Source:** {result.get('source_doc', 'Unknown')}")
                                st.write(f"**Type:** {result.get('doc_type', 'Unknown')}")
                            with col2:
                                if source_view == 'primavera':
                                    st.success("🏗️ Primavera Data")
                                else:
                                    st.info("📄 Document Data")
                            
                            st.progress(similarity_score)
                            
                            st.write("**Content:**")
                            if content and "No content available" not in str(content):
                                preview = str(content)[:500] + "..." if len(str(content)) > 500 else str(content)
                                st.write(preview)
                            else:
                                st.error("❌ No content available")
                            
                            if show_metadata:
                                st.write("**Metadata:**")
                                st.json(result)
                else:
                    st.warning("No results found. Try a different query or lower the similarity threshold.")
                    
            except Exception as e:
                st.error(f"Search error: {e}")

    # === DIRECT PRIMAVERA VECTORIZATION ===
    st.markdown("---")
    st.subheader("🏗️ Direct Primavera Vectorization")
    st.info("Vectorize Primavera data directly from the XML database without uploading files")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Test Primavera Connection"):
            xml_engine = get_xml_database_connection()
            if xml_engine:
                try:
                    with xml_engine.connect() as connection:
                        check_view_query = sql_text("""
                            SELECT COUNT(*) FROM INFORMATION_SCHEMA.VIEWS 
                            WHERE TABLE_NAME = 'primavera' AND TABLE_SCHEMA = 'dbo'
                        """)
                        result = connection.execute(check_view_query)
                        view_exists = result.fetchone()[0] > 0
                        
                        if view_exists:
                            count_query = sql_text("SELECT COUNT(*) FROM primavera")
                            result = connection.execute(count_query)
                            record_count = result.fetchone()[0]
                            
                            st.success(f"✅ Primavera view found with {record_count} records")
                        else:
                            st.error("❌ Primavera view not found")
                            
                except Exception as e:
                    st.error(f"❌ Error testing Primavera view: {e}")
            else:
                st.error("❌ Cannot connect to XML database")
    
    with col2:
        if st.button("🧮 Vectorize Primavera Data", type="primary"):
            if not LOCAL_MODULES_AVAILABLE:
                st.error("❌ Local modules not available")
            else:
                dummy_filename = f"primavera_direct_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
                
                with st.spinner("Vectorizing Primavera data..."):
                    vectorized_count = process_xml_primavera_view(dummy_filename, vectorize=True)
                    
                    if vectorized_count > 0:
                        st.balloons()
                        st.success(f"🎉 Successfully vectorized {vectorized_count} Primavera records!")
                    else:
                        st.error("❌ No Primavera records were vectorized")

    # === FOOTER ===
    st.markdown("---")
    st.markdown("### 🔧 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ **Features:**")
        st.info("🏗️ XML Primavera integration")
        st.info("🔧 Fixed metadata storage")
        st.info("🔍 Content validation")
    
    with col2:
        st.info("**XML Database:**")
        st.code("Server: SZLP112\nDatabase: Xml\nView: primavera")
    
    st.caption("📄 Multi-File Uploader with XML Primavera Integration")

if __name__ == "__main__":
    main()
