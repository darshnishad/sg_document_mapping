#!/usr/bin/env python3
"""
Vector Store Reset Script
=========================
This script completely resets all vectorized data and chunks to 0.
Run this when you want to start fresh with vectorization.

Usage: python reset_vectors.py
"""

import os
import sys
import shutil
from pathlib import Path
import json

def reset_vector_files():
    """Remove all vector index files"""
    print("🔄 Resetting vector files...")
    
    # Possible vector file locations
    vector_locations = [
        "data/processed",
        "./data/processed",
        "../data/processed",
        "backend/data/processed"
    ]
    
    files_deleted = 0
    
    for location in vector_locations:
        vector_dir = Path(location)
        if vector_dir.exists():
            print(f"📁 Checking directory: {vector_dir.absolute()}")
            
            # Files to delete
            files_to_remove = [
                "faiss_index.index",
                "faiss_metadata.json",
                "query_expansions.json",  # Smart search cache
                "index_stats.json",
                "vector_stats.json"
            ]
            
            for file_name in files_to_remove:
                file_path = vector_dir / file_name
                if file_path.exists():
                    try:
                        file_path.unlink()
                        print(f"✅ Deleted: {file_path}")
                        files_deleted += 1
                    except Exception as e:
                        print(f"❌ Error deleting {file_path}: {e}")
                else:
                    print(f"⚪ Not found: {file_path}")
    
    return files_deleted

def reset_database_chunks():
    """Optional: Clear chunks from database tables"""
    print("\n🗃️ Database chunk reset options:")
    print("⚠️  WARNING: This will delete all processed document chunks from the database!")
    print("📋 Available options:")
    print("1. Keep database chunks (recommended)")
    print("2. Clear all database chunks (DESTRUCTIVE)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "2":
        confirmation = input("⚠️  Type 'DELETE ALL CHUNKS' to confirm: ").strip()
        if confirmation == "DELETE ALL CHUNKS":
            try:
                # Import database connection
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
                from sqlalchemy import create_engine, text as sql_text
                from sqlalchemy.orm import sessionmaker
                import urllib.parse
                
                # Database connection
                params = urllib.parse.quote_plus(
                    'DRIVER={ODBC Driver 17 for SQL Server};'
                    'SERVER=SZLP112;'
                    'DATABASE=Clause;'
                    'Trusted_Connection=yes;'
                )
                engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
                
                Session = sessionmaker(bind=engine)
                session = Session()
                
                # Tables to clear
                tables = ['chunks_pdf', 'chunks_docx', 'chunks_excel', 'chunks_image', 'chunks_xml', 'chunks_xer']
                
                total_deleted = 0
                for table in tables:
                    try:
                        # Count existing records
                        count_query = sql_text(f"SELECT COUNT(*) FROM {table}")
                        result = session.execute(count_query)
                        count = result.fetchone()[0]
                        
                        if count > 0:
                            # Delete all records
                            delete_query = sql_text(f"DELETE FROM {table}")
                            session.execute(delete_query)
                            total_deleted += count
                            print(f"✅ Cleared {table}: {count} records deleted")
                        else:
                            print(f"⚪ {table}: Already empty")
                            
                    except Exception as e:
                        print(f"❌ Error clearing {table}: {e}")
                
                session.commit()
                session.close()
                
                print(f"✅ Database reset complete! Total records deleted: {total_deleted}")
                return total_deleted
                
            except Exception as e:
                print(f"❌ Database reset failed: {e}")
                return 0
        else:
            print("❌ Database reset cancelled (incorrect confirmation)")
            return 0
    else:
        print("✅ Database chunks preserved")
        return 0

def reset_cache_files():
    """Remove any cache files"""
    print("\n🗂️ Clearing cache files...")
    
    cache_files = [
        "__pycache__",
        ".streamlit",
        "*.pyc",
        ".DS_Store"
    ]
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_dir = Path(root) / dir_name
                try:
                    shutil.rmtree(cache_dir)
                    print(f"✅ Removed cache: {cache_dir}")
                except Exception as e:
                    print(f"❌ Error removing {cache_dir}: {e}")

def create_fresh_directories():
    """Create fresh empty directories"""
    print("\n📁 Creating fresh directories...")
    
    directories = [
        "data/processed",
        "data/raw",
        "data/temp"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")

def main():
    """Main reset function"""
    print("=" * 60)
    print("🚨 VECTOR STORE RESET SCRIPT")
    print("=" * 60)
    print("This script will reset all vectorized data to 0.")
    print("You will need to re-vectorize your documents after running this.")
    print()
    
    # Confirmation
    confirm = input("⚠️  Do you want to proceed? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Reset cancelled")
        return
    
    print("\n🚀 Starting reset process...\n")
    
    # Step 1: Reset vector files
    files_deleted = reset_vector_files()
    
    # Step 2: Reset database chunks (optional)
    chunks_deleted = reset_database_chunks()
    
    # Step 3: Clear cache files
    reset_cache_files()
    
    # Step 4: Create fresh directories
    create_fresh_directories()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ RESET COMPLETE!")
    print("=" * 60)
    print(f"📄 Vector files deleted: {files_deleted}")
    print(f"🗃️ Database chunks deleted: {chunks_deleted}")
    print("🧹 Cache files cleared")
    print("📁 Fresh directories created")
    print()
    print("🔄 To restore your data:")
    print("1. Run your document ingestion script")
    print("2. Use the vectorization tools in the Streamlit app")
    print("3. Re-build smart search vocabulary if needed")
    print()
    print("🎉 Your vector store is now reset to 0!")

if __name__ == "__main__":
    main()
