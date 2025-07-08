from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Text
from dotenv import load_dotenv
from pathlib import Path
import os
import urllib

# Load .env
dotenv_path = Path(__file__).resolve().parents[2] / "other-files" / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Setup DB engine
params = urllib.parse.quote_plus(os.getenv("ODBC_CONNECTION_STRING"))
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Metadata and table definitions
metadata = MetaData()

chunks_pdf = Table(
    "chunks_pdf", metadata,
    Column("chunk_id", String(50), primary_key=True),
    Column("source_doc", String(255)),
    Column("page_number", Integer),     # ← this line ensures the column exists
    Column("chunk_text", Text),
)


# Add other table definitions here (chunks_docx, etc.) as needed

def get_engine():
    return engine
