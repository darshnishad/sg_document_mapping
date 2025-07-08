import camelot
import tabula
import pandas as pd
import fitz  # PyMuPDF
import os
from typing import List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_tables_from_pdf(filepath: str, method: str = "auto", pages: str = "all") -> List[Dict]:
    """
    Enhanced table extraction from PDF with multiple methods.
    
    Args:
        filepath (str): Path to PDF file
        method (str): Extraction method - "camelot", "tabula", "auto", or "all"
        pages (str): Pages to process - "all" or specific pages like "1,2,3" or "1-5"
    
    Returns:
        List[Dict]: List of extracted tables with metadata
    """
    if not os.path.exists(filepath):
        logger.error(f"PDF file not found: {filepath}")
        return []
    
    logger.info(f"📊 Extracting tables from: {os.path.basename(filepath)}")
    logger.info(f"🔧 Method: {method}, Pages: {pages}")
    
    if method == "auto":
        # Try camelot first, then tabula as fallback
        tables = extract_with_camelot(filepath, pages)
        if not tables:
            logger.info("🔄 Camelot found no tables, trying Tabula...")
            tables = extract_with_tabula(filepath, pages)
        return tables
    
    elif method == "camelot":
        return extract_with_camelot(filepath, pages)
    
    elif method == "tabula":
        return extract_with_tabula(filepath, pages)
    
    elif method == "all":
        # Extract with both methods and combine
        camelot_tables = extract_with_camelot(filepath, pages)
        tabula_tables = extract_with_tabula(filepath, pages)
        
        # Combine and deduplicate
        all_tables = camelot_tables + tabula_tables
        return deduplicate_tables(all_tables)
    
    else:
        logger.error(f"Unknown extraction method: {method}")
        return []

def extract_with_camelot(filepath: str, pages: str = "all") -> List[Dict]:
    """
    Extract tables using Camelot (good for well-structured tables).
    
    Args:
        filepath (str): Path to PDF file
        pages (str): Pages to process
    
    Returns:
        List[Dict]: Extracted tables
    """
    try:
        logger.info("🐪 Using Camelot for table extraction...")
        
        # Try both flavors
        tables_data = []
        
        # First try 'lattice' for tables with clear borders
        try:
            tables_lattice = camelot.read_pdf(filepath, pages=pages, flavor='lattice')
            if tables_lattice:
                logger.info(f"📋 Camelot (lattice) found {len(tables_lattice)} tables")
                for i, table in enumerate(tables_lattice):
                    processed_table = process_camelot_table(table, i, "lattice")
                    if processed_table:
                        tables_data.append(processed_table)
        except Exception as e:
            logger.warning(f"Camelot lattice failed: {e}")
        
        # Then try 'stream' for tables without clear borders
        try:
            tables_stream = camelot.read_pdf(filepath, pages=pages, flavor='stream')
            if tables_stream:
                logger.info(f"📋 Camelot (stream) found {len(tables_stream)} tables")
                for i, table in enumerate(tables_stream):
                    processed_table = process_camelot_table(table, i + len(tables_data), "stream")
                    if processed_table:
                        tables_data.append(processed_table)
        except Exception as e:
            logger.warning(f"Camelot stream failed: {e}")
        
        logger.info(f"✅ Camelot extracted {len(tables_data)} tables total")
        return tables_data
        
    except Exception as e:
        logger.error(f"❌ Camelot extraction failed: {e}")
        return []

def extract_with_tabula(filepath: str, pages: str = "all") -> List[Dict]:
    """
    Extract tables using Tabula (good for various table formats).
    
    Args:
        filepath (str): Path to PDF file
        pages (str): Pages to process
    
    Returns:
        List[Dict]: Extracted tables
    """
    try:
        logger.info("🔧 Using Tabula for table extraction...")
        
        # Convert pages parameter for tabula
        if pages == "all":
            pages_param = "all"
        else:
            pages_param = pages
        
        # Extract tables
        tables = tabula.read_pdf(
            filepath, 
            pages=pages_param, 
            multiple_tables=True,
            pandas_options={'header': None}
        )
        
        if not tables:
            logger.info("📋 Tabula found no tables")
            return []
        
        logger.info(f"📋 Tabula found {len(tables)} tables")
        
        tables_data = []
        for i, df in enumerate(tables):
            processed_table = process_tabula_table(df, i)
            if processed_table:
                tables_data.append(processed_table)
        
        logger.info(f"✅ Tabula extracted {len(tables_data)} tables")
        return tables_data
        
    except Exception as e:
        logger.error(f"❌ Tabula extraction failed: {e}")
        return []

def process_camelot_table(table, table_index: int, flavor: str) -> Optional[Dict]:
    """
    Process a table extracted by Camelot.
    
    Args:
        table: Camelot table object
        table_index (int): Index of the table
        flavor (str): Camelot flavor used
    
    Returns:
        Optional[Dict]: Processed table data
    """
    try:
        df = table.df.fillna("")
        
        # Skip empty tables
        if df.empty or df.shape == (0, 0):
            return None
        
        # Clean and format the table
        cleaned_df = clean_table_dataframe(df)
        
        if cleaned_df.empty:
            return None
        
        # Format as text
        table_text = format_table_as_text(cleaned_df, f"Table {table_index + 1} (Camelot-{flavor})")
        
        return {
            "table_id": f"camelot_{flavor}_{table_index}",
            "page": getattr(table, 'page', 'unknown'),
            "text": table_text,
            "method": f"camelot_{flavor}",
            "rows": len(cleaned_df),
            "columns": len(cleaned_df.columns),
            "accuracy": getattr(table, 'accuracy', 0),
            "raw_data": cleaned_df.to_dict('records') if len(cleaned_df) < 100 else None  # Store raw data for small tables
        }
        
    except Exception as e:
        logger.warning(f"Error processing Camelot table {table_index}: {e}")
        return None

def process_tabula_table(df: pd.DataFrame, table_index: int) -> Optional[Dict]:
    """
    Process a table extracted by Tabula.
    
    Args:
        df (pd.DataFrame): Tabula table dataframe
        table_index (int): Index of the table
    
    Returns:
        Optional[Dict]: Processed table data
    """
    try:
        # Skip empty tables
        if df.empty:
            return None
        
        # Clean and format the table
        cleaned_df = clean_table_dataframe(df)
        
        if cleaned_df.empty:
            return None
        
        # Format as text
        table_text = format_table_as_text(cleaned_df, f"Table {table_index + 1} (Tabula)")
        
        return {
            "table_id": f"tabula_{table_index}",
            "page": "unknown",  # Tabula doesn't always provide page info
            "text": table_text,
            "method": "tabula",
            "rows": len(cleaned_df),
            "columns": len(cleaned_df.columns),
            "accuracy": None,
            "raw_data": cleaned_df.to_dict('records') if len(cleaned_df) < 100 else None
        }
        
    except Exception as e:
        logger.warning(f"Error processing Tabula table {table_index}: {e}")
        return None

def clean_table_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize table dataframe.
    
    Args:
        df (pd.DataFrame): Raw table dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Make a copy
    cleaned_df = df.copy()
    
    # Fill NaN values
    cleaned_df = cleaned_df.fillna("")
    
    # Convert all values to strings and strip whitespace
    for col in cleaned_df.columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    
    # Remove completely empty rows
    cleaned_df = cleaned_df[~(cleaned_df == "").all(axis=1)]
    
    # Remove completely empty columns
    cleaned_df = cleaned_df.loc[:, ~(cleaned_df == "").all(axis=0)]
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def format_table_as_text(df: pd.DataFrame, table_title: str) -> str:
    """
    Format dataframe as readable text.
    
    Args:
        df (pd.DataFrame): Table dataframe
        table_title (str): Title for the table
    
    Returns:
        str: Formatted table text
    """
    if df.empty:
        return f"{table_title}\n(Empty table)"
    
    text_parts = [table_title, "=" * len(table_title)]
    
    # Detect if first row contains headers
    has_headers = detect_table_headers(df)
    
    if has_headers and len(df) > 1:
        # Use first row as headers
        headers = df.iloc[0].tolist()
        data_rows = df.iloc[1:]
        
        text_parts.append("HEADERS:")
        text_parts.append(" | ".join(str(h) for h in headers))
        text_parts.append("-" * 50)
        
        if not data_rows.empty:
            text_parts.append("DATA:")
            for _, row in data_rows.iterrows():
                row_text = " | ".join(str(val) for val in row.tolist())
                text_parts.append(row_text)
    else:
        # All rows are data
        text_parts.append("DATA:")
        for _, row in df.iterrows():
            row_text = " | ".join(str(val) for val in row.tolist())
            text_parts.append(row_text)
    
    # Add table metadata
    text_parts.append("")
    text_parts.append(f"Table Info: {len(df)} rows × {len(df.columns)} columns")
    
    return "\n".join(text_parts)

def detect_table_headers(df: pd.DataFrame) -> bool:
    """
    Detect if first row contains headers.
    
    Args:
        df (pd.DataFrame): Table dataframe
    
    Returns:
        bool: True if first row appears to be headers
    """
    if df.empty or len(df) < 2:
        return False
    
    first_row = df.iloc[0]
    second_row = df.iloc[1]
    
    # Count text vs numeric values in each row
    first_text_count = sum(1 for val in first_row if not str(val).replace('.', '').replace('-', '').replace(',', '').isdigit())
    second_numeric_count = sum(1 for val in second_row if str(val).replace('.', '').replace('-', '').replace(',', '').isdigit())
    
    # If first row is mostly text and second row has numbers, likely headers
    return (first_text_count >= len(first_row) * 0.7 and 
            second_numeric_count >= len(second_row) * 0.3)

def deduplicate_tables(tables: List[Dict]) -> List[Dict]:
    """
    Remove duplicate tables from combined extraction results.
    
    Args:
        tables (List[Dict]): List of extracted tables
    
    Returns:
        List[Dict]: Deduplicated tables
    """
    if len(tables) <= 1:
        return tables
    
    unique_tables = []
    seen_content = set()
    
    for table in tables:
        # Create a hash of the table content for comparison
        content_hash = hash(table["text"][:200])  # Use first 200 chars for comparison
        
        if content_hash not in seen_content:
            unique_tables.append(table)
            seen_content.add(content_hash)
        else:
            logger.info(f"Removed duplicate table: {table['table_id']}")
    
    logger.info(f"📋 Deduplicated: {len(tables)} → {len(unique_tables)} tables")
    return unique_tables

def get_table_info(filepath: str) -> Dict:
    """
    Get information about tables in a PDF without extracting content.
    
    Args:
        filepath (str): Path to PDF file
    
    Returns:
        Dict: Table information
    """
    try:
        # Quick scan with Camelot lattice
        tables = camelot.read_pdf(filepath, pages="1-3", flavor='lattice')  # Check first 3 pages
        
        info = {
            "file_name": os.path.basename(filepath),
            "sample_table_count": len(tables),
            "likely_has_tables": len(tables) > 0,
            "recommended_method": "camelot" if len(tables) > 0 else "tabula"
        }
        
        return info
        
    except Exception as e:
        return {"error": f"Could not analyze tables: {e}"}

def extract_tables_simple(filepath: str) -> str:
    """
    Simple table extraction that returns concatenated text.
    
    Args:
        filepath (str): Path to PDF file
    
    Returns:
        str: Concatenated table text
    """
    tables = extract_tables_from_pdf(filepath, method="auto")
    
    if not tables:
        return ""
    
    table_texts = [table["text"] for table in tables if table["text"].strip()]
    return "\n\n".join(table_texts)

def test_table_extraction(filepath: str):
    """
    Test table extraction with different methods.
    
    Args:
        filepath (str): Path to test PDF
    """
    logger.info(f"🧪 Testing table extraction with: {filepath}")
    
    # Get table info
    info = get_table_info(filepath)
    logger.info(f"📊 Table Info: {info}")
    
    # Test different methods
    methods = ["camelot", "tabula", "auto"]
    
    for method in methods:
        logger.info(f"🔧 Testing {method} method...")
        tables = extract_tables_from_pdf(filepath, method=method, pages="1-3")
        logger.info(f"Found {len(tables)} tables with {method}")
        
        for table in tables[:2]:  # Show first 2 tables
            logger.info(f"  - {table['table_id']}: {table['rows']}×{table['columns']}")

if __name__ == "__main__":
    # Example usage
    test_file = "sample.pdf"  # Replace with your test file
    if os.path.exists(test_file):
        test_table_extraction(test_file)
    else:
        logger.info(f"Test file {test_file} not found. Create a sample PDF with tables to test.")

