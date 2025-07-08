import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
from typing import List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_or_ocr(pdf_path: str, use_ocr: bool = True, dpi: int = 300, 
                       min_text_length: int = 50, max_pages: Optional[int] = None) -> List[Dict]:
    """
    Enhanced PDF text extraction with OCR fallback.
    
    Args:
        pdf_path (str): Path to PDF file
        use_ocr (bool): Whether to use OCR for pages without text
        dpi (int): DPI for OCR processing
        min_text_length (int): Minimum text length to consider as valid
        max_pages (Optional[int]): Maximum pages to process (None for all)
    
    Returns:
        List[Dict]: List of page results with text and metadata
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []
    
    try:
        doc = fitz.open(pdf_path)
        results = []
        
        total_pages = len(doc)
        pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
        
        logger.info(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
        logger.info(f"📊 Total pages: {total_pages}, Processing: {pages_to_process}")
        
        text_pages = 0
        ocr_pages = 0
        
        for page_num in range(pages_to_process):
            try:
                logger.info(f"   Processing page {page_num + 1}/{pages_to_process}")
                
                page = doc.load_page(page_num)
                
                # First, try to extract text directly
                text = page.get_text().strip()
                
                page_result = {
                    "page": page_num + 1,
                    "text": "",
                    "method": "text_extraction",
                    "char_count": 0,
                    "has_images": False,
                    "error": None
                }
                
                # Check if page has images
                image_list = page.get_images()
                page_result["has_images"] = len(image_list) > 0
                
                if text and len(text) >= min_text_length:
                    # Text extraction successful
                    page_result["text"] = clean_extracted_text(text)
                    page_result["char_count"] = len(page_result["text"])
                    text_pages += 1
                    logger.info(f"   ✅ Text extracted: {len(text)} characters")
                    
                elif use_ocr:
                    # Fallback to OCR
                    logger.info(f"   🔍 No extractable text found, using OCR...")
                    ocr_result = perform_ocr_on_page(page, dpi)
                    
                    if ocr_result["success"]:
                        page_result["text"] = ocr_result["text"]
                        page_result["char_count"] = len(ocr_result["text"])
                        page_result["method"] = "ocr"
                        ocr_pages += 1
                        logger.info(f"   ✅ OCR completed: {len(ocr_result['text'])} characters")
                    else:
                        page_result["error"] = ocr_result["error"]
                        logger.warning(f"   ⚠️ OCR failed: {ocr_result['error']}")
                
                else:
                    # No OCR, skip page
                    page_result["method"] = "skipped"
                    logger.info(f"   ⏭️ Page skipped (no text, OCR disabled)")
                
                results.append(page_result)
                
            except Exception as e:
                error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                logger.error(f"   ❌ {error_msg}")
                results.append({
                    "page": page_num + 1,
                    "text": "",
                    "method": "error",
                    "char_count": 0,
                    "has_images": False,
                    "error": error_msg
                })
        
        doc.close()
        
        # Log summary
        total_chars = sum(r["char_count"] for r in results)
        logger.info(f"✅ PDF processing complete:")
        logger.info(f"   - Text extraction: {text_pages} pages")
        logger.info(f"   - OCR: {ocr_pages} pages")
        logger.info(f"   - Total characters: {total_chars}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ PDF processing failed: {str(e)}")
        return []

def perform_ocr_on_page(page, dpi: int = 300) -> Dict:
    """
    Perform OCR on a single PDF page.
    
    Args:
        page: PyMuPDF page object
        dpi (int): DPI for rendering
    
    Returns:
        Dict: OCR result with success status, text, and error info
    """
    try:
        # Render page as image
        pix = page.get_pixmap(dpi=dpi)
        image_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes))
        
        # Enhance image for better OCR
        image = enhance_image_for_ocr(image)
        
        # Perform OCR with custom configuration
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%^&*()_+-=[]{}|;:,.<>?/~` '
        
        ocr_text = pytesseract.image_to_string(image, config=custom_config)
        cleaned_text = clean_extracted_text(ocr_text)
        
        return {
            "success": True,
            "text": cleaned_text,
            "error": None,
            "image_size": image.size
        }
        
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": str(e),
            "image_size": None
        }

def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better OCR results.
    
    Args:
        image (Image.Image): Input image
    
    Returns:
        Image.Image: Enhanced image
    """
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize if too small (OCR works better on larger images)
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = max(1000 / width, 1000 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        logger.warning(f"Image enhancement failed: {e}")
        return image

def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text (str): Raw extracted text
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common OCR artifacts
    text = text.replace('|', 'I')  # Common OCR mistake
    text = text.replace('0', 'O')  # In context where O is more likely
    
    # Remove very short lines that are likely artifacts
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 2]
    
    return '\n'.join(cleaned_lines)

def extract_text_simple(pdf_path: str) -> str:
    """
    Simple text extraction without OCR (fast method).
    
    Args:
        pdf_path (str): Path to PDF file
    
    Returns:
        str: Extracted text
    """
    try:
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_content.append(f"Page {page_num + 1}:\n{text.strip()}")
        
        doc.close()
        return "\n\n".join(text_content)
        
    except Exception as e:
        logger.error(f"Simple text extraction failed: {e}")
        return ""

def extract_text_with_layout(pdf_path: str) -> List[Dict]:
    """
    Extract text while preserving layout information.
    
    Args:
        pdf_path (str): Path to PDF file
    
    Returns:
        List[Dict]: Text with layout information
    """
    try:
        doc = fitz.open(pdf_path)
        results = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get text with layout
            text_dict = page.get_text("dict")
            
            page_content = {
                "page": page_num + 1,
                "blocks": [],
                "full_text": ""
            }
            
            full_text_parts = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_text = []
                    for line in block["lines"]:
                        line_text = []
                        for span in line["spans"]:
                            line_text.append(span["text"])
                        block_text.append(" ".join(line_text))
                    
                    block_content = "\n".join(block_text).strip()
                    if block_content:
                        page_content["blocks"].append({
                            "type": "text",
                            "content": block_content,
                            "bbox": block["bbox"]
                        })
                        full_text_parts.append(block_content)
            
            page_content["full_text"] = "\n\n".join(full_text_parts)
            results.append(page_content)
        
        doc.close()
        return results
        
    except Exception as e:
        logger.error(f"Layout extraction failed: {e}")
        return []

def get_pdf_info(pdf_path: str) -> Dict:
    """
    Get information about a PDF file.
    
    Args:
        pdf_path (str): Path to PDF file
    
    Returns:
        Dict: PDF information
    """
    try:
        doc = fitz.open(pdf_path)
        
        metadata = doc.metadata
        page_count = len(doc)
        
        # Analyze first few pages
        has_text = False
        has_images = False
        
        for page_num in range(min(3, page_count)):  # Check first 3 pages
            page = doc.load_page(page_num)
            if page.get_text().strip():
                has_text = True
            if page.get_images():
                has_images = True
        
        info = {
            "file_name": os.path.basename(pdf_path),
            "file_size": os.path.getsize(pdf_path),
            "page_count": page_count,
            "has_extractable_text": has_text,
            "has_images": has_images,
            "metadata": {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", "")
            }
        }
        
        doc.close()
        return info
        
    except Exception as e:
        return {"error": f"Could not read PDF: {e}"}

def test_pdf_processing(pdf_path: str):
    """
    Test PDF processing with different methods.
    
    Args:
        pdf_path (str): Path to test PDF
    """
    logger.info(f"🧪 Testing PDF processing with: {pdf_path}")
    
    # Get PDF info
    info = get_pdf_info(pdf_path)
    logger.info(f"📊 PDF Info: {info.get('page_count', 'unknown')} pages")
    
    # Test simple extraction
    logger.info("📄 Testing simple extraction...")
    simple_text = extract_text_simple(pdf_path)
    logger.info(f"Simple extraction: {len(simple_text)} characters")
    
    # Test advanced extraction (first 2 pages only for testing)
    logger.info("🔍 Testing advanced extraction...")
    advanced_results = extract_text_or_ocr(pdf_path, max_pages=2)
    total_chars = sum(r["char_count"] for r in advanced_results)
    logger.info(f"Advanced extraction: {total_chars} characters from {len(advanced_results)} pages")
    
    return {
        "info": info,
        "simple_length": len(simple_text),
        "advanced_results": len(advanced_results),
        "advanced_chars": total_chars
    }

# Utility function for backward compatibility
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Backward compatible function that returns simple concatenated text.
    
    Args:
        pdf_path (str): Path to PDF file
    
    Returns:
        str: Extracted text
    """
    results = extract_text_or_ocr(pdf_path)
    
    if not results:
        return ""
    
    text_parts = []
    for result in results:
        if result["text"].strip():
            text_parts.append(f"Page {result['page']}:\n{result['text']}")
    
    return "\n\n".join(text_parts)

if __name__ == "__main__":
    # Example usage
    test_file = "sample.pdf"  # Replace with your test file
    if os.path.exists(test_file):
        result = test_pdf_processing(test_file)
        logger.info("Test completed!")
    else:
        logger.info(f"Test file {test_file} not found. Create a sample PDF to test.")

