import tiktoken
import re

def split_text_into_chunks(text, max_tokens=500, overlap=50, model_name="gpt-4"):
    """
    Splits text into overlapping chunks based on paragraph boundaries.
    
    Parameters:
    - text (str): The full text to split
    - max_tokens (int): Max tokens per chunk (default 500)
    - overlap (int): Token overlap between chunks (default 50)
    - model_name (str): Model to use for encoding (e.g. gpt-4, gpt-3.5-turbo)
    
    Returns:
    - List of text chunks that respect paragraph boundaries
    """
    
    # Load tokenizer
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback
    
    def count_tokens(text_segment):
        """Helper function to count tokens in a text segment"""
        return len(encoding.encode(text_segment))
    
    # Split text into paragraphs (by double newlines, single newlines, or other paragraph indicators)
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text.strip())
    
    # Clean up paragraphs - remove empty ones and strip whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    i = 0
    while i < len(paragraphs):
        paragraph = paragraphs[i]
        paragraph_tokens = count_tokens(paragraph)
        
        # If a single paragraph exceeds max_tokens, split it by sentences
        if paragraph_tokens > max_tokens:
            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                
                # If adding this sentence would exceed max_tokens, finalize current chunk
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap from previous chunk
                    if overlap > 0 and chunks:
                        overlap_text = get_overlap_text(current_chunk, overlap, encoding)
                        current_chunk = overlap_text + " " + sentence
                        current_tokens = count_tokens(current_chunk)
                    else:
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_tokens += sentence_tokens
            
        else:
            # Check if adding this paragraph would exceed max_tokens
            if current_tokens + paragraph_tokens > max_tokens and current_chunk:
                # Finalize current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                if overlap > 0 and chunks:
                    overlap_text = get_overlap_text(current_chunk, overlap, encoding)
                    current_chunk = overlap_text + "\n\n" + paragraph
                    current_tokens = count_tokens(current_chunk)
                else:
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
        
        i += 1
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def get_overlap_text(text, overlap_tokens, encoding):
    """
    Get the last 'overlap_tokens' worth of text from the given text.
    This will be used as the beginning of the next chunk.
    """
    tokens = encoding.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    
    overlap_token_list = tokens[-overlap_tokens:]
    return encoding.decode(overlap_token_list)

def split_text_into_chunks_advanced(text, max_tokens=500, overlap=50, model_name="gpt-4"):
    """
    Advanced version that handles multiple paragraph separators and preserves document structure.
    
    This version:
    1. Respects paragraph boundaries (double newlines)
    2. Falls back to sentence boundaries for long paragraphs
    3. Maintains semantic coherence
    4. Preserves formatting where possible
    """
    
    # Load tokenizer
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(text_segment):
        return len(encoding.encode(text_segment))
    
    # Multiple paragraph separation patterns
    paragraph_patterns = [
        r'\n\s*\n',           # Double newlines
        r'\r\n\s*\r\n',       # Windows double newlines
        r'\n\s*\r\n',         # Mixed newlines
        r'(?<=\.)\s*\n(?=[A-Z])',  # Period followed by newline and capital letter
    ]
    
    # Try different paragraph splitting patterns
    paragraphs = None
    for pattern in paragraph_patterns:
        potential_paragraphs = re.split(pattern, text.strip())
        potential_paragraphs = [p.strip() for p in potential_paragraphs if p.strip()]
        
        # Use this split if it creates reasonable paragraph sizes
        if len(potential_paragraphs) > 1:
            avg_length = sum(len(p) for p in potential_paragraphs) / len(potential_paragraphs)
            if 50 <= avg_length <= 2000:  # Reasonable paragraph length
                paragraphs = potential_paragraphs
                break
    
    # Fallback: if no good paragraph split found, use sentence splitting
    if paragraphs is None:
        paragraphs = re.split(r'(?<=[.!?])\s+', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)
        
        # If paragraph is too long, split it further
        if paragraph_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap
                    if overlap > 0:
                        overlap_text = get_overlap_text(current_chunk, overlap, encoding)
                        current_chunk = overlap_text + " " + sentence
                        current_tokens = count_tokens(current_chunk)
                    else:
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                else:
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    current_tokens += sentence_tokens
        else:
            # Check if adding this paragraph exceeds limit
            if current_tokens + paragraph_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Add overlap
                if overlap > 0:
                    overlap_text = get_overlap_text(current_chunk, overlap, encoding)
                    current_chunk = overlap_text + "\n\n" + paragraph
                    current_tokens = count_tokens(current_chunk)
                else:
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
            else:
                separator = "\n\n" if current_chunk else ""
                current_chunk = current_chunk + separator + paragraph
                current_tokens += paragraph_tokens
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Example usage and testing
if __name__ == "__main__":
    sample_text = """
    This is the first paragraph of a document. It contains some important information about contracts and legal matters.
    
    This is the second paragraph. It discusses different aspects of the agreement and provides more details about the terms and conditions.
    
    The third paragraph contains even more information. It might include specific clauses, dates, and other relevant details that are important for understanding the contract.
    
    Finally, this is the last paragraph. It concludes the document with final thoughts and any additional considerations.
    """
    
    print("=== Basic Paragraph-Based Chunking ===")
    chunks = split_text_into_chunks(sample_text, max_tokens=100, overlap=20)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"{chunk}")
        print("-" * 50)
    
    print("\n=== Advanced Paragraph-Based Chunking ===")
    chunks_advanced = split_text_into_chunks_advanced(sample_text, max_tokens=100, overlap=20)
    for i, chunk in enumerate(chunks_advanced, 1):
        print(f"Chunk {i}:")
        print(f"{chunk}")
        print("-" * 50)

