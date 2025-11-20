import logging
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import clean_text

logger = logging.getLogger(__name__)

def is_text_corrupted(text: str) -> bool:
    """Check if text is severely corrupted"""
    # Count how many lines are mostly repeated words
    lines = text.split('\n')
    bad_lines = 0
    
    for line in lines:
        words = line.split()
        if len(words) > 0:
            # If same word repeats too much, it's corrupted
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count > len(words) * 0.5:  # Same word > 50%
                bad_lines += 1
    
    # If too many bad lines, text is corrupted
    return bad_lines > len(lines) * 0.3

def clean_text_aggressively(text: str) -> str:
    """Aggressive cleaning for corrupted PDFs"""
    
    # Remove repeated words
    words = text.split()
    cleaned = []
    prev_word = None
    repeat_count = 0
    
    for word in words:
        if word.lower() == prev_word.lower() if prev_word else False:
            repeat_count += 1
            if repeat_count > 2:  # Allow max 3 repeats
                continue
        else:
            repeat_count = 0
        
        cleaned.append(word)
        prev_word = word
    
    text = ' '.join(cleaned)
    
    # Remove excessive colons, equals, dashes
    text = re.sub(r':+', ' ', text)
    text = re.sub(r'={3,}', ' ', text)
    text = re.sub(r'-{3,}', ' ', text)
    
    # Remove special repeating characters
    text = re.sub(r'[:\-=]{2,}', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF using LangChain"""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Combine text from all pages
        text = " ".join([page.page_content for page in pages])
        
        # Clean text
        text = clean_text(text)
        text = clean_text_aggressively(text)
        
        # Check if corrupted
        if is_text_corrupted(text):
            logger.warning("PDF text appears to be corrupted (scanned image?)")
            return "⚠️ PDF appears to be a scanned image. Text extraction failed."
        
        logger.info(f"Extracted {len(pages)} pages from PDF")
        logger.info(f"Text length: {len(text)} characters")
        return text
    
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split text into chunks for embedding"""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise
