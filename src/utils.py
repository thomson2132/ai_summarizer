import os
from pathlib import Path
import logging
import torch
import re
from src.config import USE_CUDA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        "data/uploaded_pdfs",
        "data/chromadb_storage",
        "models"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")

def validate_pdf(file):
    """Validate if uploaded file is a valid PDF"""
    if file is None:
        return False
    if file.type != "application/pdf":
        return False
    if file.size > 50 * 1024 * 1024:  # 50MB limit
        return False
    return True

def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove excessive colons and special characters
    text = re.sub(r':+', ' ', text)
    text = re.sub(r'={2,}', ' ', text)
    text = re.sub(r'-{2,}', ' ', text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Clean newlines
    text = text.replace("\n", " ").replace("\r", " ")
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily"""
    upload_dir = "data/uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"File saved: {file_path}")
    return file_path

def check_cuda_memory(device):
    """Check available CUDA memory"""
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        allocated = torch.cuda.memory_allocated(device) / 1e9
        free = reserved - allocated
        logger.info(f"GPU Memory - Total: {total:.2f}GB, Reserved: {reserved:.2f}GB, Allocated: {allocated:.2f}GB, Free: {free:.2f}GB")
        return free
    return None

def enable_tf32():
    """Enable TensorFloat-32 for faster computation (if using compatible GPU)"""
    if USE_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TensorFloat-32 enabled for faster computation")
