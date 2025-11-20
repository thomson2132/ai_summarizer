import os
from dotenv import load_dotenv

load_dotenv()

# REMOVED: PEGASUS_MODEL (not used)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./data/chromadb_storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_papers")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
USE_CUDA = os.getenv("USE_CUDA", "true").lower() == "true"
CUDA_DEVICE = int(os.getenv("CUDA_DEVICE", 0))
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "fp16")
