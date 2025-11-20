import logging
import torch
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL
from src.device_manager import device_manager

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        try:
            self.device = device_manager.get_device()
            self.dtype = device_manager.get_dtype()
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
            logger.info(f"Embedding model device: {self.device}, dtype: {self.dtype}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str):
        try:
            with torch.no_grad():
                embedding = self.model.encode(
                    text,
                    convert_to_tensor=True,
                    device=self.device
                )
            return embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_batch(self, texts: list):
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    device=self.device,
                    batch_size=32,
                    show_progress_bar=True
                )
            logger.info(f"Generated {len(embeddings)} embeddings on {self.device}")
            return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error batch embedding: {str(e)}")
            raise
    
    def __del__(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

embedding_model = EmbeddingModel()
