import torch
import logging
from src.config import USE_CUDA, CUDA_DEVICE, MIXED_PRECISION

logger = logging.getLogger(__name__)

class DeviceManager:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.use_cuda = USE_CUDA and self.cuda_available
        self.device = self._get_device()
        self.mixed_precision = MIXED_PRECISION
        self._log_device_info()
    
    def _get_device(self):
        if self.use_cuda:
            try:
                torch.cuda.set_device(CUDA_DEVICE)
                device = torch.device(f"cuda:{CUDA_DEVICE}")
                logger.info(f"Using CUDA device: {CUDA_DEVICE}")
                return device
            except RuntimeError as e:
                logger.warning(f"Could not set CUDA device: {e}. Falling back to CPU")
                return torch.device("cpu")
        else:
            logger.info("Using CPU device")
            return torch.device("cpu")
    
    def _log_device_info(self):
        logger.info(f"CUDA Available: {self.cuda_available}")
        logger.info(f"Using CUDA: {self.use_cuda}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.mixed_precision}")
        
        if self.cuda_available:
            logger.info(f"GPU Name: {torch.cuda.get_device_name(CUDA_DEVICE)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(CUDA_DEVICE).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
    
    def empty_cache(self):
        if self.use_cuda:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    
    def get_device(self):
        return self.device
    
    def get_dtype(self):
        if self.mixed_precision == "fp16":
            return torch.float16
        elif self.mixed_precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

device_manager = DeviceManager()
