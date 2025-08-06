"""Embedding service using sentence-transformers."""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info(f"Initializing embedding service with model: {model_name}")

    def _ensure_model_loaded(self):
        """Ensure the model is loaded (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")

    async def encode_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text asynchronously.
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Embedding vector(s) as list(s) of floats
        """
        self._ensure_model_loaded()
        
        loop = asyncio.get_event_loop()
        
        if isinstance(text, str):
            # Single text input
            embedding = await loop.run_in_executor(
                self.executor, 
                self.model.encode, 
                text
            )
            result = embedding.tolist()
            logger.debug(f"Generated embedding for text: {text[:50]}...")
            return result
        else:
            # Batch text input
            embeddings = await loop.run_in_executor(
                self.executor, 
                self.model.encode, 
                text
            )
            result = [emb.tolist() for emb in embeddings]
            logger.debug(f"Generated embeddings for {len(text)} texts")
            return result

    async def encode_query(self, query: str) -> List[float]:
        """Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding as list of floats
        """
        return await self.encode_text(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embeddings.
        
        Returns:
            Embedding vector dimension
        """
        self._ensure_model_loaded()
        return self.model.get_sentence_embedding_dimension()

    def __del__(self):
        """Cleanup executor when service is destroyed."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)