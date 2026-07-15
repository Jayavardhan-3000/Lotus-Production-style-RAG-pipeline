from sentence_transformers import SentenceTransformer
from utils import timer
from datastructures import Chunk
import torch
import numpy as np
class Embedder:
    def __init__(self, model_name : str, token : str):
        self.model = SentenceTransformer(model_name, token)
    
    def embed_query(self, query: str)-> np.ndarray:
        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )    
    @timer
    def embed_chunks(self , chunks: list[Chunk] , batch_size: int = 32 ) -> torch.Tensor:
        if not chunks:
            raise ValueError("Given chunks are empty!")
        texts = []
        for chunk in chunks:
            if not chunk.content.strip():
                raise ValueError(
                    f"Empty chunk found in section '{chunk.section_id}' "
                    f"(page {chunk.page})"
                )
            texts.append(chunk.content)
    
        return self.model.encode(
            texts, 
            normalize_embeddings= True, 
            convert_to_tensor= True, 
            show_progress_bar= False, 
            batch_size=batch_size)
        