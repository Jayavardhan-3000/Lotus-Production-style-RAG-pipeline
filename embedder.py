from sentence_transformers import SentenceTransformer

from chunk_type import Chunk

class Embedder:
    def __init__(self, model_name : str):
        self.model = SentenceTransformer(model_name)
    def embed_chunks(self , metadata: list[Chunk] , batch_size: int = 32 ) -> list[Chunk]:
        if not metadata:
            raise ValueError("Given metadata is empty!")
        all_embeddings = []
        for i in range(0, len(metadata), batch_size):
            batch = metadata[i:i + batch_size]
            texts = [chunk["content"] for chunk in batch]
            batch_embeddings = self.model.encode(texts, normalize_embeddings= True, convert_to_numpy = False, show_progress_bar= False, batch_size=batch_size)
            all_embeddings.append(batch_embeddings.tolist())
        return all_embeddings