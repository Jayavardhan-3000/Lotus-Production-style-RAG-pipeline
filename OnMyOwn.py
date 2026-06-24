from pathlib import Path #Using it so I can access all the required format files, can't be done context managers
import hashlib #Need this to maintain idempotency, similar unique_id for similar chunks, gonna use in metadata
import logging
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from prompts import SYSTEM_PROMPT 
from ollama import chat
from config import LLM
from typing import TypedDict,NotRequired

class Chunk(TypedDict): #tells editor what it is expecting and also it gives warning, if gave wrong data type to an index or auto converts it
    chunk_id: str
    chunk_index: int
    doc_id: str
    source : str
    word_count: int
    embedding: NotRequired[list[float]]
    content: str

def chunking(directory : str, chunk_size : int = 300  , overlap_by :int = 30 ) -> list[Chunk]: #Type annotations, syntax- parameter: type = default
    if overlap_by >= chunk_size:
        raise ValueError("overlap_by must be smaller than chunk_size")
    metadata : list[Chunk] = []
    for path in sorted(Path(directory).glob("*.txt")):#Iterate through the iterator
        with path.open("r", encoding = "utf-8") as file: #Opening the file via context manager 
            text = file.read()
            words = text.split()
            step = chunk_size - overlap_by #We need to consider the overlap right! so 800 - 150 = 650
            for idx, start in enumerate(range(0,len(words),step),start = 1):#0 to 800, 650 to(+800) 1450, 1300 to 2100, we are overlapping by 150 words
                chunk_words = words[start:start+chunk_size]
                chunk = " ".join(chunk_words)
                if chunk:
                    #0 to 800, 650 to(+800) 1450, 1300 to 2100, we are overlapping by 150 words
                    hashid = hashlib.sha256(chunk.encode()).hexdigest()[:8] #refer to top imports, sha256 is an algorithm,
                    #parameter should be encoded into bytes, 
                    # #hexdigest to covert raw data which was returned into hexadecimal format, atleast we can read it
                    unq_id = f"{path.stem}_{idx}_{hashid}"
                    metadata.append({
                        "chunk_id" : unq_id,
                        "chunk_index": idx,
                        "doc_id"   : path.stem,
                        "source"   : str(path),
                        "word_count": len(chunk_words),
                        "content"  : chunk
                    })
    logging.info(f"Loaded all {len(metadata)} from {directory}")
    return metadata

class Embedder:
    def __init__(self, model_name : str):
        self.model = SentenceTransformer(model_name)
    def embed_chunks(self , metadata: list[Chunk] , batch_size: int = 32 ) -> list[Chunk]:
        if not metadata:
            raise ValueError("Given metadata is empty!")
        for i in range(0, len(metadata), batch_size):
            batch = metadata[i:i + batch_size]
            texts = [chunk["content"] for chunk in batch]
            batch_embeddings = self.model.encode(texts, normalize_embeddings= True, convert_to_numpy = False, show_progress_bar= False, batch_size=batch_size)
            for chunk, embedding in zip(batch, batch_embeddings):
                chunk["embedding"] = embedding.tolist()
        return metadata

def vector_store_exists(save_dir: str = "./vector_store") -> bool:
    store = Path(save_dir)
    return ((store / "faiss.index").exists() and (store / "chunks.json").exists())

def build_faiss_index(metadata: list[Chunk] ) -> faiss.IndexFlatIP:
    if not metadata:
        raise ValueError("Metadata couldn't be found!")
    embeddings = [chunk["embedding"] for chunk in metadata]
    vectors = np.array(embeddings, dtype = "float32")
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index_and_metadata(index : faiss.IndexFlatIP, chunks : list[Chunk], save_dir:str = './vector_store'):
    Path(save_dir).mkdir(exist_ok = True)
    faiss.write_index(index , f"{save_dir}/faiss.index")
    with open(f"{save_dir}/chunks.json","w") as f:
        json.dump(chunks,f)
    logging.info(f"Successfully saved chunks and vectors in {save_dir}/chunks and {save_dir}/faiss.index respectively")
    
    
def load_index_and_metadata(save_dir:str = "./vector_store") -> tuple:
    index = faiss.read_index(f"{save_dir}/faiss.index")
    with open(f"{save_dir}/chunks.json") as f:
        chunks = json.load(f)
    return index,chunks


class Retriever():
    def __init__(self, embedder : Embedder, index : faiss.IndexFlatIP, chunks : list[Chunk], top_k : int):
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.top_k = top_k
    def embed_query(self, query: str):
        return self.embedder.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    def retrieve(self, query:str) -> list[dict]:
            query_embedding = self.embed_query(query)
            query_vector = np.asarray([query_embedding], dtype= np.float32)
            scores, indices = self.index.search(query_vector, self.top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                chunk = self.chunks[idx]
                results.append({**chunk, "score": float(score)}) 
            return results  

def generate_answer(query: str, retrieved_chunks : list[dict]) -> str:
    context = "\n\n---\n\n".join(c["content"] for c in retrieved_chunks)
    response = chat(
        model= LLM,
        messages= [
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role": "user", "content": f"""
        Context:{context}
        Question: {query}
        Answer"""},
        ],
    )

    print(response.message.content)

