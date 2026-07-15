from datastructures import Chunk
from pathlib import Path
import numpy as np
import json
import faiss
import logging
import torch
from utils import timer
from dataclasses import asdict
@timer
def vector_store_exists(save_dir: str = "./vector_store") -> bool:
    store = Path(save_dir)
    return ((store / "faiss.index").exists() and (store / "chunks.json").exists())

@timer
def build_faiss_index(embeddings : torch.Tensor) -> faiss.Index:
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Metadata couldn't be found!")
    vectors = embeddings.cpu().numpy().astype("float32")
    dim = len(vectors[0])
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(vectors)
    return index

@timer
def save_index_and_metadata(index : faiss.Index, chunks : list[Chunk], save_dir:str = './vector_store'):
    Path(save_dir).mkdir(exist_ok = True)
    faiss.write_index(index , f"{save_dir}/faiss.index")
    with open(f"{save_dir}/chunks.json","w") as f:
        json.dump(
            [asdict(chunk) for chunk in chunks],
            f,
            indent=2
        )
    logging.info(f"Successfully saved FAISS index and chunk metadata.")
    
    
def load_index_and_metadata(save_dir:str = "./vector_store") -> tuple[faiss.Index, list[Chunk]]:
    index = faiss.read_index(f"{save_dir}/faiss.index")
    with open(f"{save_dir}/chunks.json") as f:
        chunks = [
        Chunk(**item)
        for item in json.load(f)
        ]
    return index,chunks