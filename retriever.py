import faiss
from datastructures import Chunk, RetrievalResult
import numpy as np
from embedder import Embedder
from utils import timer
import logging
from rank_bm25 import BM25Okapi
class Retriever():
    def __init__(self, embedder : Embedder, index : faiss.Index, chunks : list[Chunk], top_k : int):
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.top_k = top_k
        self.corpus = [chunk.content.split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus)
    @timer
    def semantic_retrieve(self, query:str) -> list[RetrievalResult]:
        if self.top_k <= 0:
            logging.warning("top_k must be greater than 0. Using default value of 3.")            
            self.top_k = 3
        query_embedding = self.embedder.embed_query(query)
        query_vector = np.asarray([query_embedding], dtype= np.float32)
        scores, indices = self.index.search(query_vector, self.top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievalResult(chunk=chunk,score=float(score)))     
        return results  
    @timer
    def bm25_retrieve(self, query: str) -> list[RetrievalResult]:
        if self.top_k <= 0:
            logging.warning("top_k must be greater than 0. Using default value of 3.")
            self.top_k = 3
        scores = self.bm25.get_scores(query.split())
        ranked = np.argsort(scores)[::-1][:self.top_k]
        results = []
        for idx in ranked:
            results.append(RetrievalResult(chunk=self.chunks[idx],score=float(scores[idx])))
        return results

    @timer
    def retrieve(self, query: str):
        semantic_results = self.semantic_retrieve(query)
        bm25_results = self.bm25_retrieve(query)
        return semantic_results, bm25_results