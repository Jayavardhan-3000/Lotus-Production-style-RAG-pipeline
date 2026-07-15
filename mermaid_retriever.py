import numpy as np

from datastructures import MermaidDiagram, MermaidRetrievalResult ,RetrievalResult, QueryAnalysis
from embedder import Embedder
from utils import timer

class MermaidRetriever:
    def __init__(self ,embedder: Embedder, artifact_store :  dict[str, list[MermaidDiagram]] ):
        self.embedder = embedder
        self.artifact_store = artifact_store
        
    @staticmethod
    def cosine_similarity(a: np.ndarray ,b: np.ndarray ) -> float:
        return float(np.dot(a, b))

    @timer
    def retrieve(query: str, analysis: QueryAnalysis,  retrieval_results: list[RetrievalResult] ) -> list[MermaidRetrievalResult]:
        if not analysis.needs_mermaid:
            return []
        if not retrieval_results:
            return []
        query_embedding = self.embedder.embed_query(query)
        section_ids = {result.chunk.section_id for result in retrieval_results}
        results = []
        for section_id in section_ids:
            diagrams = self.artifact_store.get(section_id)
            if not diagrams:
                continue
            for diagram in diagrams:
                if diagram.previous_embedding is None or diagram.following_embedding is None:
                    continue
                previous_score = self.cosine_similarity( query_embedding ,np.asarray(diagram.previous_embedding, dtype=np.float32))
                following_score = self.cosine_similarity(query_embedding, np.asarray( diagram.following_embedding,dtype=np.float32) )

                results.append(
                    MermaidRetrievalResult(
                        diagram=diagram,
                        score = (0.7 * max(previous_score, following_score) + 0.3 * min(previous_score, following_score))
                    ))

        results.sort(key=lambda result: result.score, reverse=True)
        return results