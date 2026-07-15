from sentence_transformers import CrossEncoder
from datastructures import RetrievalResult
from utils import timer
class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
        
    @timer
    def rerank(self,query: str,results: list[RetrievalResult], final_top_k: int |None = None ) -> list[RetrievalResult]:
        if not results:
            raise ValueError("Given retrieval results are empty!")
        pairs = [(query ,result.chunk.content) for result in results ]
        scores = self.model.predict( pairs,show_progress_bar=False)
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(score)))
        reranked.sort(
            key=lambda result: result.score,
            reverse=True)
        if final_top_k is not None:
            return reranked[:final_top_k]
        return reranked