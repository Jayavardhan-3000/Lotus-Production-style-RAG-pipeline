from datastructures import RetrievalResult
from utils import timer

@timer
def reciprocal_rank_fusion(*rankings: list[RetrievalResult],k: int = 60) -> list[RetrievalResult]:
    fused_scores = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = RetrievalResult(
                    chunk=result.chunk,
                    score=0.0
                )
            fused_scores[chunk_id].score += 1 / (k + rank + 1)
    return sorted(
        fused_scores.values(),
        key=lambda result: result.score,
        reverse=True
    )