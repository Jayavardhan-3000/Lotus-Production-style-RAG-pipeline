from dataclasses import dataclass
import re
from datastructures import QueryAnalysis
from utils import timer

class QueryAnalyzer:
    Mermaid_pattern = re.compile(
        r"\b("
        r"diagram|architecture|flowchart|workflow|graph|pipeline|"
        r"flow|mermaid|visual|illustration|draw|show"
        r")\b",
        re.IGNORECASE
    )
    Formula_pattern = re.compile(
        r"\b("
        r"formula|equation|derive|derivation|proof|math|calculate"
        r")\b",
        re.IGNORECASE
    )
    Code_pattern = re.compile(
        r"\b("
        r"code|implement|write|python|java|cpp|c\+\+|function|class"
        r")\b",
        re.IGNORECASE
    )
    @timer
    def analyze( self,query: str) -> QueryAnalysis:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        normalized = query.lower().strip()
        needs_mermaid = bool(
            self.Mermaid_pattern.search(normalized)
        )
        needs_formula = bool(
            self.Formula_pattern.search(normalized)
        )
        needs_code = bool(
            self.Code_pattern.search(normalized)
        )
        if needs_mermaid:
            intent = "diagram"
        elif needs_formula:
            intent = "formula"
        elif needs_code:
            intent = "code"
        else:
            intent = "explanation"
            
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            needs_mermaid=needs_mermaid,
            needs_formula=needs_formula,
            needs_code=needs_code,
            intent=intent
        )