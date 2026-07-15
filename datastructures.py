from dataclasses import dataclass, field
from parser.enums import BlockType
    
@dataclass
class AtomicBlock:
    type: BlockType
    content: str
    page: int
    
@dataclass
class MermaidDiagram:
    previous: str
    previous_embedding: list[float] | None
    content: str
    following: str
    following_embedding: list[float] | None
    
@dataclass
class Section:
    section_id: str
    source: str
    title: str
    heading_path: list[str]
    page: int
    contains_image: bool
    contains_table: bool
    contains_formulas: bool
    blocks: list[AtomicBlock]
    mermaid_diagrams: list[MermaidDiagram]
    
@dataclass
class Chunk:
    chunk_id: int
    content: str
    source: str
    title: str
    heading_path: list[str]
    page: int
    contains_image: bool
    contains_table: bool
    contains_formulas: bool
    section_id: str
    
@dataclass
class PackedPart:
    content: str
    tokens: int

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    
@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    needs_mermaid: bool
    needs_formula: bool
    needs_code: bool
    intent: str
    
@dataclass
class MermaidRetrievalResult:
    diagram: MermaidDiagram
    score: float
    
@dataclass
class Context:
    query: str
    chunks: list[Chunk]
    diagrams: list[MermaidDiagram]