from typing import TypedDict,NotRequired

class Chunk(TypedDict): #tells editor what it is expecting and also it gives warning, if gave wrong data type to an index or auto converts it
    chunk_id: str
    chunk_index: int
    doc_id: str
    source : str
    word_count: int
    embedding: NotRequired[list[float]]
    content: str