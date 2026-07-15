from dataclasses import dataclass
from .enums import BlockType
from .metadata import Metadata
@dataclass
class Block:
    type : BlockType
    content : str
    metadata: Metadata