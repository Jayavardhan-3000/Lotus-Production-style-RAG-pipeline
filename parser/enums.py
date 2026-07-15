from enum import Enum

class BlockType(Enum):
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    EQUATION = "equation"