from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(
    "BAAI/bge-small-en-v1.5"
)

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))