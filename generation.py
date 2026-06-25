from prompts import SYSTEM_PROMPT 
from ollama import chat
from config import LLM
from a_timer import timer

@timer
def generate_answer(query: str, retrieved_chunks : list[dict]) -> str:
    context = "\n\n---\n\n".join(c["content"] for c in retrieved_chunks)
    response = chat(
        model= LLM,
        messages= [
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role": "user", "content": f"""
        Context:{context}
        Question: {query}
        Answer"""},
        ],
    )

    return response.message.content