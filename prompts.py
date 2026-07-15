from datastructures import Context
from utils import timer

class PromptBuilder:
    SYSTEM_PROMPT = """
You are an intelligent AI assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, clearly state that.

When Mermaid diagrams are provided, use them to support your explanation.

Be accurate, concise and avoid making assumptions.
""".strip()

    @timer
    def build(self, context: Context ) -> str:
        prompt = [self.SYSTEM_PROMPT, "", "### Retrieved Context"]
        for chunk in context.chunks:
            prompt.extend([
                "",
                f"Source: {chunk.source}",
                f"Title: {chunk.title}",
                f"Page: {chunk.page}",
                chunk.content
            ])
        if context.diagrams:
            prompt.extend([
                "",
                "### Mermaid Diagrams"
            ])
            for diagram in context.diagrams:
                prompt.extend([
                    "",
                    diagram.content
                ])
        prompt.extend([
            "",
            "### User Question",
            context.query
        ])
        return "\n".join(prompt)