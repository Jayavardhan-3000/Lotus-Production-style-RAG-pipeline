from copy import deepcopy
from datastructures import Chunk, Section, AtomicBlock
from recursive_split import recursive_split

def build_content(blocks : list[AtomicBlock]) -> str:
    return "\n\n".join(block.content for block in blocks)

def create_chunk(section: Section, blocks: list[AtomicBlock], chunk_id : int) -> Chunk:
    return Chunk(
        chunk_id = chunk_id,
        content=build_content(blocks),
        source=section.source,
        title=section.title,
        heading_path=deepcopy(section.heading_path),
        page=blocks[0].page,
        contains_image=section.contains_image,
        contains_table=any(block.type.name == "TABLE" for block in blocks),
        contains_formulas=section.contains_formulas,
        section_id=section.section_id
    )

def expand_blocks(section: Section, token_counter, max_tokens: int)-> list[AtomicBlock]:
    expanded = []
    for block in section.blocks:
        if token_counter(block.content) <= max_tokens:
            expanded.append(block)
        else:
            expanded.extend(
                recursive_split(
                    block,
                    token_counter,
                    max_tokens
                )
            )
    return expanded

def pack_section(
    section: Section,
    token_counter,
    chunk_id : int,
    target_tokens: int = 600,
    max_tokens: int = 800
) -> tuple[list[Chunk],int]:
    blocks = expand_blocks(
        section,
        token_counter,
        max_tokens
    )
    chunks = []
    current_blocks = []
    current_tokens = 0
    for block in blocks:
        tokens = token_counter(block.content)
        if current_blocks and current_tokens + tokens > target_tokens:
            chunks.append(
                create_chunk(
                    section,
                    current_blocks,
                    chunk_id
                )
            )
            chunk_id += 1
            current_blocks = []
            current_tokens = 0
        current_blocks.append(block)
        current_tokens += tokens
    if current_blocks:
        chunks.append(
            create_chunk(
                section,
                current_blocks,
                chunk_id
            )
        )
        chunk_id += 1
    return chunks, chunk_id

def pack_sections(
    sections: list[Section],
    token_counter,
    target_tokens: int = 350,
    max_tokens: int = 450
) -> list[Chunk]:
    chunks = []
    chunk_id = 0
    for section in sections:
        section_chunks, chunk_id = pack_section(
            section,
            token_counter,
            chunk_id,
            target_tokens,
            max_tokens
        )
        chunks.extend(section_chunks)
    return chunks
