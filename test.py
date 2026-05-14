text = """
Chonkie is a lightweight and powerful library for text chunking in RAG applications.
It supports various strategies like Recursive, Token, and Semantic chunking.
Recursive chunking works by splitting text based on a hierarchy of delimiters (e.g., paragraphs, sentences, words).
This ensures that chunks are split at natural boundaries as much as possible.
In this version of Chonkie, RecursiveChunker focuses on these structural splits.
"""

from chonkie import RecursiveChunker
from chonkie.refinery import OverlapRefinery

chunker = RecursiveChunker(
    chunk_size=512
)
chunks = chunker.chunk(text)

refinery = OverlapRefinery(
    context_size=50
)
ol_chunks = refinery.refine(chunks)

# Print the results
print(f"Original Chunks: {len(chunks)}")
print(f"Refined Chunks: {len(ol_chunks)}")
