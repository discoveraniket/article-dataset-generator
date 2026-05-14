#!/usr/bin/env python3
"""
PDF to RAG Vector Database Converter
Converts a PDF to markdown-formatted chunks, embeds them, and stores in ChromaDB.
Designed for retrieval-augmented generation (RAG) agentic systems.
"""

import os
import sys
import re
import argparse
import pypdf
import chromadb
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
from openai import OpenAI


def extract_text_to_markdown(pdf_path: str, use_raw: bool = False) -> str:
    """
    Extracts text from a PDF and applies basic markdown formatting.
    Note: PDF text extraction is inherently lossy. This applies heuristic formatting.
    """
    reader = pypdf.PdfReader(pdf_path)
    markdown_lines = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
            
        lines = text.split('\n')
        for line in lines:
            stripped = line.strip()
            if not stripped:
                markdown_lines.append("")
                continue
                
            if use_raw:
                markdown_lines.append(stripped)
                continue
                
            # Heuristic Markdown Formatting
            # 1. Detect Headings: Short, ALL CAPS lines
            if len(stripped) < 60 and stripped.isupper() and len(stripped.split()) > 1:
                markdown_lines.append(f"# {stripped}")
            # 2. Detect Subheadings: Capitalized, shorter lines
            elif len(stripped) < 40 and stripped.istitle() and len(stripped.split()) <= 8:
                markdown_lines.append(f"## {stripped}")
            # 3. Detect Lists
            elif re.match(r'^[-*•]\s', stripped):
                markdown_lines.append(stripped)
            # 4. Detect Bold (if PDF has explicit bold markers, rare but possible)
            elif re.match(r'^\*\*.*\*\*$', stripped):
                markdown_lines.append(f"**{stripped[2:-2]}**")
            else:
                markdown_lines.append(stripped)
                
        # Page separator
        markdown_lines.append("---\n")
        
    return "\n".join(markdown_lines)


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks based on paragraphs/sentences.
    """
    # Split by double newlines (paragraphs) first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If a single paragraph is longer than chunk_size, split by sentences
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 > chunk_size:
                    chunks.append(current_chunk.strip())
                    # Overlap: keep last chunk_overlap characters
                    if chunk_overlap > 0:
                        overlap_text = current_chunk.rsplit(' ', chunk_overlap)[0] if ' ' in current_chunk else ""
                        current_chunk = overlap_text + " " + sent
                    else:
                        current_chunk = sent
                else:
                    current_chunk += "\n\n" + sent if current_chunk else sent
            current_chunk += "\n\n"
        elif len(current_chunk) + len(para) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            if chunk_overlap > 0:
                overlap_text = current_chunk.rsplit(' ', chunk_overlap)[0] if ' ' in current_chunk else ""
                current_chunk = overlap_text + " " + para
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks


def get_embedding_model(model_name: str, use_lm_studio: bool = False, lm_studio_url: str = "http://localhost:1234/v1") -> Optional[SentenceTransformer]:
    """Loads the embedding model. Returns None if using LM Studio."""
    if use_lm_studio:
        print(f"Using LM Studio for embeddings at {lm_studio_url}")
        return None
    print(f"Loading local embedding model: {model_name}...")
    return SentenceTransformer(model_name)


def get_embeddings_via_lm_studio(chunks: List[str], model_name: str, lm_studio_url: str = "http://localhost:1234/v1") -> List[List[float]]:
    """Generates embeddings using LM Studio's OpenAI-compatible API."""
    print(f"Connecting to LM Studio at {lm_studio_url}...")
    client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")  # LM Studio doesn't require a real API key
    embeddings = []

    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk,
            model=model_name
        )
        emb = response.data[0].embedding
        embeddings.append(emb)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(chunks)} embeddings...")

    print(f"[OK] Generated {len(embeddings)} embeddings via LM Studio.")
    return embeddings


def query_rag_via_lm_studio(collection: chromadb.Collection, query: str, model_name: str, lm_studio_url: str = "http://localhost:1234/v1", top_k: int = 5) -> List[Dict]:
    """Performs semantic search using LM Studio for the query embedding."""
    client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")
    response = client.embeddings.create(
        input=query,
        model=model_name
    )
    query_embedding = response.data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )

    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        relevant_chunks.append({
            "id": results["ids"][0][i],
            "text": doc,
            "similarity_score": 1 - results["distances"][0][i]
        })
    return relevant_chunks


def store_in_chromadb(chunks: List[str], embeddings: List[List[float]],
                      collection_name: str = "rag_docs", db_path: str = "./rag_vector_db") -> chromadb.Collection:
    """Stores documents and embeddings in a persistent ChromaDB instance."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)
    
    # Check if already populated (idempotency)
    if collection.count() > 0:
        print(f"Collection '{collection_name}' already has {collection.count()} documents. Skipping insert.")
        return collection
        
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )
    print(f"Successfully stored {len(chunks)} chunks in '{db_path}'.")
    return collection


def query_rag(collection: chromadb.Collection, query: str, model: SentenceTransformer, top_k: int = 5) -> List[Dict]:
    """Performs semantic search against the vector DB."""
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    
    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        relevant_chunks.append({
            "id": results["ids"][0][i],
            "text": doc,
            "similarity_score": 1 - results["distances"][0][i]  # Cosine distance to similarity
        })
    return relevant_chunks


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to RAG Vector Database")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file")
    parser.add_argument("--db_path", type=str, default="./rag_vector_db", help="Path to store ChromaDB")
    parser.add_argument("--model", type=str, default="text-embedding-nomic-embed-text-v1.5", help="Embedding model name (local or OpenAI)")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI API key if using OpenAI embeddings")
    parser.add_argument("--chunk_size", type=int, default=512, help="Maximum characters per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap between chunks")
    parser.add_argument("--raw", action="store_true", help="Skip markdown formatting, use raw text")
    parser.add_argument("--query", type=str, default=None, help="Run a semantic search query after processing")
    parser.add_argument("--use_lm_studio", action="store_true", help="Use LM Studio for embeddings")
    parser.add_argument("--lm_studio_url", type=str, default="http://localhost:1234/v1", help="LM Studio API URL")
    
    args = parser.parse_args()
    
    # 1. Extract
    print(f"[EXTRACT] Extracting text from {args.pdf_path}...")
    if not os.path.exists(args.pdf_path):
        print(f"[ERROR] PDF file not found at {args.pdf_path}")
        sys.exit(1)
        
    md_text = extract_text_to_markdown(args.pdf_path, use_raw=args.raw)
    print(f"[DONE] Text extraction complete. Length: {len(md_text)} chars")
    
    # 2. Chunk
    print("[CHUNK] Chunking text...")
    chunks = chunk_text(md_text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"[DONE] Created {len(chunks)} chunks.")
    
    # 3. Embed & Store
    print("[EMBED] Embedding chunks...")
    if args.use_lm_studio:
        embeddings = get_embeddings_via_lm_studio(chunks, args.model, args.lm_studio_url)
    collection = store_in_chromadb(chunks, embeddings, db_path=args.db_path)
    
    # 4. Optional Query
    if args.query:
        print(f"\n[QUERY] Querying RAG DB for: '{args.query}'")
        results = query_rag_via_lm_studio(collection, args.query, args.model, args.lm_studio_url, top_k=3)
        print(f"[RESULTS] Top {len(results)} results:")
        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {res['similarity_score']:.4f}) ---")
            print(res['text'][:300] + "..." if len(res['text']) > 300 else res['text'])
            print(f"[ID: {res['id']}]")
    else:
        if args.openai_key:
            os.environ["OPENAI_API_KEY"] = args.openai_key
            # Note: For OpenAI, you'd typically use `langchain` or `openai` package.
            # For simplicity, this script defaults to local `sentence-transformers`.
            print("[WARN] OpenAI embedding support requires additional setup. Using local model for now.")

        model = get_embedding_model(args.model)
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()

        collection = store_in_chromadb(chunks, embeddings, db_path=args.db_path)

        # 4. Optional Query
        if args.query:
            print(f"\n[QUERY] Querying RAG DB for: '{args.query}'")
            results = query_rag(collection, args.query, model, top_k=3)
            print(f"[RESULTS] Top {len(results)} results:")
            for i, res in enumerate(results, 1):
                print(f"\n--- Result {i} (Score: {res['similarity_score']:.4f}) ---")
                print(res['text'][:300] + "..." if len(res['text']) > 300 else res['text'])
                print(f"[ID: {res['id']}]")

if __name__ == "__main__":
    main()