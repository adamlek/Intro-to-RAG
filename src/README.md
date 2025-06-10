# RAG Demo

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using Python.

## Features

- Ingests documents (PDF, DOCX, Markdown) from the `data/` folder
- Converts and chunks documents for efficient retrieval
- Indexes document chunks in a local Qdrant vector database
- Uses HuggingFace embeddings for semantic search
- Supports querying the indexed data

## Usage

1. **Install python 3.12**
2. **Install dependencies**  
   From the `src/` directory, install requirements:
   ```sh
   uv lock
3. **How to run**
    ``` sh
    uv run python3 main.py