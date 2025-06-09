# RAG QA System

Two small Streamlit apps showcase how Retrieval‑Augmented Generation can answer questions from your own PDF files. Upload a document, ask away and the app will ground its Gemini LLM responses in the relevant pages.

```
PDF ➜ Chunk & Embed ➜ Chroma Vector Store ➜ Gemini LLM ➜ Answer
```

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Repository Layout](#repository-layout)

## Features

✨ Upload a PDF and chat with it in a clean interface.

✨ Documents are split into chunks, embedded with sentence-transformer models and stored using Chroma.

✨ `app.py` performs **hybrid search** (BM25 + vector similarity) for high recall, while `app2.py` demonstrates the **Parent Document Retriever** strategy.

✨ Answers come from a Gemini model and always cite the retrieved context.

✨ Vector stores are persisted under `vector_stores/`, while uploaded files live in `uploaded_pdfs/`.

## Quickstart

Install the Python requirements:

```bash
pip install -r requirements.txt
```

Set your Google API key before launching either app:

```bash
export GOOGLE_API_KEY=your-key-here
```

## Usage

Run one of the Streamlit apps from the repository root:

```bash
streamlit run app.py   # hybrid search version
streamlit run app2.py  # parent document retriever version
```

On first upload the necessary directories are created automatically. Retrieved snippets are shown in expandable sections beneath each answer.

## Repository Layout

- `app.py` – Streamlit app implementing BM25 + vector search.
- `app2.py` – Alternative app using Parent Document Retriever.
- `vector_stores/` – Chroma databases generated after processing documents.
- `requirements.txt` – Python dependencies.

Feel free to experiment with both retrieval strategies to see which suits your documents best.
