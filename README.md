# Custom Q&A Chatbot (RAG)

A local PDF question-answering chatbot built with Streamlit, LangChain, FAISS and Ollama. Nothing leaves your machine: embeddings come from `nomic-embed-text` and answers from `llama3.2`, both served by Ollama.

![Chatbot Screenshot](pdf_chatbot_gpt.jpg)

## How it works

1. **Load**: each PDF is read page by page, keeping the file name and page number as metadata.
2. **Chunk**: pages are split with a recursive character splitter (800 characters, 120 overlap).
3. **Embed and index**: chunks are embedded and stored in a FAISS index.
4. **Retrieve and answer**: the top 4 chunks are passed to the LLM through a conversational retrieval chain with chat memory. The app shows the source file and page for every answer.

| File | Purpose |
|---|---|
| `rag.py` | Loading, chunking, embedding, indexing, retrieval |
| `app.py` | Streamlit chat UI |
| `evaluate.py` | Retrieval evaluation |
| `eval/questions.json` | 32 questions with the PDF page that answers each |

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
streamlit run app.py
```

## Evaluation

`evaluate.py` measures retrieval on the 124-page ADS Circuit Design Cookbook included in this repo. Each of the 32 questions has a known answer page; a retrieval counts as a hit at k if any of the top-k chunks comes from that page. It reports hit@1, hit@4 and mean reciprocal rank for each embedder and chunk size.

```bash
python evaluate.py --embedders hash ollama --chunk-sizes 400 800 1600
```

`hash` is a bag-of-words hashing embedder that needs no model. It serves as the lexical baseline to compare `nomic-embed-text` against.

### Results

Lexical baseline (run on 2026-10-03):

| embedder | chunk_size | chunks | hit@1 | hit@4 | mrr |
|---|---|---|---|---|---|
| hash | 400 | 893 | 0.625 | 0.781 | 0.695 |
| hash | 800 | 474 | 0.594 | 0.812 | 0.682 |
| hash | 1600 | 258 | 0.625 | 0.75 | 0.667 |

Run the command above with Ollama running to add the `nomic-embed-text` rows; the full table is written to `eval/results.md`.

## Author

Rafayel Mirijanyan
