"""Core RAG pipeline: load PDFs page by page, chunk, embed, index, retrieve."""
import hashlib
import re
from typing import Iterable, List

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashing embedder. Needs no model; used as the lexical
    baseline in evaluate.py and in tests."""

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings(name: str = "ollama", batch_size: int = 16) -> Embeddings:
    if name == "hash":
        return HashingEmbeddings()
    from langchain_ollama import OllamaEmbeddings

    class BatchedOllamaEmbeddings(OllamaEmbeddings):
        """Sends documents in small batches: one request with hundreds of
        chunks can crash the Ollama runner (connection reset)."""

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            out: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                out += super().embed_documents(texts[i:i + batch_size])
            return out

    return BatchedOllamaEmbeddings(model=EMBED_MODEL)


def load_pdf_pages(pdf_files: Iterable, source_names: Iterable[str] = None) -> List[Document]:
    """One Document per PDF page, with source and 1-based page metadata."""
    docs = []
    names = list(source_names) if source_names else None
    for i, pdf in enumerate(pdf_files):
        name = names[i] if names else getattr(pdf, "name", str(pdf))
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": name, "page": page_no}))
    return docs


def chunk_documents(pages: List[Document], chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def build_index(chunks: List[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def retrieve(index: FAISS, question: str, k: int = DEFAULT_TOP_K) -> List[Document]:
    return index.similarity_search(question, k=k)
