"""Retrieval evaluation for the RAG pipeline.

For each question in eval/questions.json the gold answer lives on known PDF
pages. A retrieval is a hit at k if any of the top-k chunks comes from a gold
page. Reports hit@1, hit@k and MRR for every (embedder, chunk size) pair.

    python evaluate.py --embedders hash ollama --chunk-sizes 400 800 1600
"""
import argparse
import json
import time
from pathlib import Path

from rag import build_index, chunk_documents, get_embeddings, load_pdf_pages

PDF = "Ads cookbook .pdf"


def score(index, questions, k):
    hit1 = hitk = rr = 0.0
    start = time.perf_counter()
    for item in questions:
        pages = [d.metadata["page"] for d in index.similarity_search(item["q"], k=k)]
        gold = set(item["pages"])
        ranks = [i for i, p in enumerate(pages, start=1) if p in gold]
        if ranks:
            hitk += 1
            rr += 1.0 / ranks[0]
            hit1 += ranks[0] == 1
    n = len(questions)
    return {"hit@1": round(hit1 / n, 3), f"hit@{k}": round(hitk / n, 3), "mrr": round(rr / n, 3),
            "ms_per_query": round(1000 * (time.perf_counter() - start) / n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", default=["hash", "ollama"])
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[400, 800, 1600])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()

    questions = json.loads(Path("eval/questions.json").read_text())
    pages = load_pdf_pages([PDF])
    results = []
    for name in args.embedders:
        emb = get_embeddings(name)
        for size in args.chunk_sizes:
            chunks = chunk_documents(pages, chunk_size=size, chunk_overlap=int(size * 0.15))
            row = {"embedder": name, "chunk_size": size, "chunks": len(chunks)}
            row.update(score(build_index(chunks, emb), questions, args.k))
            results.append(row)
            print(row)
    Path(args.out).write_text(json.dumps(
        {"questions": len(questions), "pages": len(pages), "k": args.k, "results": results}, indent=2))

    cols = ["embedder", "chunk_size", "chunks", "hit@1", f"hit@{args.k}", "mrr", "ms_per_query"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in results]
    Path("eval/results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
