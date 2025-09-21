
import argparse
from pathlib import Path
import glob

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_docs(src_dir: Path):
    paths = []
    for ext in ("*.md", "*.txt"):
        paths.extend(glob.glob(str(src_dir / "**" / ext), recursive=True))
    docs = []
    for p in sorted(set(paths)):
        try:
            text = Path(p).read_text(encoding="utf-8")
        except Exception:
            continue
        docs.append({"path": p, "text": text})
    return docs


def chunk(docs, chunk_size=750, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for d in docs:
        for i, c in enumerate(splitter.split_text(d["text"])):
            meta = {"source": d["path"], "chunk": i}
            chunks.append((c, meta))
    return chunks


def build_faiss(chunks, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [c[0] for c in chunks]
    metas = [c[1] for c in chunks]
    vs = FAISS.from_texts(texts, embeddings, metadatas=metas)
    vs.save_local(str(out_dir), index_name="index")
    print(f"Saved vector store -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Build FAISS vector store from Markdown/TXT docs.")
    ap.add_argument("--src", default="docs/schema", help="Folder containing .md/.txt documents (recursively)")
    ap.add_argument("--out", default="data/vectorstore", help="Output folder for FAISS store")
    ap.add_argument("--chunk-size", type=int, default=750, help="Chunk size for splitting")
    ap.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap for splitting")
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)

    docs = load_docs(src_dir)
    if not docs:
        raise SystemExit(f"No .md or .txt files found under {src_dir}. Create docs first.")
    print(f"Loaded {len(docs)} source file(s).")

    chunks = chunk(docs, args.chunk_size, args.chunk_overlap)
    print(f"Chunked into {len(chunks)} pieces (size={args.chunk_size}, overlap={args.chunk_overlap}).")

    build_faiss(chunks, out_dir)


if __name__ == "__main__":
    main()
