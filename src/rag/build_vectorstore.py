# src/rag/build_vectorstore.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_schema_texts(schema_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Load data/schema.json produced by prepare_docs.py and convert each table
    into a short, embed-ready text chunk with minimal metadata.
    """
    data = json.loads(schema_path.read_text(encoding="utf-8"))

    texts: list[str] = []
    metas: list[dict[str, Any]] = []

    for t in data:
        table = t.get("table", "UNKNOWN")
        cols = t.get("columns", [])
        fks = t.get("foreign_keys", [])

        lines: list[str] = [f"Table: {table}", "Columns:"]
        for c in cols:
            name = c.get("name")
            ctype = c.get("type")
            if name is None:
                continue
            lines.append(f"- {name}: {ctype}")

        if fks:
            lines.append("Foreign Keys:")
            for fk in fks:
                frm = fk.get("from")
                to_table = fk.get("to_table")
                to_col = fk.get("to_col")
                if frm and to_table and to_col:
                    lines.append(f"- {frm} -> {to_table}.{to_col}")

        text = "\n".join(lines)
        texts.append(text)
        metas.append({"table": table})

    return texts, metas


def build_vectorstore(
    texts: list[str],
    metas: list[dict[str, Any]],
    persist_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> Path:
    """
    Create a FAISS vector store from texts using HuggingFaceEmbeddings and persist it.
    """
    if not texts:
        raise RuntimeError("No texts provided to index.")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},                 # "cpu" or "cuda"
        encode_kwargs={"normalize_embeddings": True},    # improves inner-product/cosine
    )

    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)
    persist_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(persist_dir))
    return persist_dir.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS vector store from schema.json")
    parser.add_argument(
        "--schema",
        default="data/schema.json",
        help="Path to schema.json (default: data/schema.json)",
    )
    parser.add_argument(
        "--out",
        default="data/vectorstore",
        help="Persist directory (default: data/vectorstore)",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for embeddings model (default: cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    schema_path = Path(args.schema)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    texts, metas = load_schema_texts(schema_path)
    out_dir = Path(args.out)

    saved_path = build_vectorstore(
        texts=texts,
        metas=metas,
        persist_dir=out_dir,
        model_name=args.model,
        device=args.device,
    )

    print(f"Saved FAISS vector store to: {saved_path}")
    if metas:
        tables = [m.get("table", "UNKNOWN") for m in metas]
        print(f"Indexed tables ({len(tables)}): {', '.join(tables)}")


if __name__ == "__main__":
    main()