# src/rag/build_index.py
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_env() -> Path:
    """
    Load .env from project root with BOM-tolerant encoding.
    Returns the resolved path to the .env used.
    """
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    loaded = load_dotenv(dotenv_path, override=True, encoding="utf-8-sig")
    print(f"dotenv: {dotenv_path} exists={dotenv_path.exists()} loaded={loaded}")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY missing. Add it to .env or env vars.")
    return dotenv_path


def read_schema_doc(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Doc not found: {path}")
    text = path.read_text(encoding="utf-8")
    print(f"Loaded doc: {path} ({len(text)} chars)")
    return text


def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = [Document(page_content=chunk, metadata={"source": "schema.md", "category": "schema"}) for chunk in splitter.split_text(text)]
    print(f"Created {len(docs)} chunks")
    return docs


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    api_key = os.getenv("GOOGLE_API_KEY")
    emb = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    print(f"Using embedding model: {model_name}")
    return emb


def persist_faiss(docs: List[Document], emb: GoogleGenerativeAIEmbeddings, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vectordb = FAISS.from_documents(docs, embedding=emb)
    vectordb.save_local(str(out_dir))
    print(f"Saved FAISS index to: {out_dir}")


def smoke_test(out_dir: Path, emb: GoogleGenerativeAIEmbeddings) -> None:
    print("Running retrieval smoke test...")
    db = FAISS.load_local(str(out_dir), embeddings=emb, allow_dangerous_deserialization=True)
    q = "Top 5 countries by total sales"
    results = db.similarity_search(q, k=2)
    for i, d in enumerate(results, 1):
        snippet = d.page_content[:200].replace("\n", " ")
        print(f"({i}) score=N/A meta={d.metadata} snippet={snippet}...")


def main():
    project_root = Path(__file__).resolve().parents[2]
    load_env()

    schema_path = project_root / "data" / "docs" / "schema.md"
    text = read_schema_doc(schema_path)
    docs = chunk_text(text)

    emb = build_embeddings()
    out_dir = project_root / "data" / "index" / "faiss"
    persist_faiss(docs, emb, out_dir)

    smoke_test(out_dir, emb)


if __name__ == "__main__":
    main()