import os, sqlite3, textwrap
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_tables(conn) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]

def get_schema_block(conn, table: str, sample_rows: int = 3) -> str:
    cur = conn.cursor()
    # Columns
    cur.execute(f"PRAGMA table_info('{table}')")
    cols = cur.fetchall()
    col_lines = [f"- {c[1]}: {c[2]}" for c in cols]  # name, type

    # FKs
    cur.execute(f"PRAGMA foreign_key_list('{table}')")
    fks = cur.fetchall()
    fk_lines = [f"- {fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks] if fks else []

    # Sample rows
    cur.execute(f'SELECT * FROM "{table}" LIMIT ?', (sample_rows,))
    rows = cur.fetchall()
    headers = [d[0] for d in cur.description] if cur.description else []
    sample_text = ""
    if rows:
        sample_text = "Sample rows (limited):\n" + "\n".join(
            [", ".join(f"{h}={repr(v)}" for h, v in zip(headers, r)) for r in rows]
        )

    block = f"""Table: {table}
Columns:
{os.linesep.join(col_lines) if col_lines else "(none)"}
{"Foreign Keys:\n" + os.linesep.join(fk_lines) if fk_lines else "Foreign Keys: (none)"}
{sample_text if sample_text else ""}
"""
    return textwrap.dedent(block).strip()

def build_docs(db_path: Path, sample_rows: int = 3) -> List[str]:
    conn = sqlite3.connect(db_path)
    try:
        tables = get_tables(conn)
        docs = []
        for t in tables:
            docs.append(get_schema_block(conn, t, sample_rows))
        return docs
    finally:
        conn.close()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite DB")
    ap.add_argument("--out", default="data/vectorstore", help="Output folder for FAISS index")
    ap.add_argument("--rows", type=int, default=3, help="Sample rows per table to include as context")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    args = ap.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building schema docs from: {db_path}")
    docs = build_docs(db_path, args.rows)
    print(f"Collected {len(docs)} table docs")

    texts = docs
    embeddings = HuggingFaceEmbeddings(model_name=args.model)
    vs = FAISS.from_texts(texts, embeddings)
    vs.save_local(str(out_dir))
    print(f"Saved FAISS index to: {out_dir}")

if __name__ == "__main__":
    main()
