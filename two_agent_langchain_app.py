
import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv

# LLM + embeddings (Google Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Vector store + loaders
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# DB + data handling
from sqlalchemy import create_engine, text
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

# Pretty output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# -----------------------------
# Simple conversation memory (no deprecation warnings)
# -----------------------------
class SimpleTextMemory:
    """
    Minimal text-based memory for multi-turn refinement.
    Stores "User: ...\nSQL: ..." entries and returns a flat text history.
    API mirrors just what the app needs from LangChain's memory.
    """
    def __init__(self):
        self.lines: List[str] = []

    def load_memory_variables(self, _):
        return {"history": "\n".join(self.lines)}

    def save_context(self, inputs, outputs):
        user = inputs.get("input", "")
        sql = outputs.get("output", "")
        self.lines.append(f"User: {user}\nSQL: {sql}\n")

    def clear(self):
        self.lines.clear()


# -----------------------------
# Helpers
# -----------------------------
def normalize_embed_model(name: str) -> str:
    """
    Ensure embeddings model name has the 'models/' prefix for the gRPC path.
    Do NOT apply this to chat models.
    """
    name = (name or "").strip()
    return name if name.startswith("models/") else f"models/{name}"


# -----------------------------
# Configuration & environment
# -----------------------------
def load_config():
    # Load envs from data/.env.ini (override), then .env if present
    if Path("data/.env.ini").exists():
        load_dotenv("data/.env.ini", override=True)
    load_dotenv(override=False)

    cfg = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        "GEMINI_EMBEDDING_MODEL": normalize_embed_model(os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")),
        "DATABASE_PATH": os.getenv("DATABASE_PATH", "data/northwind.sqlite3"),
        "VECTOR_DIR": os.getenv("VECTOR_DIR", "data/vectorstore_lc"),
        "SCHEMA_DIR": os.getenv("SCHEMA_DIR", "docs/schema"),
    }
    if not cfg["GOOGLE_API_KEY"]:
        console.print("[bold red]ERROR:[/bold red] GOOGLE_API_KEY is not set.")
        sys.exit(1)
    return cfg


# -----------------------------
# Vector store build/load
# -----------------------------
def build_or_load_vectorstore(schema_dir: str, vector_dir: str, embed_model_name: str) -> FAISS:
    """
    Build FAISS from docs/schema if not present; else load.
    """
    vpath = Path(vector_dir)
    vpath.mkdir(parents=True, exist_ok=True)

    # Try load existing
    if (vpath / "index.faiss").exists() and (vpath / "index.pkl").exists():
        embeddings = GoogleGenerativeAIEmbeddings(model=embed_model_name)
        console.print(f"[green]Loaded existing FAISS index from[/green] {vector_dir}")
        return FAISS.load_local(vector_dir, embeddings, allow_dangerous_deserialization=True)

    # Build new from text files in schema_dir
    schema_path = Path(schema_dir)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema dir not found: {schema_dir}")

    loaders = []
    if (schema_path / "schema.md").exists():
        loaders.append(TextLoader(str(schema_path / "schema.md"), autodetect_encoding=True))
    else:
        loaders.append(DirectoryLoader(schema_dir, glob="**/*.md"))
        loaders.append(DirectoryLoader(schema_dir, glob="**/*.txt"))

    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception:
            pass

    if not docs:
        raise RuntimeError(f"No documents loaded from {schema_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model_name)
    console.print(f"[cyan]Building FAISS index with embeddings model[/cyan] {embed_model_name}")
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(vector_dir)
    console.print(f"[green]Saved FAISS index to[/green] {vector_dir}")
    return store


# -----------------------------
# Agents
# -----------------------------
SQL_EXTRA_RULES = """
- Use the exact table and column names from context.
- Date: use Orders.OrderDate. For SQLite, STRFTIME on datetime(Orders.OrderDate) for grouping/filtering.
- Measure TotalRevenue = SUM("Order Details".Quantity * "Order Details".UnitPrice * (1 - "Order Details".Discount)).
- Joins:
  Orders.OrderID = "Order Details".OrderID
  Products.ProductID = "Order Details".ProductID
  Categories.CategoryID = Products.CategoryID
  Customers.CustomerID = Orders.CustomerID
- When grouping by country, use Customers.Country (do NOT use Customers.CompanyName).
- Avoid nested aggregates like AVG(SUM(...)). Compute inner aggregates (e.g., order totals) in a subquery/CTE then aggregate.
- Quarter index in SQLite: ((CAST(STRFTIME('%m', datetime(Orders.OrderDate)) AS INTEGER) - 1) / 3) + 1 to get 1–4.
"""

PLANNER_SYS_PROMPT = f"""
You are Agent 2 (Planner), an expert NL→SQL generator for SQLite on the Northwind dataset.
You DO NOT execute SQL. You only produce correct SQL given the user's question and the retrieved context.

Requirements:
- Use only tables/columns that exist in the provided context.
- Prefer STRFTIME on datetime(Orders.OrderDate) for date grouping and filtering.
- Unless the user explicitly asked for a LIMIT, DO NOT include a LIMIT clause.
- Return only a single SQL query in a fenced code block like:

```sql
SELECT ...
```

Extra context and rules:
{SQL_EXTRA_RULES}
"""

EXECUTOR_SYS_PROMPT = """
You are Agent 1 (Executor). You receive a SQL query, execute it on the SQLite database, and return clean, tabular results.
You MUST NOT modify the SQL; just run it. If it fails, return the error message.
"""


def extract_sql(text: str) -> str:
    """Pull the first fenced ```sql ... ``` block or fallback to first SELECT ... ; occurrence."""
    fence = re.search(r"```sql(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip().strip(";") + ";"
    m = re.search(r"(SELECT\b[\s\S]+?;)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def strip_trailing_limit(sql: str) -> str:
    return re.sub(r"\s+LIMIT\s+\d+\s*;?\s*$", "", sql, flags=re.IGNORECASE)


class PlannerAgent:
    def __init__(self, model_name: str, retriever, api_key: str):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        self.retriever = retriever
        # Simple conversation memory (no LangChain deprecation warnings)
        self.memory = SimpleTextMemory()

    def plan(self, question: str) -> Tuple[str, list, str]:
        # Load conversation history (text)
        history = self.memory.load_memory_variables({}).get("history", "")

        # Retrieve top-k docs (invoke is the new API)
        docs = self.retriever.invoke(question)
        context_texts = [d.page_content for d in docs]
        context = "\n\n".join(context_texts)

        # Build prompt with memory + retrieved context
        prompt_parts = [PLANNER_SYS_PROMPT]
        if history:
            prompt_parts.append(f"Conversation history:\n{history}")
        prompt_parts.extend([
            f"Retrieved context:\n{context}",
            f"User question:\n{question}",
            "Return only one SQL query in a single ```sql fenced block.",
        ])
        prompt = "\n\n".join(prompt_parts)

        resp = self.llm.invoke(prompt)
        content = getattr(resp, "content", str(resp))
        sql = extract_sql(content)

        # Save this turn in memory for follow-ups
        self.memory.save_context({"input": question}, {"output": sql})
        return sql, context_texts, content


class ExecutorAgent:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")

    def run(self, sql: str, rows: int = 50):
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
        except Exception as e:
            return ["error"], [[str(e)]]
        if rows and rows > 0:
            df = df.head(rows)
        return list(df.columns), df.astype(object).values.tolist()


# -----------------------------
# Utilities
# -----------------------------
def print_table(headers: List[str], rows: List[List]):
    table = Table(show_lines=False)
    for h in headers:
        table.add_column(str(h))
    for r in rows:
        table.add_row(*[str(x) for x in r])
    console.print(table)


def save_side_outputs(headers: List[str], rows: List[List], csv_path: Optional[str], md_path: Optional[str]):
    if not (csv_path or md_path):
        return
    df = pd.DataFrame(rows, columns=headers)
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        console.print(f"Saved CSV -> {csv_path}")
    if md_path:
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(md_path).write_text(df.to_markdown(index=False), encoding="utf-8")
        console.print(f"Saved Markdown -> {md_path}")


def save_plot(headers: List[str], rows: List[List], plot_path: Optional[str], kind: str = "line"):
    if not plot_path:
        return
    if len(headers) < 2 or not rows:
        console.print("[yellow]Skip plot: need at least 2 columns and some rows.[/yellow]")
        return
    df = pd.DataFrame(rows, columns=headers)
    xcol, ycol = headers[0], headers[1]
    plt.figure(figsize=(9, 4))
    if kind == "bar":
        plt.bar(df[xcol], pd.to_numeric(df[ycol], errors="coerce"))
    else:
        plt.plot(df[xcol], pd.to_numeric(df[ycol], errors="coerce"), marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{ycol} by {xcol}")
    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=160)
    plt.close()
    console.print(f"Saved plot -> {plot_path}")


# -----------------------------
# Main run modes
# -----------------------------
def run_once(planner: PlannerAgent, executor: ExecutorAgent, question: str, rows: int,
             csv: Optional[str], md: Optional[str], plot: Optional[str], plot_kind: str,
             strip_limit: bool):
    sql, ctx, raw = planner.plan(question)
    if strip_limit:
        sql = strip_trailing_limit(sql)

    console.rule("[bold]SQL[/bold]")
    console.print(sql)

    headers, data = executor.run(sql, rows=rows)

    console.rule("[bold]Results[/bold]")
    print_table(headers, data)

    save_side_outputs(headers, data, csv, md)
    save_plot(headers, data, plot, kind=plot_kind)


def repl(planner: PlannerAgent, executor: ExecutorAgent, rows: int, strip_limit: bool):
    console.print(Panel.fit("Agentic RAG REPL — type your question. Type 'exit' to quit.\nCommands: history | clear | help", style="cyan"))
    while True:
        try:
            q = input("q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        low = q.lower()
        if low in {"exit", "quit", ":q"}:
            break
        if low in {"help", "/help", "?"}:
            console.print("Commands: exit | quit | :q | clear | history | help")
            continue
        if low in {"clear", "/clear"}:
            planner.memory.clear()
            console.print("[green]Conversation memory cleared.[/green]")
            continue
        if low in {"history", "/history"}:
            hist = planner.memory.load_memory_variables({}).get("history", "")
            console.print(Panel(hist or "(empty)", title="Conversation history"))
            continue

        run_once(planner, executor, q, rows, None, None, None, "line", strip_limit)


# -----------------------------
# CLI
# -----------------------------
def main():
    cfg = load_config()

    # Build or load vectorstore & retriever
    store = build_or_load_vectorstore(
        schema_dir=cfg["SCHEMA_DIR"],
        vector_dir=cfg["VECTOR_DIR"],
        embed_model_name=cfg["GEMINI_EMBEDDING_MODEL"],
    )
    retriever = store.as_retriever(search_kwargs={"k": 4})

    # Agents
    planner = PlannerAgent(
        model_name=cfg["GEMINI_MODEL"],   # chat models should NOT have 'models/' prefix
        retriever=retriever,
        api_key=cfg["GOOGLE_API_KEY"],
    )
    executor = ExecutorAgent(db_path=cfg["DATABASE_PATH"])

    parser = argparse.ArgumentParser(description="Two-agent Agentic RAG for NL→SQL on Northwind")
    parser.add_argument("--q", type=str, help="One-shot question")
    parser.add_argument("--repl", action="store_true", help="Interactive REPL")
    parser.add_argument("--rows", type=int, default=50, help="Rows to display")
    parser.add_argument("--csv", type=str, help="Path to save CSV")
    parser.add_argument("--md", type=str, help="Path to save Markdown table")
    parser.add_argument("--plot", type=str, help="Path to save plot image")
    parser.add_argument("--plot-kind", type=str, default="line", choices=["line", "bar"], help="Plot kind")
    parser.add_argument("--strip-limit", action="store_true", help="Remove trailing LIMIT from generated SQL")
    args = parser.parse_args()

    if args.q:
        run_once(planner, executor, args.q, args.rows, args.csv, args.md, args.plot, args.plot_kind, args.strip_limit)
    elif args.repl:
        repl(planner, executor, args.rows, args.strip_limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    # Optional: quiet TensorFlow logs if present
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    main()
