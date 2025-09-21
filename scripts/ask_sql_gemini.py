
import argparse, os, re, sqlite3, time, csv, json, math
from pathlib import Path
from typing import Tuple, List, Optional, Any

# Gemini SDK and errors
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

# Env and RAG deps
from dotenv import load_dotenv
load_dotenv("data/.env.ini")  # loads key=value lines into os.environ
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# =========================
# RAG: load vector context
# =========================
def load_context(vs_path: str, question: str, k: int = 6) -> str:
    if k <= 0:
        return ""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question, k=k)
    return "\n\n".join(d.page_content for d in docs)


# =========================
# SQL normalization
# =========================
def strip_sql(text: str) -> str:
    t = (text or "").strip()
    # 1) Remove code fences (```sql, ```sqlite, ``` etc.)
    t = re.sub(r"```[a-zA-Z0-9_-]*\s*", "", t)  # opening fence with optional lang
    t = t.replace("```", "")                     # any remaining backticks
    # 2) Keep from the first WITH/SELECT onward
    m = re.search(r"(?is)\b(with|select)\b[\s\S]*", t)
    if m:
        t = m.group(0)
    t = t.strip("` \n\r\t;")
    # 3) Ensure we start at WITH/SELECT even if stray chars remain
    m = re.search(r"(?is)\b(with|select)\b", t)
    if m:
        t = t[m.start():].strip("` \n\r\t;")
    return t


def is_safe_sql(sql: str) -> bool:
    s = re.sub(r"(?m)^\s*--.*$", "", sql or "").strip().lower()
    return bool(re.match(r"^(with|select)\b", s))


def ensure_limit(sql: str, default_limit: int = 10) -> str:
    base = sql.strip().rstrip(";")
    return base if re.search(r"\blimit\b", base, re.I) else base + f" LIMIT {default_limit}"


# =========================
# DB helpers
# =========================
def run_sql(db_path: Path, sql: str) -> Tuple[List[str], List[tuple]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        headers = [d[0] for d in cur.description] if cur.description else []
        return headers, rows
    finally:
        conn.close()


def run_raw_sql(db_path: Path, sql: str) -> Tuple[str, List[str], List[tuple]]:
    cleaned = sql.strip().rstrip(";")
    headers, rows = run_sql(db_path, cleaned)
    return cleaned, headers, rows


def _all_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def _sample_date_columns(conn: sqlite3.Connection, max_tables: int = 12, samples_per_col: int = 3) -> List[str]:
    """
    Find columns likely to be dates/times and collect a few non-null samples with typeof.
    Returns lines like: '- Orders.OrderDate (typeof: text) e.g. 1997-04-23 00:00:00, 1997-05-10 00:00:00'
    """
    lines: List[str] = []
    tables = _all_tables(conn)[:max_tables]
    for t in tables:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = cur.fetchall()
        dateish = [c for c in cols if re.search(r"(date|time)", c[1], re.I)]
        for c in dateish:
            col = c[1]
            try:
                cur.execute(f'SELECT {col}, typeof({col}) FROM "{t}" WHERE {col} IS NOT NULL LIMIT ?', (samples_per_col,))
                rows = cur.fetchall()
                if not rows:
                    continue
                types = list({r[1] for r in rows})
                samples = [str(r[0]) for r in rows]
                samples = [s[:25] + "…" if len(s or "") > 28 else s for s in samples]
                lines.append(f"- {t}.{col} (typeof: {', '.join(types)}) e.g. {', '.join(samples)}")
            except Exception:
                continue
    return lines


def build_date_hints(db_path: Path) -> str:
    """
    Introspects the DB to provide the model hints about date columns and formats.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        lines = _sample_date_columns(conn)
        if not lines:
            return ""
        prefix = "Date/time columns discovered (use STRFTIME; wrap in datetime(...) if needed):\n"
        return prefix + "\n".join(lines)
    finally:
        conn.close()


# =========================
# Output helpers
# =========================
def to_markdown(headers: List[str], rows: List[tuple]) -> str:
    if not headers:
        return ""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_outputs(headers, rows, csv_path=None, json_path=None, md_path=None):
    if csv_path and headers:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        print(f"Saved CSV -> {csv_path}")
    if json_path and headers:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        data = [dict(zip(headers, r)) for r in rows]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON -> {json_path}")
    if md_path and headers:
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        md = to_markdown(headers, rows)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved Markdown -> {md_path}")


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", ""))
    except Exception:
        return None


def plot_results(headers: List[str], rows: List[tuple], out_path: Optional[str], kind: str = "line"):
    """
    Saves a simple plot if --plot was provided and matplotlib is available.
    Uses first column as X (labels) and second column as Y (numeric).
    """
    if not out_path:
        return
    if not headers or len(headers) < 2 or not rows:
        print("(Plot) Skipped: need at least two columns and some rows")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(Plot) matplotlib not installed. Run: pip install matplotlib")
        return

    x = [str(r[0]) for r in rows]
    y = [_to_float(r[1]) for r in rows]
    y = [val for val in y if val is not None]
    if not y:
        print("(Plot) Skipped: second column is not numeric")
        return

    plt.figure(figsize=(8, 4))
    if kind == "bar":
        plt.bar(x, y, color="#4e79a7")
    else:
        plt.plot(x, y, marker="o", color="#4e79a7")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot -> {out_path}")


# =========================
# Gemini SDK compatibility
# =========================
def _get_model_and_configure(api_key: str, model_name: str):
    """
    Returns a configured GenerativeModel instance.
    - First tries public API via dynamic getattr (prevents Pylance export warnings).
    - Falls back to internal modules if attributes are missing.
    """
    try:
        cfg = getattr(genai, "configure")
        GenModel = getattr(genai, "GenerativeModel")
        cfg(api_key=api_key)
        return GenModel(model_name)
    except AttributeError:
        from google.generativeai.client import configure as _configure
        from google.generativeai.generative_models import GenerativeModel as _GenerativeModel
        _configure(api_key=api_key)
        return _GenerativeModel(model_name)


# =========================
# Prompting
# =========================
MEASURE_HINTS = """Common measures and aliases (prefer exactly these):
- TotalRevenue = SUM("Order Details".Quantity * "Order Details".UnitPrice * (1 - "Order Details".Discount))
- TotalQuantityOrdered = SUM("Order Details".Quantity)
- OrderCount = COUNT(DISTINCT Orders.OrderID)
Use Orders.OrderDate for time grouping when relevant.
Return stable, descriptive aliases that match the measure names above.
"""

PROMPT_SYSTEM = """You are an expert SQLite query writer.
Given a user question and a schema context, produce ONE correct, safe SQLite SELECT query.
Rules:
- Use only tables/columns that appear in the context.
- Use double quotes for identifiers that contain spaces (e.g., "Order Details").
- Do NOT modify data. No INSERT/UPDATE/DELETE/CREATE/DROP/ALTER.
- Prefer explicit JOINs with correct keys.
- If the user didn't specify a limit, include LIMIT 10 at the end.
- Return only the SQL, no explanations.
- Do NOT use Markdown code fences; return raw SQL only.
"""

PROMPT_USER_TEMPLATE = """Schema context:
{context}

Additional hints:
{hints}

User question:
{question}

Write a single SQLite query. Return SQL only.
"""


# =========================
# Core ask function
# =========================
def ask_sql_gemini(
    db_path: Path,
    vs_path: Path,
    question: str,
    default_limit: int = 10,
    retry: int = 1,
    model_name: str = "gemini-1.5-flash",
    kdocs: int = 6,
):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Put it in a .env at project root or in your environment.")

    # Enrich prompt with date hints from the live DB
    date_hints = build_date_hints(db_path)
    extra_hints = (MEASURE_HINTS + ("\n" + date_hints if date_hints else "")).strip()

    context = load_context(str(vs_path), question, k=kdocs)
    models_to_try = [model_name] + ([] if model_name == "gemini-1.5-flash" else ["gemini-1.5-flash"])

    def call_model(mname: str, prompt: str, attempts: int = 2) -> str:
        delay = 8
        last = None
        for _ in range(attempts):
            try:
                model = _get_model_and_configure(api_key, mname)
                resp = model.generate_content(prompt)
                return (getattr(resp, "text", None) or "").strip()
            except ResourceExhausted as e:
                last = e
                time.sleep(delay)
                delay = min(int(delay * 1.6), 30)
            except GoogleAPIError as e:
                last = e
                break
        if last:
            raise last
        raise RuntimeError("Unknown error calling model.")

    def gen_sql(q: str, ctx: str) -> str:
        prompt = PROMPT_SYSTEM + "\n\n" + PROMPT_USER_TEMPLATE.format(context=ctx, hints=extra_hints, question=q)
        last_err = None
        for mname in models_to_try:
            try:
                text = call_model(mname, prompt, attempts=2)
                sql = strip_sql(text)
                if not is_safe_sql(sql):
                    m = re.search(r"(?is)\b(with|select)\b[\s\S]*", text or "")
                    if m:
                        sql = m.group(0).strip("` \n\r\t;")
                if not is_safe_sql(sql):
                    raise ValueError(f"Generated SQL not safe after normalization: {sql[:160]}")
                if mname != model_name:
                    print(f"(Info) Fell back to model: {mname}")
                return ensure_limit(sql, default_limit)
            except (ResourceExhausted, GoogleAPIError, ValueError) as e:
                last_err = e
                continue
        raise last_err if last_err else RuntimeError("All models failed.")

    sql = gen_sql(question, context)
    try:
        headers, rows = run_sql(db_path, sql)
        return sql, headers, rows
    except Exception as e:
        if retry <= 0:
            raise
        fix_q = f"{question}\nNote: The previous SQL failed with error: {e}\nPlease correct it."
        sql2 = gen_sql(fix_q, context)
        headers, rows = run_sql(db_path, sql2)
        return sql2, headers, rows


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/1751978414.sqlite3", help="Path to SQLite DB")
    ap.add_argument("--q", help="Natural language question for Gemini")
    ap.add_argument("--k", type=int, default=10, help="Default LIMIT if not present")
    ap.add_argument("--model", default="gemini-1.5-flash", help="Gemini model name (e.g., gemini-1.5-flash or gemini-1.5-pro)")
    ap.add_argument("--csv", help="Write results to CSV")
    ap.add_argument("--json", help="Write results to JSON")
    ap.add_argument("--md", help="Write results to Markdown table")
    ap.add_argument("--plot", help="Save a simple plot (PNG). Uses first column as X, second as numeric Y.")
    ap.add_argument("--plot-kind", choices=["line", "bar"], default="line", help="Plot style (default: line)")
    ap.add_argument("--rows", type=int, default=20, help="Rows to print to console")
    ap.add_argument("--kdocs", type=int, default=6, help="Number of retrieved context chunks (lower reduces tokens)")
    ap.add_argument("--repl", action="store_true", help="Interactive loop")
    # New: direct SQL execution
    ap.add_argument("--sql", help="Run this literal SQL directly (bypass Gemini)")
    ap.add_argument("--sql-file", help="Run SQL from a file (bypass Gemini)")
    args = ap.parse_args()

    db_path = Path(args.db)
    vs_path = Path("data/vectorstore")

    def run_once_question(q: str):
        sql, headers, rows = ask_sql_gemini(
            db_path, vs_path, q,
            default_limit=args.k,
            model_name=args.model,
            kdocs=args.kdocs
        )
        print("\n--- SQL ---")
        print(sql)
        print("\n--- Results ---")
        if headers:
            print("\t".join(headers))
            for r in rows[:args.rows]:
                print("\t".join(str(x) for x in r))
        else:
            print("(no result columns)")
        write_outputs(headers, rows, args.csv, args.json, args.md)
        plot_results(headers, rows, args.plot, kind=args.plot_kind)

    def run_once_sql(sql_text: str):
        sql_text = sql_text.strip()
        print("\n--- SQL (raw) ---")
        print(sql_text)
        cleaned, headers, rows = run_raw_sql(db_path, sql_text)
        print("\n--- Results ---")
        if headers:
            print("\t".join(headers))
            for r in rows[:args.rows]:
                print("\t".join(str(x) for x in r))
        else:
            print("(no result columns)")
        write_outputs(headers, rows, args.csv, args.json, args.md)
        plot_results(headers, rows, args.plot, kind=args.plot_kind)

    # Choose mode
    if args.sql or args.sql_file:
        if args.sql_file:
            with open(args.sql_file, "r", encoding="utf-8") as f:
                sql_text = f.read()
        else:
            sql_text = args.sql
        run_once_sql(sql_text)
        return

    if args.repl:
        # Interactive mode (Gemini)
        while True:
            try:
                q = input("\nq> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit", "q"}:
                break
            run_once_question(q)
    else:
        if not args.q:
            raise SystemExit("Provide --q for Gemini mode, or --sql/--sql-file for direct SQL mode.")
        run_once_question(args.q)


if __name__ == "__main__":
    main()
