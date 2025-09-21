
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


# Optional: load .env if available and report status like your logs show
def load_dotenv_and_report(project_root: Path) -> None:
    env_path = project_root / ".env"
    exists = env_path.exists()
    loaded = False
    try:
        from dotenv import load_dotenv  # type: ignore
        loaded = load_dotenv(env_path if exists else None)
    except Exception:
        # dotenv not installed or failed; keep loaded = False
        loaded = False
    print(f"dotenv: {env_path} exists={exists} loaded={bool(loaded)}")


def debug_meta_prints() -> None:
    """Print interpreter/debug info when DEBUG_META=1."""
    if os.environ.get("DEBUG_META") not in {"1", "true", "True"}:
        return
    try:
        import debugpy  # type: ignore
        dbg_ver = getattr(debugpy, "__version__", "unknown")
    except Exception:
        dbg_ver = "not-loaded"
    print("exe:", sys.executable)
    print("debugpy:", dbg_ver)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))


def discover_db_path(project_root: Path) -> Path:
    """
    Resolve the SQLite database path.

    Priority:
    1) DB_PATH from env (absolute or relative to project root)
    2) First *.sqlite3 under ./data
    3) Default to ./data/northwind.sqlite3
    """
    env_val = os.environ.get("DB_PATH")
    if env_val:
        p = Path(env_val)
        if not p.is_absolute():
            p = project_root / p
        return p

    data_dir = project_root / "data"
    if data_dir.exists():
        candidates = sorted(data_dir.glob("*.sqlite3"))
        if candidates:
            return candidates[0]

    return project_root / "data" / "northwind.sqlite3"


def question_to_sql(question: str) -> str:
    """
    Extremely small heuristic text-to-SQL for demo purposes.

    NOTE: This is not a general NL2SQL model. It only handles a few patterns
    relevant to Northwind-like schemas. Extend as needed.
    """
    q_norm = " ".join(question.lower().split())

    # Example handled in your run:
    # "Revenue by category and quarter for 2018"
    if (
        ("revenue" in q_norm)
        and ("category" in q_norm or "categories" in q_norm)
        and ("quarter" in q_norm or "by quarter" in q_norm)
        and ("2018" in q_norm)
    ):
        return (
            "SELECT "
            "CAST(((CAST(strftime('%m', Orders.OrderDate) AS INTEGER) + 2) / 3) AS INTEGER) AS Quarter, "
            "Categories.CategoryName, "
            "SUM(\"Order Details\".UnitPrice * \"Order Details\".Quantity * (1 - \"Order Details\".Discount)) AS Revenue "
            "FROM Orders "
            "INNER JOIN \"Order Details\" ON Orders.OrderID = \"Order Details\".OrderID "
            "INNER JOIN Products ON \"Order Details\".ProductID = Products.ProductID "
            "INNER JOIN Categories ON Products.CategoryID = Categories.CategoryID "
            "WHERE CAST(strftime('%Y', Orders.OrderDate) AS INTEGER) = 2018 "
            "GROUP BY Quarter, Categories.CategoryName"
        )

    # Add more patterns here as needed.
    # Fallback: raise a helpful error to prompt adding a new pattern.
    raise ValueError(
        "I don't recognize this question. "
        "Please add a pattern in question_to_sql() to handle it."
    )


def execute_query(conn: sqlite3.Connection, sql: str, limit: int | None = 100) -> Tuple[List[str], List[Tuple[object, ...]]]:
    """
    Execute SQL and return (column_names, rows). Optionally wrap with LIMIT.
    """
    to_run = sql
    if limit is not None:
        to_run = f"SELECT * FROM ({sql}) AS _sub LIMIT {int(limit)}"

    print("Generated SQL:")
    print(sql)
    print("---- SQL to execute ----")
    print(to_run)
    print("------------------------")

    cur = conn.cursor()
    cur.execute(to_run)
    rows: List[Tuple[object, ...]] = cur.fetchall()
    headers: List[str] = [d[0] for d in cur.description] if cur.description else []
    return headers, rows


def _col_widths(headers: Sequence[str], rows: Sequence[Sequence[object]], max_width: int = 50) -> List[int]:
    widths = [min(len(h), max_width) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            cell_str = "" if cell is None else str(cell)
            widths[i] = min(max(widths[i], len(cell_str)), max_width)
    return widths


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    if not headers:
        print("(no columns)")
        return

    widths = _col_widths(headers, rows)
    # Header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * w for w in widths)
    print(header_line)
    print(sep_line)
    # Rows
    for row in rows:
        print(" | ".join(("" if v is None else str(v)).ljust(widths[i]) for i, v in enumerate(row)))
    print(f"Rows: {len(rows)}")


# ---- Type-safe conversion helpers (resolve Pylance "reportArgumentType") ----
def to_int(x: object) -> int:
    if x is None:
        raise ValueError("Expected int-like value, got None")
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            raise ValueError("Empty string is not int-like")
        # Handle numbers formatted as '12.0' or scientific notation
        return int(float(s)) if any(c in s for c in (".", "e", "E")) else int(s)
    raise TypeError(f"Unsupported type for int conversion: {type(x).__name__}")


def to_float(x: object) -> float:
    if x is None:
        raise ValueError("Expected float-like value, got None")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s == "":
            raise ValueError("Empty string is not float-like")
        return float(s)
    raise TypeError(f"Unsupported type for float conversion: {type(x).__name__}")


def to_float_or_zero(x: object) -> float:
    return 0.0 if x is None else to_float(x)


def build_pivot(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> Tuple[List[int], List[str], Dict[str, Dict[int, float]]]:
    """Return (quarters, categories, pivot) where pivot[cat][q] = revenue."""
    q_idx = headers.index("Quarter")
    c_idx = headers.index("CategoryName")
    r_idx = headers.index("Revenue")

    quarters = sorted({to_int(r[q_idx]) for r in rows})
    categories = sorted({str(r[c_idx]) for r in rows})

    pivot: Dict[str, Dict[int, float]] = {c: {q: 0.0 for q in quarters} for c in categories}
    for r in rows:
        q = to_int(r[q_idx])
        c = str(r[c_idx])
        v = to_float_or_zero(r[r_idx])
        pivot[c][q] = v
    return quarters, categories, pivot


def print_pivot_wide(headers: Sequence[str], rows: Sequence[Sequence[object]], ndigits: int = 2) -> None:
    quarters, categories, pivot = build_pivot(headers, rows)

    q_headers = [f"Q{q}" for q in quarters]
    title = ["Category"] + q_headers + ["Total"]
    widths = [max(8, len(h)) for h in title]

    # Compute totals for sorting and width
    totals = {c: round(sum(pivot[c].values()), ndigits) for c in categories}
    widths[0] = max(widths[0], max(len(c) for c in categories))
    for i, q in enumerate(quarters, start=1):
        widths[i] = max(widths[i], max(len(f"{pivot[c][q]:,.{ndigits}f}") for c in categories))
    widths[-1] = max(widths[-1], max(len(f"{totals[c]:,.{ndigits}f}") for c in categories))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(title))
    sep_line = "-+-".join("-" * w for w in widths)
    print(header_line)
    print(sep_line)

    # Rows
    for c in categories:
        row_vals = [c]
        row_vals += [f"{pivot[c][q]:,.{ndigits}f}" for q in quarters]
        row_vals += [f"{totals[c]:,.{ndigits}f}"]
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row_vals)))


def save_pivot_csv(headers: Sequence[str], rows: Sequence[Sequence[object]], csv_path: str, ndigits: int = 2) -> None:
    quarters, categories, pivot = build_pivot(headers, rows)
    fieldnames = ["Category"] + [f"Q{q}" for q in quarters] + ["Total"]
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in categories:
            row: Dict[str, Any] = {"Category": c}
            total = 0.0
            for q in quarters:
                v = round(pivot[c][q], ndigits)
                row[f"Q{q}"] = v  # value is float; Dict[str, Any] is acceptable
                total += v
            row["Total"] = round(total, ndigits)
            w.writerow(row)
    print(f"Saved CSV: {out_path}")


def plot_pivot(headers: Sequence[str], rows: Sequence[Sequence[object]], out_path: str | None = None, ndigits: int = 0) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Plot skipped (matplotlib not available). Install with: pip install matplotlib")
        return

    quarters, categories, pivot = build_pivot(headers, rows)
    x = list(range(len(categories)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4f46e5", "#16a34a", "#f59e0b", "#ef4444", "#0ea5e9"]
    for i, q in enumerate(quarters):
        vals = [pivot[c][q] for c in categories]
        ax.bar([xi + (i - (len(quarters)-1)/2) * width for xi in x], vals, width, label=f"Q{q}", color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue by Category and Quarter")
    ax.legend()
    fig.tight_layout()

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150)
        print(f"Saved chart: {out_p}")
    else:
        plt.show()


def ensure_db_exists(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at: {db_path}\n"
            "Set DB_PATH in your .env or place a .sqlite3 file in ./data"
        )


def get_project_root() -> Path:
    # Assume this file lives at <project_root>/src/app.py
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Northwind NL2SQL demo runner")
    parser.add_argument("question", nargs="?", help="Natural language question", default=None)
    parser.add_argument("--limit", type=int, default=100, help="Row limit (default: 100)")
    parser.add_argument("--no-limit", action="store_true", help="Disable LIMIT wrapper")

    # New options
    parser.add_argument("--wide", action="store_true", help="Print a wide pivot (Category x Quarter)")
    parser.add_argument("--round", dest="round_ndigits", type=int, default=2, help="Decimal places for rounding")
    parser.add_argument("--csv", type=str, default=None, help="Save pivot as CSV to this path")
    parser.add_argument("--plot", type=str, default=None, help="Save a grouped bar chart image (PNG path)")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(argv)

    project_root = get_project_root()
    load_dotenv_and_report(project_root)
    debug_meta_prints()

    db_path = discover_db_path(project_root)
    print(f"DB path: {db_path}")

    ensure_db_exists(db_path)

    # Resolve the user's question
    question = args.question or os.environ.get(
        "DEFAULT_QUESTION", "Revenue by category and quarter for 2018"
    )
    print(f"Q: {question}")

    # Translate to SQL (naive heuristic patterns)
    try:
        sql = question_to_sql(question)
    except Exception as e:
        print("Could not generate SQL for this question.")
        print("Reason:", e)
        return 2

    # Run query
    conn = sqlite3.connect(str(db_path))
    try:
        limit = None if args.no_limit else args.limit
        headers, rows = execute_query(conn, sql, limit=limit)
    finally:
        conn.close()

    # Pretty print (narrow)
    if rows:
        print_table(headers, rows)
    else:
        print("(no rows)")

    # Optional wide/CSV/plot
    if rows:
        if args.wide:
            print("\nWide view (Category x Quarter):")
            print_pivot_wide(headers, rows, ndigits=args.round_ndigits)

        if args.csv:
            save_pivot_csv(headers, rows, args.csv, ndigits=args.round_ndigits)

        if args.plot:
            plot_pivot(headers, rows, out_path=args.plot, ndigits=args.round_ndigits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
