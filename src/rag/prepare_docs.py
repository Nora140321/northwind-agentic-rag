
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


def qident(name: str) -> str:
    """
    Quote an SQLite identifier (table/column) safely for PRAGMA statements.
    Handles names with spaces or quotes.
    """
    return '"' + name.replace('"', '""') + '"'


def list_tables(conn: sqlite3.Connection, include_views: bool = False) -> List[str]:
    """
    Return a sorted list of user tables (and optionally views), excluding internal sqlite_%.
    """
    cur = conn.cursor()
    if include_views:
        rows = cur.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    else:
        rows = cur.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

    names = [r[0] for r in rows]
    # Exclude common internal artifacts if present
    exclude = {"sqlite_sequence"}
    return [n for n in names if n not in exclude]


def get_table_columns(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    """
    Return columns for a table using PRAGMA table_info.
    """
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA table_info({qident(table)})").fetchall()
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    cols: List[Dict[str, Any]] = []
    for cid, name, coltype, notnull, dflt_value, pk in rows:
        cols.append(
            {
                "name": name,
                "type": coltype,
                "notnull": bool(notnull),
                "default": dflt_value,
                "pk": bool(pk),
            }
        )
    return cols


def get_foreign_keys(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    """
    Return foreign keys for a table using PRAGMA foreign_key_list.
    """
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA foreign_key_list({qident(table)})").fetchall()
    # PRAGMA foreign_key_list returns:
    # (id, seq, table, from, to, on_update, on_delete, match)
    fks: List[Dict[str, Any]] = []
    for (_id, _seq, ref_table, frm, to, on_update, on_delete, match) in rows:
        fks.append(
            {
                "from": frm,
                "to_table": ref_table,
                "to_col": to,
                "on_update": on_update,
                "on_delete": on_delete,
                "match": match,
            }
        )
    return fks


def extract_schema(
    db_path: Path,
    include_views: bool = False,
    only_tables: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Extract schema for all (or selected) tables in an SQLite database.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        all_tables = list_tables(conn, include_views=include_views)
        if only_tables:
            lower_allow = {t.lower() for t in only_tables}
            tables = [t for t in all_tables if t.lower() in lower_allow]
        else:
            tables = all_tables

        schema: List[Dict[str, Any]] = []
        for t in tables:
            cols = get_table_columns(conn, t)
            fks = get_foreign_keys(conn, t)
            schema.append(
                {
                    "table": t,
                    "columns": cols,
                    "foreign_keys": fks,
                }
            )
        return schema
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract SQLite schema and write to data/schema.json"
    )
    p.add_argument(
        "db_path",
        nargs="?",
        default=None,
        help="Path to SQLite DB file. If omitted, uses environment variable DB_PATH.",
    )
    p.add_argument(
        "--out",
        default="data/schema.json",
        help="Output schema JSON path (default: data/schema.json)",
    )
    p.add_argument(
        "--include-views",
        action="store_true",
        help="Include views in schema (default: false)",
    )
    p.add_argument(
        "--tables",
        default="",
        help="Comma-separated list of tables to include (case-insensitive). Default: all user tables",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    db_path_str = args.db_path or os.environ.get("DB_PATH")
    if not db_path_str:
        raise SystemExit("Provide DB path as argument or set DB_PATH environment variable.")

    db_path = Path(db_path_str)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    only_tables = [t.strip() for t in args.tables.split(",") if t.strip()] or None

    schema = extract_schema(
        db_path=db_path,
        include_views=args.include_views,
        only_tables=only_tables,
    )

    out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

    table_names = [s["table"] for s in schema]
    print(f"Wrote schema to: {out_path.resolve()}")
    print(f"Tables found ({len(table_names)}): {', '.join(table_names) if table_names else '(none)'}")


if __name__ == "__main__":
    main()
