# src/agents/sql_executor.py
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SQLExecutor:
    """
    Safe SQLite executor with:
    - Select-only guard (by default)
    - Busy timeout
    - Dict rows
    - Simple table formatting

    Quality-of-life tweak:
    - When adding a LIMIT and the SQL starts with a CTE (WITH ...), do NOT wrap
      the whole query. Instead, append LIMIT to the final SELECT. For non-CTE
      queries, keep wrapping with SELECT * FROM (...) LIMIT N to preserve ORDER BY.
    """

    def __init__(self, db_path: Path, strict_select: bool = True, timeout_ms: int = 30000):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite DB not found: {self.db_path}")
        self.strict_select = strict_select
        self.timeout_ms = timeout_ms

    @staticmethod
    def _is_select_only(sql: str) -> bool:
        """
        Very conservative guard: no semicolons (to block multi-statements),
        and the statement must be a SELECT or a CTE leading to a SELECT.
        """
        s = sql.strip().strip(";").lstrip("(")
        if ";" in sql:
            return False
        # allow CTEs: WITH ... SELECT ...
        if re.match(r"(?is)^\s*(with\s+.+)?\s*select\s", s):
            return True
        return False

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute(f"PRAGMA busy_timeout = {self.timeout_ms};")
        return con

    def execute(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        *,
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute SQL safely and return (rows_as_dicts, column_names).

        - Tolerates a trailing semicolon by stripping it.
        - If a LIMIT is provided and the query doesn't already contain LIMIT:
          - For CTE queries (starting with WITH): append "LIMIT N" at the end.
          - Otherwise, wrap as "SELECT * FROM ( ... ) AS _sub LIMIT N" to preserve ORDER BY.
        """
        # Normalize and guard
        sql = sql.strip().rstrip(";")  # tolerate trailing semicolons
        if self.strict_select and not self._is_select_only(sql):
            raise ValueError("Only SELECT queries are allowed in strict mode (no semicolons, single statement).")

        # Decide how to apply LIMIT, if requested and not already present
        if limit is not None:
            # Conservative LIMIT detection (any LIMIT in the statement)
            has_limit = re.search(r"(?is)\blimit\b", sql) is not None
            if not has_limit:
                lower = sql.lstrip().lower()
                starts_with_cte = lower.startswith("with ")
                if starts_with_cte:
                    # For CTEs, append LIMIT to the final SELECT.
                    # We rely on well-formed SQL where appending LIMIT at the end is valid.
                    sql = f"{sql}\nLIMIT {int(limit)}"
                else:
                    # For non-CTE, wrap to preserve ORDER BY semantics.
                    sql = f"SELECT * FROM ({sql}) AS _sub LIMIT {int(limit)}"

        if verbose:
            print("---- SQL to execute ----")
            print(sql)
            print("------------------------")

        with self._connect() as con:
            cur = con.cursor()
            cur.execute(sql, tuple(params or []))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            data = [dict(zip(cols, r)) for r in rows]

        if verbose:
            self._print_table_preview(data, cols)

        return data, cols

    @staticmethod
    def _print_table_preview(rows: List[Dict[str, Any]], cols: List[str], max_rows: int = 20) -> None:
        if not cols:
            print("(No columns returned)")
            return

        sample = rows[:max_rows]
        # Robust width calculation (avoid star-unpack edge cases)
        col_widths: Dict[str, int] = {}
        for c in cols:
            lengths = [len(str(c))]
            for r in sample:
                try:
                    lengths.append(len(str(r.get(c, ""))))
                except Exception:
                    lengths.append(len(str(r)))
            col_widths[c] = max(lengths) if lengths else len(str(c))

        header = " | ".join(str(c).ljust(col_widths[c]) for c in cols)
        sep = "-+-".join("-" * col_widths[c] for c in cols)
        print(header)
        print(sep)
        for r in sample:
            print(" | ".join(str(r.get(c, "")).ljust(col_widths[c]) for c in cols))
        if len(rows) > max_rows:
            print(f"... ({len(rows) - max_rows} more rows)")
        print(f"Rows: {len(rows)}")