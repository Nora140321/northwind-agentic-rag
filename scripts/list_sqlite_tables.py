import sys
import sqlite3
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/list_sqlite_tables.py <path-to-sqlite-db>")
        raise SystemExit(1)

    db_path = Path(sys.argv[1])
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        raise SystemExit(1)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    ).fetchall()
    print([r[0] for r in rows])
    conn.close()

if __name__ == "__main__":
    main()
