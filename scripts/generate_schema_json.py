import json, sqlite3, sys
from pathlib import Path

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def list_tables(conn):
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    names = [r[0] for r in rows]
    return [n for n in names if n != "sqlite_sequence"]

def get_columns(conn, table: str):
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA table_info({qident(table)})").fetchall()
    cols = []
    for cid, name, coltype, notnull, dflt_value, pk in rows:
        cols.append({
            "name": name,
            "type": coltype,
            "notnull": bool(notnull),
            "default": dflt_value,
            "pk": bool(pk),
        })
    return cols

def get_foreign_keys(conn, table: str):
    cur = conn.cursor()
    rows = cur.execute(f"PRAGMA foreign_key_list({qident(table)})").fetchall()
    fks = []
    for (_id, _seq, ref_table, frm, to, on_update, on_delete, match) in rows:
        fks.append({
            "from": frm,
            "to_table": ref_table,
            "to_col": to,
            "on_update": on_update,
            "on_delete": on_delete,
            "match": match,
        })
    return fks

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_schema_json.py <path-to-sqlite-db>")
        raise SystemExit(1)

    db_path = Path(sys.argv[1])
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        raise SystemExit(1)

    out_path = Path("data/schema.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        tables = list_tables(conn)
        schema = []
        for t in tables:
            schema.append({
                "table": t,
                "columns": get_columns(conn, t),
                "foreign_keys": get_foreign_keys(conn, t),
            })
    finally:
        conn.close()

    out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote schema to: {out_path.resolve()}")
    print(f"Tables found ({len(schema)}): {', '.join([s['table'] for s in schema])}")
if __name__ == "__main__":
    main()
