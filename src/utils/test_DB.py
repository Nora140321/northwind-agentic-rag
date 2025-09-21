from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys

def resolve_db_path() -> Path:
    load_dotenv()
    env_path = os.getenv("DATABASE_PATH", "").strip()
    if env_path:
        p = Path(env_path)
    else:
        p = Path(__file__).resolve().parents[2] / "data" / "northwind.db"
    if not p.exists():
        raise FileNotFoundError(f"Database not found at: {p}")
    return p

def sqlite_uri_from_path(p: Path) -> str:
    # Use forward slashes for Windows-safe SQLite URI
    return f"sqlite:///{p.as_posix()}"

def main():
    db_path = resolve_db_path()
    print(f"Using database: {db_path}")

    engine = create_engine(sqlite_uri_from_path(db_path))

    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        ).fetchall()
        print("\nTables found:")
        for (name,) in tables:
            print(f" - {name}")

        try:
            orders_count = conn.execute(text("SELECT COUNT(*) FROM Orders;")).scalar()
            print(f"\nOrders row count: {orders_count}")
        except Exception as e:
            print("\nCouldn't count Orders table. Error:")
            print(e)

        try:
            cols = conn.execute(text("PRAGMA table_info(Orders);")).fetchall()
            col_names = [c[1] for c in cols]
            print(f"Orders columns: {col_names}")
        except Exception as e:
            print("\nCouldn't fetch Orders columns. Error:")
            print(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e, file=sys.stderr)
        sys.exit(1)