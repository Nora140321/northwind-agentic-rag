from pathlib import Path
from ask_sql_gemini import ask_sql_gemini

DB = Path("data/1751978414.sqlite3")
VS = Path("data/vectorstore")

tests = [
    ("Top 5 products by total quantity ordered", ["ProductName","TotalQuantityOrdered"]),
    ("Top 10 customers by total revenue", ["CustomerID","TotalRevenue"]),
]

fail = 0
for q, cols in tests:
    sql, headers, rows = ask_sql_gemini(DB, VS, q)
    missing = [c for c in cols if c not in headers]
    if missing:
        fail += 1
        print(f"[FAIL] {q}\nSQL: {sql}\nMissing: {missing}\nHeaders: {headers}\n")
    else:
        print(f"[OK] {q}")

if fail:
    raise SystemExit(1)
