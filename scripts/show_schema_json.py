import json, sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/schema.json")
if not p.exists():
    print(f"Not found: {p}")
    raise SystemExit(1)

data = json.loads(p.read_text(encoding="utf-8"))
print("Count:", len(data))
print("Tables:", [d.get("table") for d in data])
