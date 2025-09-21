from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv, find_dotenv


def load_project_dotenv(
    root_levels_up: int = 2,
    filename: str = ".env",
    *,
    override: bool = True,
    tolerate_bom: bool = True,
) -> Tuple[Path, bool]:
    """
    Load a .env located at the project root (N levels up from this file),
    with optional BOM-tolerant parsing (utf-8-sig).

    Returns:
        (dotenv_path, loaded)
    """
    here = Path(__file__).resolve()
    root = here.parents[root_levels_up]
    dotenv_path = root / filename

    encoding = "utf-8-sig" if tolerate_bom else None

    # Try explicit path first
    loaded = load_dotenv(dotenv_path, override=override, encoding=encoding)

    # If not found/loaded, fall back to discovery from CWD (useful in notebooks)
    if not loaded:
        discovered = find_dotenv(usecwd=True)
        if discovered:
            loaded = load_dotenv(discovered, override=override, encoding=encoding)
            return Path(discovered), loaded

    return dotenv_path, loaded