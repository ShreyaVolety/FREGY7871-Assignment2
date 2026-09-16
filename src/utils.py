from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def stable_id(*parts: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def chair_for_date(value: str | pd.Timestamp) -> str:
    date = pd.Timestamp(value).tz_localize(None).normalize()
    return "Warsh" if date >= pd.Timestamp("2026-05-22") else "Powell"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

