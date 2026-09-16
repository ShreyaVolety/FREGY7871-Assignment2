from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import INTERIM, ROOT, ensure_directories
from .tone_dictionary import DOVISH_PHRASES, HAWKISH_PHRASES
from .utils import normalize_space, save_csv


def phrase_count(text: str, phrase: str) -> int:
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)"
    return len(re.findall(pattern, text, flags=re.I))


def score_text(text: str) -> dict[str, float]:
    clean = normalize_space(text).lower()
    hawk_raw = sum(weight * phrase_count(clean, phrase) for phrase, weight in HAWKISH_PHRASES.items())
    dove_raw = sum(weight * phrase_count(clean, phrase) for phrase, weight in DOVISH_PHRASES.items())
    words = max(len(re.findall(r"\b[a-z][a-z'-]*\b", clean)), 1)
    scale = 1_000 / words
    hawk = hawk_raw * scale
    dove = dove_raw * scale
    return {
        "lexicon_hawk_per_1k": hawk,
        "lexicon_dove_per_1k": dove,
        "lexicon_score": hawk - dove,
        "lexicon_mentions": hawk_raw + dove_raw,
    }


def run() -> pd.DataFrame:
    ensure_directories()
    docs = pd.read_csv(INTERIM / "documents.csv")
    scores = []
    for row in docs.itertuples(index=False):
        text = (ROOT / row.text_path).read_text(encoding="utf-8")
        scores.append({"document_id": row.document_id, **score_text(text)})
    result = docs.merge(pd.DataFrame(scores), on="document_id", validate="one_to_one")
    save_csv(result, INTERIM / "documents_lexicon.csv")
    return result


if __name__ == "__main__":
    run()

