from __future__ import annotations

import pandas as pd

from .config import INTERIM, PROCESSED, WARSH_START


def validate_documents(path=INTERIM / "documents.csv") -> None:
    docs = pd.read_csv(path)
    required = {
        "document_id", "document_type", "document_group", "chair",
        "release_date", "release_datetime_et", "source_url", "word_count",
    }
    missing = required - set(docs.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert docs.document_id.is_unique, "Duplicate document IDs"
    assert (docs.word_count > 50).all(), "One or more documents contain almost no text"
    dates = pd.to_datetime(docs.release_date)
    expected_chair = dates.ge(pd.Timestamp(WARSH_START)).map({True: "Warsh", False: "Powell"})
    assert docs.chair.reset_index(drop=True).equals(expected_chair.reset_index(drop=True)), "Chair/date mismatch"
    assert set(docs.document_group) <= {"statement", "minutes", "chair_communication"}
    print(f"Validated {len(docs)} documents")
    print(pd.crosstab(docs.document_group, docs.chair, margins=True))


def validate_events(path=PROCESSED / "event_sample.csv") -> None:
    events = pd.read_csv(path)
    outcomes = [
        "dxy_change_pct", "spread_10s2s_change_bp", "dgs1_change_bp",
        "growth_minus_value_pct", "dgs3mo_change_bp",
    ]
    assert events.document_id.is_unique
    assert pd.to_datetime(events.event_day).ge(pd.to_datetime(events.prior_trading_day)).all()
    print("Missing event outcomes:")
    print(events[outcomes].isna().sum())


if __name__ == "__main__":
    validate_documents()
    if (PROCESSED / "event_sample.csv").exists():
        validate_events()

