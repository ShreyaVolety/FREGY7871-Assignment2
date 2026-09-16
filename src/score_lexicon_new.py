"""Score FOMC documents using the professor-style phrase lexicon.

For each sentence i and topic k:

    sentence_score_ik = sign(sum_p L(p) * S(p) * match(p, i))

The document score is the average sentence score:

    x_k(t) = (1 / n_t) * sum_i sentence_score_ik

Here L(p) is the number of words in signature p, S(p) is +1 (hawkish),
-1 (dovish), or 0 (neutral), and match(p, i) is one when every comma-
separated keyword/chunk in p occurs in sentence i. The chunks need not be
consecutive or adjacent to one another.

Examples
--------
Directory of TXT documents:

    python score_lexicon.py \
        --input documents \
        --lexicon professor_style_phrase_lexicon.csv \
        --output document_lexicon_scores.csv

CSV containing document text:

    python score_lexicon.py \
        --input event_sample.csv \
        --text-column text \
        --id-column document_id \
        --lexicon professor_style_phrase_lexicon.csv \
        --output event_sample_with_lexicon_scores.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TOPICS = ("Interest Rate", "Economy", "Job Market", "Sentiment")
DIRECTION_TO_SCORE = {"H": 1, "D": -1, "N": 0}


def normalize(text: object) -> str:
    """Normalize text while preserving words inside multiword chunks."""
    value = str(text).lower()
    value = value.replace("\u00ad", "").replace("\u2011", "-")
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = re.sub(r"[^a-z0-9%'-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def split_sentences(text: object) -> list[str]:
    """Split extracted Fed text without requiring an NLTK download."""
    value = str(text).replace("\u00ad", "")
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        return []

    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|U\.S|F\.O\.M\.C)\.",
        lambda match: match.group(0).replace(".", "<DOT>"),
        value,
    )
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
    return [
        sentence.replace("<DOT>", ".").strip()
        for sentence in sentences
        if sentence.strip()
    ]


def topic_slug(topic: str) -> str:
    return normalize(topic).replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class LexiconEntry:
    topic: str
    direction: str
    sentiment: int
    keywords: str
    chunks: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]
    length: int

    def matches(self, normalized_sentence: str) -> bool:
        return all(pattern.search(normalized_sentence) for pattern in self.patterns)


def load_lexicon(path: Path) -> list[LexiconEntry]:
    frame = pd.read_csv(path)
    required = {"topic", "direction", "sentiment", "keywords"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Lexicon is missing columns: {sorted(missing)}")

    entries: list[LexiconEntry] = []
    for row in frame.itertuples(index=False):
        topic = str(row.topic).strip()
        direction = str(row.direction).strip().upper()
        sentiment = int(row.sentiment)

        if topic not in TOPICS:
            raise ValueError(f"Unexpected topic: {topic!r}")
        if direction not in DIRECTION_TO_SCORE:
            raise ValueError(f"Unexpected direction: {direction!r}")
        if sentiment != DIRECTION_TO_SCORE[direction]:
            raise ValueError(
                f"Direction/sentiment mismatch for {row.keywords!r}: "
                f"{direction} versus {sentiment}"
            )

        chunks = tuple(
            chunk
            for piece in str(row.keywords).split(",")
            if (chunk := normalize(piece))
        )
        if not chunks:
            raise ValueError("A lexicon entry has no usable keywords")

        patterns = tuple(
            re.compile(rf"\b{re.escape(chunk)}\b")
            for chunk in chunks
        )
        length = sum(len(chunk.split()) for chunk in chunks)

        entries.append(
            LexiconEntry(
                topic=topic,
                direction=direction,
                sentiment=sentiment,
                keywords=str(row.keywords),
                chunks=chunks,
                patterns=patterns,
                length=length,
            )
        )

    duplicates = frame.duplicated(["topic", "keywords"], keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, ["topic", "keywords"]].to_dict("records")
        raise ValueError(f"Duplicate lexicon entries: {values}")

    return entries


def sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def score_document(text: object, lexicon: list[LexiconEntry]) -> dict[str, object]:
    """Return the four category scores and transparent match diagnostics."""
    sentences = split_sentences(text)
    n_sentences = len(sentences)

    topic_sentence_sum = {topic: 0 for topic in TOPICS}
    topic_matched_sentences = {topic: 0 for topic in TOPICS}
    topic_signature_matches = {topic: 0 for topic in TOPICS}
    hawkish_matches = {topic: 0 for topic in TOPICS}
    dovish_matches = {topic: 0 for topic in TOPICS}
    neutral_matches = {topic: 0 for topic in TOPICS}

    for sentence in sentences:
        normalized_sentence = normalize(sentence)
        weighted_sum = {topic: 0.0 for topic in TOPICS}
        matched_topics: set[str] = set()

        for entry in lexicon:
            if not entry.matches(normalized_sentence):
                continue

            # Slide formula: phrase length times manually assigned direction.
            weighted_sum[entry.topic] += entry.length * entry.sentiment
            topic_signature_matches[entry.topic] += 1
            matched_topics.add(entry.topic)

            if entry.direction == "H":
                hawkish_matches[entry.topic] += 1
            elif entry.direction == "D":
                dovish_matches[entry.topic] += 1
            else:
                neutral_matches[entry.topic] += 1

        for topic in TOPICS:
            topic_sentence_sum[topic] += sign(weighted_sum[topic])
            if topic in matched_topics:
                topic_matched_sentences[topic] += 1

    result: dict[str, object] = {"lexicon_n_sentences": n_sentences}

    for topic in TOPICS:
        slug = topic_slug(topic)

        # The slide values are small because all document sentences form the
        # denominator; sentences without a match contribute zero.
        result[f"lexicon_{slug}_score"] = (
            topic_sentence_sum[topic] / n_sentences
            if n_sentences
            else np.nan
        )
        result[f"lexicon_{slug}_matched_sentences"] = topic_matched_sentences[topic]
        result[f"lexicon_{slug}_signature_matches"] = topic_signature_matches[topic]
        result[f"lexicon_{slug}_hawkish_matches"] = hawkish_matches[topic]
        result[f"lexicon_{slug}_dovish_matches"] = dovish_matches[topic]
        result[f"lexicon_{slug}_neutral_matches"] = neutral_matches[topic]

    return result


def load_documents(
    input_path: Path,
    text_column: str,
    id_column: str,
) -> pd.DataFrame:
    if input_path.is_dir():
        rows = []
        for path in sorted(input_path.glob("*.txt")):
            rows.append(
                {
                    id_column: path.stem,
                    text_column: path.read_text(encoding="utf-8", errors="replace"),
                }
            )
        if not rows:
            raise ValueError(f"No TXT files found in {input_path}")
        return pd.DataFrame(rows)

    if input_path.suffix.lower() == ".csv":
        frame = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(input_path)
    else:
        raise ValueError("Input must be a directory of TXT files, CSV, or Parquet")

    missing = {text_column, id_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="document_id")
    args = parser.parse_args()

    lexicon = load_lexicon(args.lexicon)
    documents = load_documents(args.input, args.text_column, args.id_column)

    score_rows = [
        score_document(text, lexicon)
        for text in documents[args.text_column].fillna("")
    ]
    scores = pd.DataFrame(score_rows, index=documents.index)
    output = pd.concat([documents, scores], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in {".parquet", ".pq"}:
        output.to_parquet(args.output, index=False)
    else:
        output.to_csv(args.output, index=False)

    score_columns = [
        f"lexicon_{topic_slug(topic)}_score"
        for topic in TOPICS
    ]
    print(f"Documents scored: {len(output)}")
    print(f"Lexicon entries: {len(lexicon)}")
    print(f"Output: {args.output}")
    print(output[score_columns].describe().round(6).to_string())


if __name__ == "__main__":
    main()
