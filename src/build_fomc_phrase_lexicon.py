"""Build and apply a professor-style FOMC phrase lexicon.

The lexicon is constructed from keyword signatures observed in pre-Warsh
Chair press-conference introductory remarks. Keywords in a signature must all
occur in the same sentence, but need not be consecutive. Each signature has a
topic and a policy-direction label: H=+1, D=-1, N=0.

Usage
-----
python build_fomc_phrase_lexicon.py \
    --corpus-dir path/to/documents \
    --output-dir outputs/wordlist
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


CUTOFF_DEFAULT = "2026-05-22"

# Curated after reviewing the pre-Warsh introductory remarks in the supplied
# corpus. These are contextual keyword sets rather than exact quoted phrases.
SIGNATURES = [
    # Interest rates: hawkish/tighter
    ("Interest Rate", "H", "committee, raised, target range", "Policy action"),
    ("Interest Rate", "H", "ongoing increases, target range, appropriate", "Forward guidance"),
    ("Interest Rate", "H", "restrictive, policy stance, inflation", "Policy stance"),
    ("Interest Rate", "H", "maintaining, restrictive, policy stance", "Policy stance"),
    ("Interest Rate", "H", "policy rate, raised", "Policy action"),
    ("Interest Rate", "H", "additional policy firming, appropriate", "Forward guidance"),

    # Interest rates: dovish/easier
    ("Interest Rate", "D", "committee, lower, policy interest rate", "Policy action"),
    ("Interest Rate", "D", "reduced, target range, policy interest rate", "Policy action"),
    ("Interest Rate", "D", "lower path, federal funds rate", "Forward guidance"),
    ("Interest Rate", "D", "accommodative stance, monetary policy", "Policy stance"),
    ("Interest Rate", "D", "dialing back, policy restraint", "Policy stance"),
    ("Interest Rate", "D", "lowered, policy rate", "Policy action"),
    ("Interest Rate", "D", "cut, federal funds rate, appropriate", "Forward guidance"),

    # Interest rates: neutral/hold
    ("Interest Rate", "N", "committee, maintain, target range", "Policy hold"),
    ("Interest Rate", "N", "leave, policy rate, unchanged", "Policy hold"),
    ("Interest Rate", "N", "interest rate, unchanged", "Policy hold"),
    ("Interest Rate", "N", "decisions, meeting by meeting", "Data dependence"),
    ("Interest Rate", "N", "assess, incoming data, balance of risks", "Data dependence"),

    # Economy/inflation: hawkish
    ("Economy", "H", "inflation, much too high", "Inflation"),
    ("Economy", "H", "inflation, elevated, 2 percent", "Inflation"),
    ("Economy", "H", "inflation, above, longer-run goal", "Inflation"),
    ("Economy", "H", "inflation pressures, run high", "Inflation"),
    ("Economy", "H", "higher energy prices, overall inflation", "Inflation"),
    ("Economy", "H", "strong economy, benefiting", "Growth"),
    ("Economy", "H", "economy, robust, expected", "Growth"),
    ("Economy", "H", "demand, very strong, supply, subdued", "Demand and supply"),
    ("Economy", "H", "inflation, more persistent, risk", "Inflation risk"),
    ("Economy", "H", "lack of further progress, inflation", "Inflation progress"),

    # Economy/inflation: dovish
    ("Economy", "D", "inflation, eased notably", "Inflation"),
    ("Economy", "D", "inflation, moderated somewhat", "Inflation"),
    ("Economy", "D", "inflation, declined, past year", "Inflation"),
    ("Economy", "D", "inflation, below, 2 percent", "Inflation"),
    ("Economy", "D", "weaker demand, consumer prices", "Demand and inflation"),
    ("Economy", "D", "economic activity, slowed substantially", "Growth"),
    ("Economy", "D", "economy, slowed significantly", "Growth"),
    ("Economy", "D", "business investment, exports, weak", "Investment"),
    ("Economy", "D", "consumer spending, slowed, disposable income", "Consumption"),
    ("Economy", "D", "disinflation, progress", "Inflation progress"),

    # Economy: neutral/balanced
    ("Economy", "N", "risks, employment, inflation, roughly in balance", "Balance of risks"),
    ("Economy", "N", "economic activity, moderate pace", "Growth"),
    ("Economy", "N", "inflation, near, 2 percent", "Inflation"),

    # Job market: hawkish/strong
    ("Job Market", "H", "labor market, extremely tight", "Labor-market tightness"),
    ("Job Market", "H", "labor demand, very strong", "Labor demand"),
    ("Job Market", "H", "job gains, robust", "Job growth"),
    ("Job Market", "H", "unemployment rate, 50-year low", "Unemployment"),
    ("Job Market", "H", "job vacancies, historical highs", "Vacancies"),
    ("Job Market", "H", "strong labor market", "Labor-market strength"),
    ("Job Market", "H", "jobs market, strong, wage gains", "Labor-market strength"),
    ("Job Market", "H", "employment, continued to strengthen", "Employment"),
    ("Job Market", "H", "labor demand, exceeds, supply", "Labor demand"),

    # Job market: dovish/weaker
    ("Job Market", "D", "labor market, cooled, overheated", "Labor-market cooling"),
    ("Job Market", "D", "job gains, slowed", "Job growth"),
    ("Job Market", "D", "labor market tightness, eased", "Labor-market cooling"),
    ("Job Market", "D", "wage growth, eased", "Wages"),
    ("Job Market", "D", "job vacancies, declined", "Vacancies"),
    ("Job Market", "D", "unemployment rate, edged up", "Unemployment"),
    ("Job Market", "D", "downside risks, employment, risen", "Employment risk"),
    ("Job Market", "D", "labor demand, softened", "Labor demand"),

    # Job market: neutral/balanced
    ("Job Market", "N", "labor market, broadly in balance", "Labor-market balance"),
    ("Job Market", "N", "job gains, low, unemployment rate, little changed", "Labor-market balance"),

    # Sentiment/confidence: hawkish
    ("Sentiment", "H", "inflation, still too high, progress, not assured", "Inflation concern"),
    ("Sentiment", "H", "committed, restrictive, bring inflation down", "Policy commitment"),
    ("Sentiment", "H", "upside risks, inflation", "Risk assessment"),
    ("Sentiment", "H", "not gained, confidence, inflation", "Policy confidence"),
    ("Sentiment", "H", "uncertainty, elevated", "Uncertainty"),

    # Sentiment/confidence: dovish
    ("Sentiment", "D", "gained greater confidence, inflation", "Policy confidence"),
    ("Sentiment", "D", "risks, moved toward better balance", "Risk assessment"),
    ("Sentiment", "D", "downside risks, employment", "Employment risk"),
    ("Sentiment", "D", "inflation expectations, declined", "Inflation expectations"),

    # Sentiment: neutral/data-dependent
    ("Sentiment", "N", "outlook, balance of risks, meeting by meeting", "Data dependence"),
    ("Sentiment", "N", "monitor, risks, both sides", "Risk assessment"),
    ("Sentiment", "N", "carefully assess, incoming data", "Data dependence"),
]


DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"\d{1,2},\s+\d{4}\b"
)

END_MARKERS = [
    "let's go to questions", "let us go to questions", "begin with questions",
    "begin our questions", "start with questions", "questions now",
    "happy to take your questions", "take your questions now",
    "turn to your questions", "look forward to taking your questions",
    "look forward to your questions", "look forward to answering your questions",
]


def normalize(text: str) -> str:
    text = text.lower().replace("\u00ad", "").replace("\u2011", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9%'-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u2011", "-")
    text = re.sub(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"\d{1,2},\s+\d{4}\s+(?:Chair|Chairman)\s+\w+['’]s Press Conference\s+FINAL\s+Page\s+\d+\s+of\s+\d+",
        " ", text, flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|U\.S|F\.O\.M\.C)\.",
        lambda m: m.group(0).replace(".", "<DOT>"), text,
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if len(p.split()) >= 5]


def extract_date(text: str) -> str:
    match = DATE_RE.search(text[:700])
    return datetime.strptime(match.group(0), "%B %d, %Y").date().isoformat() if match else ""


def extract_intro(text: str) -> str | None:
    if not re.search(r"Transcript of (?:Chair|Chairman).*?Press Conference", text, re.I):
        return None
    start = re.search(r"CHAIR(?:MAN)?\s+[A-Z]+\.\s+", text)
    if not start:
        return None
    intro = text[start.end():]
    lowered = intro.lower()
    stops = [lowered.find(marker) for marker in END_MARKERS if lowered.find(marker) >= 0]
    # The first non-Chair all-caps speaker normally marks the Q&A moderator or
    # first reporter. This catches transcripts whose transition wording varies.
    speaker = re.search(r"\s([A-Z][A-Z'-]+(?:\s+[A-Z][A-Z'-]+){1,3})\.\s+[A-Z][a-z]", intro)
    if speaker and not speaker.group(1).startswith(("CHAIR POWELL", "CHAIRMAN POWELL")):
        stops.append(speaker.start())
    if stops:
        intro = intro[:min(stops)]
    return clean_text(intro)


def keyword_list(value: str) -> list[str]:
    return [normalize(piece) for piece in value.split(",") if normalize(piece)]


def signature_matches(sentence: str, keywords: list[str]) -> bool:
    normalized = normalize(sentence)
    return all(re.search(rf"\b{re.escape(keyword)}\b", normalized) for keyword in keywords)


def document_type(text: str) -> str:
    lead = text[:800].lower()
    if re.search(r"transcript of (?:chair|chairman).*press conference", lead):
        return "press_conference"
    if "minutes of the federal open market committee" in lead:
        return "minutes"
    if "issues fomc statement" in lead or "approved the following statement" in lead:
        return "statement"
    if "testimony" in lead or "monetary policy report to the congress" in lead:
        return "testimony"
    return "speech_or_other"


def build_lexicon(corpus: Path, cutoff: str) -> list[dict]:
    cutoff_date = datetime.strptime(cutoff, "%Y-%m-%d").date()
    source_sentences = []
    for path in sorted(corpus.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        date = extract_date(raw)
        if not date or datetime.strptime(date, "%Y-%m-%d").date() >= cutoff_date:
            continue
        intro = extract_intro(raw)
        if intro:
            source_sentences.extend((date, path.name, s) for s in split_sentences(intro))

    output = []
    for topic, direction, signature, subtopic in SIGNATURES:
        keys = keyword_list(signature)
        matches = [(d, f, s) for d, f, s in source_sentences if signature_matches(s, keys)]
        if not matches:
            continue
        date, source_file, sentence = matches[0]
        output.append({
            "topic": topic,
            "direction": direction,
            "sentiment": {"H": 1, "D": -1, "N": 0}[direction],
            "keywords": signature,
            "subtopic": subtopic,
            "source_date": date,
            "source_file": source_file,
            "source_sentence": sentence,
            "prewarsh_match_count": len(matches),
        })
    return output


def prepare_matchers(lexicon: list[dict]) -> list[tuple[dict, list[re.Pattern]]]:
    return [
        (
            row,
            [re.compile(rf"\b{re.escape(keyword)}\b") for keyword in keyword_list(row["keywords"])],
        )
        for row in lexicon
    ]


def score_document(text: str, matchers: list[tuple[dict, list[re.Pattern]]]) -> dict:
    sentences = split_sentences(clean_text(text))
    topic_totals = defaultdict(float)
    topic_match_counts = defaultdict(int)
    audit = []
    for sentence_number, sentence in enumerate(sentences, start=1):
        normalized_sentence = normalize(sentence)
        sentence_contributions = defaultdict(float)
        for row, patterns in matchers:
            if all(pattern.search(normalized_sentence) for pattern in patterns):
                keys = keyword_list(row["keywords"])
                contribution = len(keys) * int(row["sentiment"])
                sentence_contributions[row["topic"]] += contribution
                topic_match_counts[row["topic"]] += 1
                audit.append(
                    f'{sentence_number}:{row["topic"]}:{row["direction"]}:{row["keywords"]}'
                )
        for topic, value in sentence_contributions.items():
            topic_totals[topic] += 1 if value > 0 else -1 if value < 0 else 0

    denominator = max(len(sentences), 1)
    result = {"n_sentences": len(sentences), "match_audit": " | ".join(audit)}
    for topic in ["Interest Rate", "Economy", "Job Market", "Sentiment"]:
        slug = topic.lower().replace(" ", "_")
        result[f"wordlist_{slug}_score"] = topic_totals[topic] / denominator
        result[f"wordlist_{slug}_matches"] = topic_match_counts[topic]
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default=CUTOFF_DEFAULT)
    args = parser.parse_args()

    lexicon = build_lexicon(args.corpus_dir, args.cutoff)
    lexicon_path = args.output_dir / "professor_style_phrase_lexicon.csv"
    write_csv(lexicon_path, lexicon)
    matchers = prepare_matchers(lexicon)

    scores = []
    for path in sorted(args.corpus_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        scores.append({
            "document_id": path.stem,
            "release_date": extract_date(raw),
            "document_type": document_type(raw),
            **score_document(raw, matchers),
        })
    scores_path = args.output_dir / "professor_style_document_scores.csv"
    write_csv(scores_path, scores)

    print(f"Validated lexicon entries: {len(lexicon)}")
    print(f"Documents scored: {len(scores)}")
    print(f"Lexicon: {lexicon_path}")
    print(f"Scores: {scores_path}")


if __name__ == "__main__":
    main()
