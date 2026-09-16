from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import INTERIM, ROOT, ensure_directories
from .utils import save_csv

MODEL_NAME = "ProsusAI/finbert"
FACTOR_SENTENCES = {
    "factor_inflation_score": "Inflation will rise",
    "factor_interest_rate_score": "Interest rates will rise",
}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text).strip())
    return [part for part in parts if 5 <= len(part.split()) <= 150]


class FinBERTScorer:
    """Presentation-style factor similarity and FinBERT sentiment."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()
        labels = {str(label).lower(): int(i) for i, label in self.model.config.id2label.items()}
        self.positive_index = labels.get("positive")
        if self.positive_index is None:
            raise ValueError(f"Positive label not found in model labels: {labels}")
        self.factor_embeddings = self._encode(list(FACTOR_SENTENCES.values()), 2)[0]

    def _encode(self, sentences: list[str], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        embeddings, probabilities = [], []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                output = self.model(**tokens, output_hidden_states=True, return_dict=True)
                hidden = output.hidden_states[-1]
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                probs = torch.softmax(output.logits, dim=1)
            embeddings.append(pooled.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
        return np.vstack(embeddings), np.vstack(probabilities)

    def score(self, text: str, batch_size: int = 32) -> dict[str, float]:
        sentences = split_sentences(text)
        if not sentences:
            return {
                **{name: np.nan for name in FACTOR_SENTENCES},
                "finbert_score": np.nan,
                "finbert_segment_1": np.nan,
                "finbert_segment_2": np.nan,
                "finbert_segment_3": np.nan,
                "finbert_weighted_score": np.nan,
                "finbert_sentences": 0,
            }
        embeddings, probabilities = self._encode(sentences, batch_size)
        positive = probabilities[:, self.positive_index]
        factor_scores = embeddings @ self.factor_embeddings.T
        segments = np.array_split(positive, 3)
        segment_means = [float(x.mean()) if len(x) else np.nan for x in segments]
        weights = np.array([len(x) for x in segments], dtype=float)
        return {
            "factor_inflation_score": float(factor_scores[:, 0].mean()),
            "factor_interest_rate_score": float(factor_scores[:, 1].mean()),
            "finbert_score": float(positive.mean()),
            "finbert_segment_1": segment_means[0],
            "finbert_segment_2": segment_means[1],
            "finbert_segment_3": segment_means[2],
            "finbert_weighted_score": float(np.average(segment_means, weights=weights)),
            "finbert_sentences": len(sentences),
        }


def run(batch_size: int = 32) -> pd.DataFrame:
    ensure_directories()
    path = INTERIM / "documents_lexicon.csv"
    docs = pd.read_csv(path if path.exists() else INTERIM / "documents.csv")
    scorer = FinBERTScorer()
    scores = []
    for number, row in enumerate(docs.itertuples(index=False), 1):
        text = (ROOT / row.text_path).read_text(encoding="utf-8")
        scores.append({"document_id": row.document_id, **scorer.score(text, batch_size)})
        print(f"[{number}/{len(docs)}] {row.document_id}")
    result = docs.merge(pd.DataFrame(scores), on="document_id", validate="one_to_one")
    save_csv(result, INTERIM / "documents_scored.csv")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.batch_size)
