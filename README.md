# Fed Communication Tone and Asset Prices

Reproducible code for FRE-GY 7871 Assignment 2. The project collects Federal
Reserve communications from February 2018 onward, estimates hawkish/dovish
tone using a monetary-policy phrase list and FinBERT sentence embeddings, and
relates tone to one-day market changes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set a descriptive `FED_USER_AGENT`. A FRED API key is optional because the
collector can use FRED's CSV endpoint.

## Run

```bash
python -m src.pipeline --through analysis
```

Stages may also be run separately:

```bash
python -m src.collect_fed_documents
python -m src.score_lexicon
python -m src.score_finbert
python -m src.collect_market_data
python -m src.run_analysis
```

Open `notebooks/assignment_2_analysis.ipynb` after running the pipeline. Commit
the notebook with saved output, but do not commit anything under `data/`.

## Definitions

- Powell era: 2018-02-05 through 2026-05-21.
- Warsh era: 2026-05-22 onward.
- Intraday release: previous trading close to that day's close.
- After-close/weekend release: previous close to the next trading close.
- Yield and spread changes are in basis points. ETF and DXY changes are percent.

The event-time assumptions and any imputed release times are retained in the
processed event file for auditing.

