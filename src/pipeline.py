from __future__ import annotations

import argparse

from .collect_fed_documents import collect as collect_documents
from .collect_market_data import collect as collect_market, construct_events
from .run_analysis import run as run_analysis
from .score_finbert import run as score_finbert
from .score_lexicon import run as score_lexicon

STAGES = ["documents", "lexicon", "finbert", "market", "analysis"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=STAGES, default="analysis")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    stop = STAGES.index(args.through)
    if stop >= 0:
        collect_documents()
    if stop >= 1:
        score_lexicon()
    if stop >= 2:
        score_finbert(args.batch_size)
    if stop >= 3:
        collect_market()
        construct_events()
    if stop >= 4:
        run_analysis()


if __name__ == "__main__":
    main()

