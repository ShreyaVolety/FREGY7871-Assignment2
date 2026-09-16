from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"

START_DATE = "2018-02-05"
WARSH_START = "2026-05-22"
END_DATE = "2026-09-14"


def ensure_directories() -> None:
    for path in (RAW, INTERIM, PROCESSED, TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)

