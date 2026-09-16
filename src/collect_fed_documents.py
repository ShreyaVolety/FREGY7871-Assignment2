from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader

from .config import END_DATE, INTERIM, RAW, START_DATE, ensure_directories
from .utils import chair_for_date, normalize_space, save_csv, stable_id

BASE = "https://www.federalreserve.gov"
NY_TZ = "America/New_York"
DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d{2}"
)
TIME_RE = re.compile(r"(?:For release at\s+)?(\d{1,2}:\d{2})\s*(a\.m\.|p\.m\.)\s*(EST|EDT|ET)?", re.I)


@dataclass
class Document:
    document_id: str
    document_type: str
    document_group: str
    title: str
    chair: str
    release_date: str
    release_time_et: str
    release_datetime_et: str
    time_imputed: bool
    source_url: str
    text_path: str
    word_count: int


class FedClient:
    def __init__(self, delay: float = 0.2):
        load_dotenv()
        agent = os.getenv("FED_USER_AGENT", "Shreya Volety sv3101@nyu.edu")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": agent})
        self.delay = delay

    def get(self, url: str) -> requests.Response:
        time.sleep(self.delay)
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return response


def page_text_and_title(content: bytes) -> tuple[str, str]:
    soup = BeautifulSoup(content, "lxml")
    og_title = soup.select_one('meta[property="og:title"]')
    heading = soup.select_one("#article h3, main h3, h3")
    title = normalize_space(
        (og_title.get("content", "") if og_title else "")
        or (heading.get_text(" ", strip=True) if heading else "")
    )
    for tag in soup.select("script, style, nav, header, footer, .shareTools, #leftText"):
        tag.decompose()
    article = soup.select_one("#article, .col-xs-12.col-sm-8.col-md-8, main") or soup
    return normalize_space(article.get_text(" ", strip=True)), title


def pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    return normalize_space(" ".join(page.extract_text() or "" for page in reader.pages))


def parse_release_datetime(text: str, fallback_date: str, default_time: str) -> tuple[pd.Timestamp, bool]:
    # The listing/calendar date is authoritative. A minutes page prominently
    # displays the meeting date in its body, which is not the release date.
    release_area = text[:1200]
    time_match = TIME_RE.search(release_area)
    imputed = time_match is None
    if time_match:
        clock = datetime.strptime(
            f"{time_match.group(1)} {time_match.group(2).replace('.', '')}", "%I:%M %p"
        ).strftime("%H:%M")
    else:
        clock = default_time
    stamp = pd.Timestamp(f"{pd.Timestamp(fallback_date).date()} {clock}", tz=NY_TZ)
    return stamp, imputed


def listing_date(node, fallback_url: str) -> str | None:
    container = node.find_parent(["div", "li", "article", "tr"]) or node.parent
    text = normalize_space(container.get_text(" ", strip=True))
    match = DATE_RE.search(text)
    if match:
        return str(pd.Timestamp(match.group(0)).date())
    url_match = re.search(r"(20\d{2})(\d{2})(\d{2})", fallback_url)
    return "-".join(url_match.groups()) if url_match else None


def discover_fomc_links(client: FedClient, years: range) -> list[dict]:
    pages = [f"{BASE}/monetarypolicy/fomccalendars.htm"]
    # 2021 onward is retained directly on the current calendar; dedicated
    # historical pages exist for the earlier years in our sample.
    pages += [f"{BASE}/monetarypolicy/fomchistorical{year}.htm" for year in years if year <= 2020]
    found: dict[str, dict] = {}
    for page in pages:
        try:
            soup = BeautifulSoup(client.get(page).content, "lxml")
        except requests.RequestException as exc:
            print(f"Warning: could not read {page}: {exc}")
            continue
        for link in soup.select("a[href]"):
            href = link.get("href", "")
            href_lower = href.lower()
            label = normalize_space(link.get_text(" ", strip=True)).lower()
            url = urljoin(BASE, href)
            doc_type = None
            if re.search(r"pressreleases/monetary20\d{6}a\.htm", href_lower):
                doc_type = "statement"
            elif "fomcminutes20" in href_lower and href_lower.endswith((".htm", ".html")):
                doc_type = "minutes"
            elif "fomcpresconf20" in href_lower and href_lower.endswith(".pdf"):
                doc_type = "press_conference"
            if doc_type:
                found[url] = {
                    "url": url,
                    "document_type": doc_type,
                    "fallback_date": listing_date(link, url),
                }
    # Press-conference transcript URLs are systematic but are not linked from
    # every calendar/archive page. Generate candidates from statement dates;
    # non-conference dates simply return 404 and are skipped during collection.
    for item in list(found.values()):
        if item["document_type"] != "statement":
            continue
        match = re.search(r"monetary(20\d{6})a\.htm", item["url"], re.I)
        if match:
            transcript_url = f"{BASE}/mediacenter/files/FOMCpresconf{match.group(1)}.pdf"
            found[transcript_url] = {
                "url": transcript_url,
                "document_type": "press_conference",
                "fallback_date": item["fallback_date"],
            }
    return list(found.values())


def discover_chair_links(client: FedClient, years: range) -> list[dict]:
    found: dict[str, dict] = {}
    for year in years:
        for kind, doc_type in (("speech", "speech"), ("testimony", "testimony")):
            url = f"{BASE}/newsevents/{kind}/{year}-{kind}{'es' if kind == 'speech' else ''}.htm"
            try:
                soup = BeautifulSoup(client.get(url).content, "lxml")
            except requests.RequestException as exc:
                print(f"Warning: could not read {url}: {exc}")
                continue
            for link in soup.select("a[href]"):
                href = link.get("href", "")
                if f"/newsevents/{kind}/" not in href or not href.endswith((".htm", ".html")):
                    continue
                container = link.find_parent(["div", "li", "article"]) or link.parent
                context = normalize_space(container.get_text(" ", strip=True))
                if not re.search(r"\b(Powell|Warsh)\b", context, re.I):
                    continue
                full_url = urljoin(BASE, href)
                date = listing_date(link, full_url)
                speaker_match = re.search(r"\b(Powell|Warsh)\b", context, re.I)
                speaker = speaker_match.group(1).title() if speaker_match else None
                # "Chair's speeches" means the incumbent chair, not every later
                # appearance by a former chair who remains on the Board.
                if date and speaker != chair_for_date(date):
                    continue
                found[full_url] = {
                    "url": full_url,
                    "document_type": doc_type,
                    "fallback_date": date,
                    "speaker": speaker,
                }
    return list(found.values())


def collect(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    ensure_directories()
    client = FedClient()
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    links = discover_fomc_links(client, years) + discover_chair_links(client, years)
    links = list({item["url"]: item for item in links}.values())
    records: list[Document] = []

    for number, item in enumerate(links, 1):
        url = item["url"]
        fallback = item.get("fallback_date")
        if not fallback:
            print(f"Skipping undated URL: {url}")
            continue
        if not (pd.Timestamp(start) <= pd.Timestamp(fallback) <= pd.Timestamp(end)):
            continue
        try:
            content = client.get(url).content
            if url.lower().endswith(".pdf"):
                text, title = pdf_text(content), f"FOMC Press Conference - {fallback}"
            else:
                text, title = page_text_and_title(content)
            if item["document_type"] == "statement" and "fomc statement" not in (title + " " + text[:250]).lower():
                print(f"Skipping non-meeting monetary release: {url}")
                continue
        except Exception as exc:
            print(f"Warning: failed {url}: {exc}")
            continue

        default = "14:30" if item["document_type"] == "press_conference" else (
            "14:00" if item["document_type"] in {"statement", "minutes"} else "09:00"
        )
        stamp, imputed = parse_release_datetime(text, fallback, default)
        release_date = str(stamp.date())
        if not (pd.Timestamp(start) <= pd.Timestamp(release_date) <= pd.Timestamp(end)):
            continue
        doc_id = stable_id(item["document_type"], release_date, url)
        path = RAW / "documents" / f"{doc_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        relative_path = str(path.relative_to(RAW.parent.parent))
        records.append(
            Document(
                document_id=doc_id,
                document_type=item["document_type"],
                document_group=(
                    item["document_type"]
                    if item["document_type"] in {"statement", "minutes"}
                    else "chair_communication"
                ),
                title=title,
                chair=chair_for_date(release_date),
                release_date=release_date,
                release_time_et=stamp.strftime("%H:%M"),
                release_datetime_et=stamp.isoformat(),
                time_imputed=imputed,
                source_url=url,
                text_path=relative_path,
                word_count=len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text)),
            )
        )
        print(f"[{number}/{len(links)}] {item['document_type']}: {release_date}")

    df = pd.DataFrame(asdict(record) for record in records)
    if df.empty:
        raise RuntimeError("No documents collected. Check connectivity and Fed page structure.")
    df = df.drop_duplicates("document_id").sort_values(["release_date", "document_type"])
    save_csv(df, INTERIM / "documents.csv")
    print(df.groupby(["chair", "document_type"]).size())
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    args = parser.parse_args()
    collect(args.start, args.end)


if __name__ == "__main__":
    main()
