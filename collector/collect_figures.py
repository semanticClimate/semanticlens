"""
Collect figure metadata and captions from IPCC AR6 Working Groups I, II and III

Installation:
    pip install requests beautifulsoup4

Usage:
    python -m collector.collect_figures
    python -m collector.collect_figures --retry-failed

Note:
    Respect IPCC's robots.txt and copyright policies.
    Captions are collected for local research purposes.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger("ipcc_collector")
SCHEMA_VERSION = "1.0"
SITE_ORIGIN = "https://www.ipcc.ch"
WG_ROOTS = {
    wg: f"{SITE_ORIGIN}/report/ar6/{wg}/figures/"
    for wg in ("wg1", "wg2", "wg3")
}
USER_AGENT = (
    "IPCC-Visual-Knowledge-Graph/0.1 "
    "(+https://github.com/neelkumar01/IPCC-visual-knowledge-graph)"
)
COPYRIGHT_URL = "https://www.ipcc.ch/copyright/"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".svg"})
SKIP_SUFFIXES = IMAGE_SUFFIXES | frozenset({".pdf", ".zip", ".csv", ".xlsx", ".docx", ".xml", ".json", ".js", ".css"})
CAPTION_HEADING = re.compile(r"^figure\s+caption\s*:?$", re.IGNORECASE)
FIGURE_LABEL = re.compile(r"^(?:figure\b|box\b|cross.chapter\b|faq\b)", re.IGNORECASE)
CHAPTER_SLUG = re.compile(r"^chapter-[\w-]+$", re.IGNORECASE)
MAX_HTML_BYTES = 8_000_000


def utc_timestamp() -> str:
    """
    UTC ISO 8601 timestamp for provenance and reports
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compact_text(value: str) -> str:
    """
    Normalize whitespace (including non breaking spaces) without paraphrasing
    """
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def normalize_url(href: str, base: str) -> str | None:
    """
    Convert a link into a valid IPCC URL

    - Allow only HTTP/HTTPS links from IPCC websites
    - Keep the original capitalization of URL paths
    - Remove query parameters and fragments to avoid duplicate URLs
    """
    if not href or href.strip().startswith(("#", "mailto:", "javascript:", "tel:")):
        return None
    absolute = urljoin(base, href.strip())
    parsed = urlsplit(absolute)
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower() not in {"ipcc.ch", "www.ipcc.ch"}:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path)
    if not path:
        path = "/"
    if Path(path).suffix.lower() not in SKIP_SUFFIXES | {".html", ".htm"} and not path.endswith("/"):
        path += "/"
    return urlunsplit(("https", "www.ipcc.ch", path, "", ""))


def is_html_in_scope(url: str, wg: str) -> bool:
    """
    Only follow figure collection pages of this WG, never arbitrary IPCC links
    """
    root = WG_ROOTS[wg]
    path = urlsplit(url).path.lower()
    return url.startswith(root) and Path(path.rstrip("/")).suffix.lower() not in SKIP_SUFFIXES


def discover_links(soup: BeautifulSoup, source_url: str, wg: str) -> list[str]:
    """
    Find detail pages, chapter indexes, SPM / TS / Atlas and other linked sections
    """
    found: list[str] = []
    seen: set[str] = set()
    for link in soup.select("a[href]"):
        target = normalize_url(str(link.get("href", "")), source_url)
        if target and target != source_url and target not in seen and is_html_in_scope(target, wg):
            found.append(target)
            seen.add(target)
    return found


def caption_heading(soup: BeautifulSoup) -> Tag | None:
    """
    Locate a figure detail page's explicit caption section
    """
    for heading in soup.find_all(("h2", "h3", "h4", "h5", "h6")):
        if CAPTION_HEADING.match(compact_text(heading.get_text(" ", strip=True))):
            return heading
    return None


def _caption_from_siblings(node: Tag) -> tuple[str, str]:
    """
    Extract the figure caption without including unrelated content

    - Stop when the caption section ends
    - Preserve both HTML formatting and plain text for math and subscripts
    - Ignore scripts, navigation, and styling elements
    """
    text_chunks: list[str] = []
    html_chunks: list[str] = []
    for sibling in node.next_siblings:
        if isinstance(sibling, NavigableString):
            value = compact_text(str(sibling))
            if value:
                text_chunks.append(value)
                html_chunks.append(str(sibling))
            continue
        if not isinstance(sibling, Tag):
            continue
        if sibling.name in {"script", "style"}:
            continue
        if sibling.name in {"h1", "h2", "h3", "h4", "h5", "h6", "nav", "footer", "aside"}:
            break
        if sibling.find(("h1", "h2", "h3", "h4", "h5", "h6", "nav", "footer")):
            break
        value = compact_text(sibling.get_text(" ", strip=True))
        if not value:
            continue
        text_chunks.append(value)
        html_chunks.append(str(sibling))
        if sum(map(len, text_chunks)) > 100_000:
            LOGGER.warning("Caption exceeded 100KB; truncating HTML extraction")
            break
    return compact_text(" ".join(text_chunks)), "\n".join(html_chunks)


def extract_caption(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """
    Extract the figure caption as plain text and HTML

    - First, check elements immediately after the caption heading
    - If not found, check elements after its parent container
    - Finally, look for caption or figcaption elements
    - Return (None, None) if no caption is found
    """
    heading = caption_heading(soup)
    if heading:
        node: Tag | None = heading
        for _ in range(3):
            if node is None:
                break
            text, html = _caption_from_siblings(node)
            if text:
                return text, html
            node = node.parent if isinstance(node.parent, Tag) else None

    for candidate in soup.select("figcaption, .figure-caption, .caption-text, [id='figure-caption']"):
        text = compact_text(candidate.get_text(" ", strip=True))
        if text and not CAPTION_HEADING.match(text):
            return text, str(candidate)
    return None, None


def _media_link(url: str) -> bool:
    return Path(urlsplit(url).path.lower()).suffix in IMAGE_SUFFIXES


def extract_image_urls(soup: BeautifulSoup, source_url: str) -> list[str]:
    """
    Find direct URLs to figure images without downloading them

    - Check image download links on IPCC figure pages
    - Look for images stored in data-src or data-srcset
    - Accept only images related to the specific figure
    - Ignore generic images, logos, and unrelated thumbnails
    - Never guess image filenames or URLs
    """
    ranked: list[tuple[int, str]] = []
    for a in soup.select("a[href]"):
        url = normalize_url(str(a.get("href", "")), source_url)
        if not url or not _media_link(url):
            continue
        label = compact_text(a.get_text(" ", strip=True)).lower()
        path = urlsplit(url).path.lower()
        if "/downloads/figures/" in path or "figure" in label or "download" in label:
            ranked.append((0, url))

    for img in soup.select("img"):
        alt = compact_text(str(img.get("alt", ""))).lower()
        figure_image = (
            img.has_attr("data-main-image")
            or "figure" in alt
            or (img.find_parent("figure") is not None)
        )
        if not figure_image:
            continue
        for attribute in ("data-src", "src"):
            value = img.get(attribute)
            if isinstance(value, str):
                url = normalize_url(value, source_url)
                if url and _media_link(url):
                    ranked.append((1, url))
        for attribute in ("data-srcset", "srcset"):
            value = img.get(attribute)
            if not isinstance(value, str):
                continue
            for part in value.split(","):
                candidate = part.strip().split()
                if candidate:
                    url = normalize_url(candidate[0], source_url)
                    if url and _media_link(url):
                        ranked.append((2, url))

    ranked.sort(key=lambda entry: entry[0])
    return list(dict.fromkeys(url for _, url in ranked))


def parse_figure_page(soup: BeautifulSoup, url: str, wg: str) -> dict[str, Any] | None:
    """
    Extract one figure page record or None for section / navigation pages
    """
    heading = soup.find("h1")
    label = compact_text(heading.get_text(" ", strip=True)) if heading else ""
    section_heading = label.lower().startswith(("figures:", "all figures"))
    has_caption_heading = caption_heading(soup) is not None
    is_detail_label = bool(FIGURE_LABEL.match(label)) and not section_heading

    if not has_caption_heading and not is_detail_label:
        return None
    if not has_caption_heading and not extract_image_urls(soup, url):
        return None

    caption, caption_html = extract_caption(soup)
    images = extract_image_urls(soup, url)
    relative_path = urlsplit(url).path[len(urlsplit(WG_ROOTS[wg]).path):].strip("/")
    segments = [segment for segment in relative_path.split("/") if segment]
    chapter = next((s for s in segments if CHAPTER_SLUG.match(s)), None)
    section = "/".join(segments[:-1]) or "root"

    figure_id = f"AR6-{wg.upper()}-" + "--".join(segments)
    return {
        "schema_version": SCHEMA_VERSION,
        "figure_id": figure_id,
        "working_group": wg.upper(),
        "figure_label": label or None,
        "title": None,
        "chapter": chapter,
        "section": section,
        "caption": caption,
        "caption_html": caption_html,
        "image_url": images[0] if images else None,
        "image_urls": images,
        "source_url": url,
        "source_collection": WG_ROOTS[wg],
        "copyright_policy_url": COPYRIGHT_URL,
        "collected_at": utc_timestamp(),
    }


def atomic_json(path: Path, obj: Any) -> None:
    """
    Replace JSON atomically so an interrupted run cannot corrupt a checkpoint
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class Options:
    output_dir: Path = Path("data/raw")
    report_dir: Path = Path("reports")
    delay: float = 1.5
    timeout: float = 30.0
    checkpoint_every: int = 20
    max_pages: int | None = None
    fresh: bool = False
    retry_failed: bool = False
    ignore_robots: bool = False


class IPCCClient:
    """
    Polite, bounded HTTP client & respects robots.txt by default
    """

    def __init__(self, options: Options) -> None:
        self.options = options
        self.session = requests.Session()
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        })
        self._last_request = 0.0
        self._robots: RobotFileParser | None = None
        self._effective_delay = options.delay

    def __enter__(self) -> "IPCCClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.session.close()

    def _throttle(self) -> None:
        remaining = self._effective_delay - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> str:
        self._throttle()
        with self.session.get(url, timeout=self.options.timeout, stream=True) as response:
            if (urlsplit(response.url).hostname or "").lower() not in {"ipcc.ch", "www.ipcc.ch"}:
                raise ValueError(f"Refusing a redirect outside IPCC: {response.url}")
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not ("text/" in content_type or "html" in content_type):
                raise ValueError(f"Unexpected Content-Type {content_type!r} for {url}")
            chunks = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                chunks.extend(chunk)
                if len(chunks) > MAX_HTML_BYTES:
                    raise ValueError(f"Response exceeds {MAX_HTML_BYTES} bytes: {url}")
            declared = re.search(r"charset\s*=\s*['\"]?([\w.-]+)", content_type)
            encoding = declared.group(1) if declared else "utf-8-sig"
            return bytes(chunks).decode(encoding, errors="replace")

    def load_robots(self) -> None:
        if self.options.ignore_robots:
            LOGGER.warning("robots.txt checks explicitly disabled (--ignore-robots)")
            return
        robots_url = f"{SITE_ORIGIN}/robots.txt"
        try:
            content = self._get(robots_url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                LOGGER.info("IPCC robots.txt returned 404; no rules published at this URL")
                return
            raise RuntimeError("Could not verify IPCC robots.txt; stopping (fail closed)") from exc
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError("Could not verify IPCC robots.txt; stopping (fail closed)") from exc
        if "<html" in content[:2048].lower() or "<!doctype html" in content[:2048].lower():
            raise RuntimeError("Received an HTML page instead of robots.txt; stopping (fail closed)")
        robot = RobotFileParser()
        robot.set_url(robots_url)
        robot.parse(content.splitlines())
        self._robots = robot
        crawl_delay = robot.crawl_delay(USER_AGENT) or robot.crawl_delay("*")
        if crawl_delay is not None:
            self._effective_delay = max(self._effective_delay, float(crawl_delay))
        LOGGER.info("robots.txt loaded; effective delay %.1fs", self._effective_delay)

    def get_html(self, url: str) -> str:
        if self._robots is not None and not self._robots.can_fetch(USER_AGENT, url):
            raise PermissionError(f"robots.txt disallows {url}")
        return self._get(url)


class FigureCollector:
    """
    Breadth first crawl of WG figure directories with resumable checkpoints
    """

    def __init__(self, client: IPCCClient, options: Options) -> None:
        self.client = client
        self.options = options

    def _paths(self, wg: str) -> tuple[Path, Path]:
        return (
            self.options.output_dir / f"{wg}.json",
            self.options.report_dir / f"collector_state_{wg}.json",
        )

    def _load_state(self, wg: str) -> tuple[deque[str], set[str], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
        raw_path, state_path = self._paths(wg)
        if self.options.fresh:
            return deque([WG_ROOTS[wg]]), set(), {}, {}
        rows = read_json(raw_path, [])
        if not isinstance(rows, list):
            raise ValueError(f"Invalid figure corpus in {raw_path}: expected JSON array")
        figures = {row["source_url"]: row for row in rows}
        saved = read_json(state_path, {})
        if saved and saved.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Incompatible checkpoint in {state_path}; use --fresh")
        visited = set(saved.get("visited", []))
        failures = dict(saved.get("failures", {}))
        pending = deque(saved.get("pending", [WG_ROOTS[wg]]))
        if self.options.retry_failed:
            pending.extend(url for url in failures if url not in visited)
            failures.clear()
        
        if not saved:
            pending = deque([WG_ROOTS[wg]])
        return pending, visited, figures, failures

    def _save(self, wg: str, pending: Iterable[str], visited: set[str], figures: dict[str, dict[str, Any]], failures: dict[str, dict[str, str]]) -> None:
        raw_path, state_path = self._paths(wg)
        atomic_json(raw_path, sorted(figures.values(), key=lambda row: row["source_url"]))
        atomic_json(state_path, {
            "schema_version": SCHEMA_VERSION,
            "working_group": wg.upper(),
            "pending": list(pending),
            "visited": sorted(visited),
            "failures": failures,
            "checkpoint_at": utc_timestamp(),
        })

    def collect(self, wg: str) -> dict[str, Any]:
        pending, visited, figures, failures = self._load_state(wg)
        queued = set(pending)
        processed_this_run = 0
        root = WG_ROOTS[wg]

        while pending:
            if self.options.max_pages is not None and processed_this_run >= self.options.max_pages:
                break
            url = pending.popleft()
            queued.discard(url)
            if url in visited:
                continue
            processed_this_run += 1
            try:
                html = self.client.get_html(url)
                soup = BeautifulSoup(html, "html.parser")
                page_title = compact_text(soup.title.get_text(" ", strip=True)).lower() if soup.title else ""
                if any(phrase in page_title for phrase in ("just a moment", "access denied", "attention required", "verify you are human")):
                    raise ValueError(f"Site challenge or blocked response at {url}: {page_title}")
                for target in discover_links(soup, url, wg):
                    if target not in visited and target not in queued and target not in failures:
                        pending.append(target)
                        queued.add(target)
                record = parse_figure_page(soup, url, wg)
                if record:
                    figures[url] = record
                    LOGGER.info("%s figure %s", wg.upper(), record["figure_label"] or record["figure_id"])
                visited.add(url)
                failures.pop(url, None)
            except (requests.RequestException, PermissionError, ValueError) as exc:
                failures[url] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "last_attempt_at": utc_timestamp(),
                }
                LOGGER.warning("%s failed: %s: %s", wg.upper(), url, exc)
            if processed_this_run % self.options.checkpoint_every == 0:
                self._save(wg, pending, visited, figures, failures)
                LOGGER.info("%s checkpoint: %d figures, %d pending", wg.upper(), len(figures), len(pending))

        self._save(wg, pending, visited, figures, failures)
        chapter_counts = Counter(row["section"] for row in figures.values())
        result = {
            "working_group": wg.upper(),
            "seed_url": root,
            "status": "complete" if not pending and not failures else "partial",
            "visited_pages": len(visited),
            "pages_processed_this_run": processed_this_run,
            "figures_collected": len(figures),
            "figures_missing_caption": sum(not row.get("caption") for row in figures.values()),
            "figures_missing_image": sum(not row.get("image_url") for row in figures.values()),
            "figures_by_section": dict(sorted(chapter_counts.items())),
            "pending_pages": len(pending),
            "failed_pages": len(failures),
            "failures": failures,
            "raw_data_path": str(self._paths(wg)[0]),
        }
        LOGGER.info("%s: %d figures; %d pending pages; %d failed", wg.upper(), len(figures), len(pending), len(failures))
        return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wg", nargs="+", choices=tuple(WG_ROOTS), default=list(WG_ROOTS), help="Working groups to collect (default: all three)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--delay", type=float, default=1.5, help="Minimum seconds between requests (default: 1.5)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--max-pages", type=int, help="Process at most N HTML pages per WG in this run (test mode)")
    parser.add_argument("--fresh", action="store_true", help="Start selected WGs from scratch; replace their saved crawl progress")
    parser.add_argument("--retry-failed", action="store_true", help="Retry pages previously recorded as failed")
    parser.add_argument("--ignore-robots", action="store_true", help="Override robots.txt checks ONLY if permission has been verified")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.delay < 0 or args.timeout <= 0 or args.checkpoint_every <= 0 or (args.max_pages is not None and args.max_pages <= 0):
        parser.error("delay must be >= 0; timeout, checkpoint-every and max-pages must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    opts = Options(
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        delay=args.delay,
        timeout=args.timeout,
        checkpoint_every=args.checkpoint_every,
        max_pages=args.max_pages,
        fresh=args.fresh,
        retry_failed=args.retry_failed,
        ignore_robots=args.ignore_robots,
    )
    try:
        with IPCCClient(opts) as client:
            client.load_robots()
            collector = FigureCollector(client, opts)
            results = [collector.collect(wg) for wg in args.wg]
    except (RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("Collector stopped: %s", exc)
        return 2
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "raw_collection",
        "generated_at": utc_timestamp(),
        "scope": "IPCC AR6 Working Group figures (linked pages only)",
        "working_groups": results,
        "total_figures": sum(item["figures_collected"] for item in results),
        "all_complete": all(item["status"] == "complete" for item in results),
        "notes": [
            "Completeness means linked pages were visited without logged failures; it does not independently verify the IPCC site's total inventory.",
            "Image URLs are references; image binaries were not downloaded.",
            "IPCC image and caption reuse is subject to the IPCC copyright policy.",
        ],
    }
    atomic_json(opts.report_dir / "collection_report.json", report)
    LOGGER.info("Report: %s", opts.report_dir / "collection_report.json")
    return 0 if report["all_complete"] and report["total_figures"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
