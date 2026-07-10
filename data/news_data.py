"""
News headline fetcher — Google News RSS via feedparser, no API key required.
Headline/link/published only; sentiment scoring only needs the headline text,
so the full-article scrape used in the source prototype is intentionally dropped.
"""
import time
import logging
from typing import Any, Dict, List
from urllib.parse import quote

import feedparser

from config.settings import NEWS_CACHE_TTL, NEWS_MAX_ARTICLES

logger = logging.getLogger(__name__)

_cache: Dict[str, Dict] = {}


def _fresh(entry: dict, ttl: int) -> bool:
    return (time.time() - entry["ts"]) < ttl


def fetch_ticker_news(
    ticker: str,
    company_name: str = "",
    max_articles: int = NEWS_MAX_ARTICLES,
    ttl: int = NEWS_CACHE_TTL,
) -> List[Dict[str, Any]]:
    """
    Fetch recent headlines for a ticker from Google News RSS.

    Returns a list of dicts with title/link/published/published_ts/source.
    published_ts is a unix timestamp (or None if unparseable) for sorting.
    Returns an empty list on any fetch error rather than raising, since this
    is a display-only signal that should never break page rendering.
    """
    ticker = ticker.upper().strip()
    key = f"news_{ticker}_{max_articles}"
    if key in _cache and _fresh(_cache[key], ttl):
        return _cache[key]["data"]

    query = f"{company_name} {ticker} stock".strip() if company_name else f"{ticker} stock"

    try:
        rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
        feed = feedparser.parse(rss_url)
        articles = []
        for entry in feed.entries[:max_articles]:
            source = entry.get("source")
            published_parsed = entry.get("published_parsed")
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "published_ts": time.mktime(published_parsed) if published_parsed else None,
                "source": source.get("title", "") if source else "",
            })
        _cache[key] = {"data": articles, "ts": time.time()}
        return articles
    except Exception as e:
        logger.error(f"News fetch error for {ticker}: {e}")
        return []
