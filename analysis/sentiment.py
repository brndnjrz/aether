"""
News headline sentiment scoring — VADER lexicon-based analysis.

Headline-only (not full-article), consistent with what actually drives the
compound score in the source prototype this was ported from. Google News RSS
has no historical query-by-date capability, so this signal has no backfill
path and is intentionally kept out of the ML training pipeline
(data/feature_engineering.py) — display-only, current-headlines only.
"""
import logging
from typing import Any, Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def score_headline(text: str) -> Dict[str, Any]:
    """Score a single headline. compound is in [-1, 1]; label follows VADER's own thresholds."""
    if not text:
        logger.debug("score_headline: empty headline text, defaulting to Neutral.")
        return {"compound": 0.0, "label": "Neutral"}
    compound = _analyzer.polarity_scores(text)["compound"]
    if compound > POSITIVE_THRESHOLD:
        label = "Positive"
    elif compound < NEGATIVE_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"
    return {"compound": compound, "label": label}


def analyze_ticker_sentiment(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score a list of articles (from data.news_data.fetch_ticker_news) and return
    an aggregate summary plus per-article scores, most recent first.
    """
    if not articles:
        logger.warning("analyze_ticker_sentiment: no articles provided, returning neutral default.")
        return {
            "n_articles": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "mean_compound": 0.0,
            "overall_label": "Neutral",
            "articles": [],
        }

    scored = [{**article, **score_headline(article.get("title", ""))} for article in articles]

    n = len(scored)
    n_pos = sum(1 for a in scored if a["label"] == "Positive")
    n_neg = sum(1 for a in scored if a["label"] == "Negative")
    n_neu = n - n_pos - n_neg
    mean_compound = sum(a["compound"] for a in scored) / n

    if mean_compound > POSITIVE_THRESHOLD:
        overall = "Bullish"
    elif mean_compound < NEGATIVE_THRESHOLD:
        overall = "Bearish"
    else:
        overall = "Neutral"

    scored.sort(key=lambda a: a.get("published_ts") or 0, reverse=True)

    logger.info(
        f"analyze_ticker_sentiment: n_articles={n} overall_label={overall} mean_compound={round(mean_compound, 3)}"
    )
    return {
        "n_articles": n,
        "positive_pct": round(n_pos / n * 100, 1),
        "negative_pct": round(n_neg / n * 100, 1),
        "neutral_pct": round(n_neu / n * 100, 1),
        "mean_compound": round(mean_compound, 3),
        "overall_label": overall,
        "articles": scored,
    }
