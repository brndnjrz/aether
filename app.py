"""
Aether — app entrypoint. Owns page config, theme, and navigation.
"""
import logging
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Aether",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger.info("Aether app initializing — page config set")

# Aether global stylesheet — theme-aware tokens + component classes.
# Single source of truth for all pages (see docs/UI_DESIGN_SPEC.md).
st.markdown("""
<style>
:root {
    /* ---- Base surfaces (theme-aware, derived from Streamlit vars) ---- */
    --aeth-surface: var(--secondary-background-color);
    --aeth-surface-raised: color-mix(in srgb, var(--secondary-background-color) 92%, var(--text-color) 8%);
    --aeth-border: color-mix(in srgb, var(--text-color) 14%, transparent);
    --aeth-border-strong: color-mix(in srgb, var(--text-color) 24%, transparent);
    --aeth-text-muted: color-mix(in srgb, var(--text-color) 62%, transparent);

    /* ---- Semantic signal colors ----
       Tuned so the SAME variable resolves to a readable, sufficiently-
       saturated color on both a near-white and a near-black background.
       Base hues kept close to the existing #26a69a / #ef5350 pair so the
       app's visual identity doesn't shift, just becomes theme-safe. */
    --aeth-bull: color-mix(in srgb, #1a9c85 85%, var(--text-color) 15%);
    --aeth-bull-bg: color-mix(in srgb, #1a9c85 14%, var(--background-color) 86%);
    --aeth-bull-border: color-mix(in srgb, #1a9c85 45%, var(--background-color) 55%);

    --aeth-bear: color-mix(in srgb, #e5484d 85%, var(--text-color) 15%);
    --aeth-bear-bg: color-mix(in srgb, #e5484d 14%, var(--background-color) 86%);
    --aeth-bear-border: color-mix(in srgb, #e5484d 45%, var(--background-color) 55%);

    --aeth-neutral: var(--aeth-text-muted);
    --aeth-neutral-bg: color-mix(in srgb, var(--text-color) 8%, var(--background-color) 92%);
    --aeth-neutral-border: color-mix(in srgb, var(--text-color) 20%, var(--background-color) 80%);

    --aeth-warn: color-mix(in srgb, #d97706 85%, var(--text-color) 15%);
    --aeth-warn-bg: color-mix(in srgb, #d97706 14%, var(--background-color) 86%);
    --aeth-warn-border: color-mix(in srgb, #d97706 40%, var(--background-color) 60%);

    /* ---- Spacing scale (4px base unit) ---- */
    --aeth-space-1: 0.25rem;  /* 4px  */
    --aeth-space-2: 0.5rem;   /* 8px  */
    --aeth-space-3: 0.75rem;  /* 12px */
    --aeth-space-4: 1rem;     /* 16px */
    --aeth-space-6: 1.5rem;   /* 24px */
    --aeth-space-8: 2rem;     /* 32px */

    /* ---- Radius ---- */
    --aeth-radius-sm: 6px;
    --aeth-radius-md: 10px;
    --aeth-radius-pill: 999px;

    /* ---- Shadow (subtle, single-layer — no gradient/glow) ---- */
    --aeth-shadow-sm: 0 1px 2px color-mix(in srgb, var(--text-color) 8%, transparent);
    --aeth-shadow-md: 0 2px 8px color-mix(in srgb, var(--text-color) 10%, transparent);

    /* ---- Motion ---- */
    --aeth-transition: 150ms ease;
}

/* ---- 3.1 Metric / stat card ---- */
div[data-testid="metric-container"],
div[data-testid="stMetric"] {
    background-color: var(--aeth-surface);
    border: 1px solid var(--aeth-border);
    border-radius: var(--aeth-radius-md);
    padding: var(--aeth-space-3) var(--aeth-space-4);
    box-shadow: var(--aeth-shadow-sm);
    transition: border-color var(--aeth-transition), box-shadow var(--aeth-transition);
}

div[data-testid="metric-container"]:hover,
div[data-testid="stMetric"]:hover {
    border-color: var(--aeth-border-strong);
    box-shadow: var(--aeth-shadow-md);
}

/* Streamlit's own metric label/value typography, restated for hierarchy */
div[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--aeth-text-muted);
    letter-spacing: 0.01em;
}

div[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: 700;
}

/* ---- 3.2 Signal card (bull / bear / neutral) ---- */
.signal-card {
    border-radius: var(--aeth-radius-md);
    padding: var(--aeth-space-3) var(--aeth-space-4);
    margin-bottom: var(--aeth-space-2);
    border: 1px solid transparent;
    border-left-width: 4px;
    box-shadow: var(--aeth-shadow-sm);
}

.signal-card strong {
    font-size: 0.95rem;
    font-weight: 600;
}

.signal-card .signal-value {
    font-size: 1.1rem;
    font-weight: 600;
    display: block;
    margin: var(--aeth-space-1) 0;
}

.signal-card .signal-note {
    font-size: 0.8rem;
    color: var(--aeth-text-muted);
}

.signal-bull {
    background-color: var(--aeth-bull-bg);
    border-color: var(--aeth-bull-border);
    border-left-color: var(--aeth-bull);
}
.signal-bull strong,
.signal-bull .signal-value { color: var(--aeth-bull); }

.signal-bear {
    background-color: var(--aeth-bear-bg);
    border-color: var(--aeth-bear-border);
    border-left-color: var(--aeth-bear);
}
.signal-bear strong,
.signal-bear .signal-value { color: var(--aeth-bear); }

.signal-neutral {
    background-color: var(--aeth-neutral-bg);
    border-color: var(--aeth-neutral-border);
    border-left-color: var(--aeth-neutral);
}
.signal-neutral strong,
.signal-neutral .signal-value { color: var(--aeth-neutral); }

/* ---- 3.3 Badge / pill (confidence, reliability, IV rank, etc.) ---- */
.aeth-badge {
    display: inline-flex;
    align-items: center;
    gap: var(--aeth-space-1);
    padding: var(--aeth-space-1) var(--aeth-space-3);
    border-radius: var(--aeth-radius-pill);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    border: 1px solid transparent;
    line-height: 1.6;
}

.aeth-badge--bull   { background: var(--aeth-bull-bg);    color: var(--aeth-bull);    border-color: var(--aeth-bull-border); }
.aeth-badge--bear   { background: var(--aeth-bear-bg);    color: var(--aeth-bear);    border-color: var(--aeth-bear-border); }
.aeth-badge--neutral{ background: var(--aeth-neutral-bg); color: var(--aeth-neutral); border-color: var(--aeth-neutral-border); }
.aeth-badge--warn   { background: var(--aeth-warn-bg);    color: var(--aeth-warn);    border-color: var(--aeth-warn-border); }

/* ---- 3.4 Disclaimer / warning box ---- */
.aeth-disclaimer {
    background-color: var(--aeth-warn-bg);
    border: 1px solid var(--aeth-warn-border);
    border-radius: var(--aeth-radius-md);
    padding: var(--aeth-space-4) var(--aeth-space-6);
    margin-top: var(--aeth-space-2);
    font-size: 0.85rem;
    line-height: 1.5;
}

.aeth-disclaimer strong.aeth-disclaimer__title {
    color: var(--aeth-warn);
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    display: block;
    margin-bottom: var(--aeth-space-2);
}

/* ---- 3.5 Status strip (market status / VIX / regime bar) ---- */
.aeth-status-strip {
    background-color: var(--aeth-surface);
    border: 1px solid var(--aeth-border);
    border-radius: var(--aeth-radius-sm);
    padding: var(--aeth-space-2) var(--aeth-space-4);
    margin-bottom: var(--aeth-space-3);
    font-size: 0.85rem;
}

.aeth-status-strip .aeth-status-open   { color: var(--aeth-bull); font-weight: 700; }
.aeth-status-strip .aeth-status-closed{ color: var(--aeth-warn); font-weight: 700; }

/* ---- 3.6 Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: var(--aeth-surface);
    border-right: 1px solid var(--aeth-border);
}

[data-testid="stSidebar"] hr {
    border-color: var(--aeth-border);
}

/* ---- 3.7 Page background ---- */
.main {
    background-color: var(--background-color);
}

/* ---- 4. Typography & hierarchy ---- */
h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 1.375rem; font-weight: 700; }
h3 { font-size: 1.125rem; font-weight: 600; }

/* st.caption renders as small/muted already; reinforce the muted token
   rather than leaving it to Streamlit's own default gray, which does not
   reliably meet 4.5:1 contrast in light mode */
[data-testid="stCaptionContainer"] {
    color: var(--aeth-text-muted);
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 📈 Aether")
    st.markdown("*Premium analysis. Real data. No mocks.*")
    st.markdown("---")

    quick_ticker = st.text_input("Quick Lookup", placeholder="Enter ticker...")
    if quick_ticker:
        quick_ticker = quick_ticker.upper().strip()
        if st.session_state.get("quick_lookup_ticker") != quick_ticker:
            logger.info(f"Quick Lookup ticker set to {quick_ticker}")
        st.session_state["quick_lookup_ticker"] = quick_ticker
        st.page_link("pages/research.py", label=f"Research {quick_ticker} →")

    from ai.client import active_provider
    provider_label = active_provider()
    if "None" in provider_label:
        logger.warning(f"AI provider check: {provider_label} — no provider configured")
        st.warning("⚠️ No AI provider — start Ollama or add ANTHROPIC_API_KEY")
    elif "no key" in provider_label:
        logger.warning(f"AI provider check: {provider_label} — AI_PROVIDER set to claude but no key configured")
        st.warning("⚠️ AI_PROVIDER=claude but no key set")
    else:
        logger.info(f"AI provider check: {provider_label} — active and available")
        st.success(f"🤖 AI: {provider_label}")

    st.markdown("---")

pg = st.navigation([
    st.Page("pages/home.py", title="Dashboard", icon="📈", default=True),
    st.Page("pages/research.py", title="Research", icon="🔍"),
    st.Page("pages/portfolio.py", title="Portfolio", icon="💼"),
    st.Page("pages/watchlist.py", title="Watchlist", icon="📋"),
    st.Page("pages/screener.py", title="Screener", icon="🔎"),
    st.Page("pages/trading.py", title="Trading Desk", icon="🎯"),
    st.Page("pages/strategy_lab.py", title="Strategy Lab", icon="🧭"),
])
pg.run()
