"""
Aether — app entrypoint. Owns page config, theme, and navigation.
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Aether",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark theme CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1f2e; border-radius: 8px; padding: 8px; }
    .metric-positive { color: #26a69a; }
    .metric-negative { color: #ef5350; }
    [data-testid="stSidebar"] { background-color: #0a0f1a; }
    .st-emotion-cache-1y4p8pa { max-width: 100%; }
    div[data-testid="metric-container"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 12px;
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
        st.session_state["quick_lookup_ticker"] = quick_ticker
        st.page_link("pages/research.py", label=f"Research {quick_ticker} →")

    from ai.client import active_provider
    provider_label = active_provider()
    if "None" in provider_label:
        st.warning("⚠️ No AI provider — start Ollama or add ANTHROPIC_API_KEY")
    elif "no key" in provider_label:
        st.warning("⚠️ AI_PROVIDER=claude but no key set")
    else:
        st.success(f"🤖 AI: {provider_label}")

    st.markdown("---")

pg = st.navigation([
    st.Page("pages/home.py", title="Dashboard", icon="📈", default=True),
    st.Page("pages/research.py", title="Research", icon="🔍"),
    st.Page("pages/portfolio.py", title="Portfolio", icon="💼"),
    st.Page("pages/watchlist.py", title="Watchlist", icon="📋"),
    st.Page("pages/screener.py", title="Screener", icon="🔎"),
    st.Page("pages/trading.py", title="Trading Desk", icon="🎯"),
])
pg.run()
