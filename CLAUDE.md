# Aether — Claude Code Notes

Streamlit intraday/swing trading research dashboard. Seven pages, all data live
from yfinance, all state on local disk. See `README.md` for what the app *does*
and how a user drives it — this file is conventions, commands, and traps.

## Repo context

- **Standalone git repo.** Lives at `Finance/stock-app/aether/` but is gitignored
  by the parent `Finance` monorepo and has its own history. `git` commands run
  here affect only Aether.
- Python 3.12, no build step, no server — `streamlit run app.py` is the whole app.

## Commands

```bash
streamlit run app.py          # run the dashboard (localhost:8501)
pytest tests/ -q              # full suite, ~90s, no network needed
pytest tests/test_orbc_strategy.py -q   # fast subset, ~3s
python3 -m py_compile <files> # quick syntax check before running anything slow
sqlite3 storage/journal.db "select * from activity_log"
```

## Layer rules

- `analysis/` and `data/` are **Streamlit-free** — pure functions over DataFrames.
  Never import `streamlit` there; it makes the logic untestable.
- `pages/` are **display-only**. Detection/scoring/backtest logic belongs in
  `analysis/`. If a page starts computing, move it.
- Every `pages/*.py` calls `render()` at module level, so **importing a page runs
  it**. Streamlit tolerates this in bare mode (`test_pages_trading_imports_cleanly`
  relies on it as an import-crash guard), but a page that fetches on load will hit
  the network. Prefer testing the `analysis/` module plus an import-contract
  assertion (see `test_page_import_contract_is_satisfied`).
- `config/settings.py` centralizes thresholds. Don't hardcode a cutoff in a page.

## Timestamps — the recurring source of bugs

- Always use `config/tz.py` (`now_et()`, `now_et_iso()`, `utc_iso_to_et_str()`,
  `MARKET_TZ`). Never bare `datetime.now()` or `datetime.utcnow()`.
- **Stored logs mix formats.** `storage/*_predictions.jsonl` has both legacy naive
  (`datetime.utcnow().isoformat()`) and current tz-aware rows. Parsing them needs
  `pd.to_datetime(..., utc=True, format="mixed")` — without `format="mixed"`,
  pandas infers one format from row 1 and silently coerces the rest to `NaT`,
  which a downstream `dropna()` then drops. This exact bug hid every prediction
  newer than 2026-07-09 for weeks.
- When parsing a stored timestamp, treat **naive as UTC** (matches
  `utc_iso_to_et_str`). But for **intraday market data**, treat naive as
  market-local ET — yfinance returns exchange-local timestamps, and localizing to
  UTC instead would shift 09:30 ET to 05:30 and break session-window logic. See
  `orbc_strategy.to_market_tz()`.
- Time-of-day comparisons (market open, entry cutoffs) must happen *after*
  normalizing to `MARKET_TZ`.

## Backtest conventions

- **Close-to-close, no intrabar fills.** Entries and exits are evaluated on bar
  closes. Consistent across `analysis/backtest.py`,
  `mtf_strategy.evaluate_setup_trade`, `flag_pennant_backtest`, and
  `orbc_strategy.evaluate_orbc_trade`. Don't mix in high/low touches.
- **No lookahead.** Two specific traps:
  - Resampled bars (e.g. `resample_to_4h`) are labeled at their **start**, so
    `df[df.index <= ts]` includes a bar that isn't finished yet and whose
    High/Low/Close aggregate future data. Require the bar's full duration to have
    elapsed: `index <= ts - Timedelta(hours=4)`.
  - Simulate from the bar **after** the signal bar, never from the signal bar.
- Intraday history is capped by yfinance at ~60 days for 5m/30m bars. Report the
  window actually fetched rather than the window requested.
- Report sample size alongside win rate. A few dozen trades is not a result.

## Testing

- `storage/` holds **real trained models and real trade history** and is
  gitignored. Tests must never write there — use the `isolated_storage` fixture
  (monkeypatches `ml_prediction._STORAGE_DIR` to `tmp_path`).
- No network in tests. yfinance is unreachable in sandbox/CI. Use the synthetic
  fixtures in `tests/conftest.py`:
  - `synthetic_indicators_df` — daily OHLCV through real `calculate_indicators()`
  - `make_intraday_session(closes, date=...)` — one ET session from an explicit
    list of closes, for testing bar-by-bar state machines deterministically
- Functions that fetch accept an optional `df=` override so they can be tested
  offline (see `backtest_orbc`).

## Gotchas

- **Two prediction models, deliberately separate.** `analysis/ml_prediction.py`
  is daily; `analysis/intraday_prediction.py` is intraday. Do **not** merge them
  or add an `interval` param to the daily one — `ml_prediction.predict()` is
  consumed by four pages (trading, research, watchlist, screener), so changes
  there have a wide blast radius. The intraday module imports the daily module's
  `_run_walk_forward` / `_xgb_config` / `_rf_config` / `_filter_directional`
  **read-only**; keep it that way. Storage is interval-scoped
  (`SPY_15m_xgb.pkl`) so the two never collide.
- **Intraday labels need two guards.** Forward returns must not cross a session
  boundary, and the neutral band must scale with *trailing* volatility (a
  full-series sigma makes each label depend on future bars). A fixed percentage
  band calibrated for daily bars labels 66-93% of 15m bars neutral, and neutral
  rows are dropped before training. See `build_intraday_labels()`.
- **`st.session_state` cache keys need a date component** or they serve stale
  results for the whole browser session while the rest of the page refreshes.
- **`st.cache_data(ttl=...)`** — pick a TTL that matches the bar size; a 60s TTL
  on a 60-day fetch refetches constantly.
- **Filters skip, don't fail, on missing indicator columns.** Short history means
  NaN ATR/VWAP/`vol_sma_20`. The convention (`passes_trend_filter`,
  `orbc_strategy._check_filters`) is to skip the filter and record that it was
  skipped — never silently reject every signal.
- **Growth/ratio units.** `data/fundamentals.py` returns growth rates as
  *fractions*. Don't add "is it already a percent?" heuristics like
  `x * 100 if abs(x) < 1 else x` — they misread ≥100% growth as ~1%.
- **`ai/client.py` is multi-provider.** Gate AI features on `ai_available()`, not
  on `ANTHROPIC_API_KEY` — Ollama works with no key.
- Optional imports (`_ML_AVAILABLE`, `_NEWS_AVAILABLE`) let pages degrade if
  sklearn/xgboost/feedparser are missing. Keep that pattern when adding heavy deps.
- `narwhals` is pinned in `requirements.txt` on purpose — a transitive
  sklearn/plotly dep whose version drift once took down the whole Predictions tab.

## Docs

`docs/ML_PREDICTION.md` (ensemble internals), `docs/workflow.md` (intended daily
usage), `docs/ORBC_PLAYBOOK.md` (ORBC rules + design decisions + trading routine),
`docs/UI_DESIGN_SPEC.md` (theme tokens — global stylesheet lives only in `app.py`),
`docs/VERIFICATION_CHECKLIST.md` (manual checks; its line numbers drift).
`docs/DATA_STORAGE.md` and `docs/MONOREPO_EXTRACTION.md` are gitignored/local-only.
