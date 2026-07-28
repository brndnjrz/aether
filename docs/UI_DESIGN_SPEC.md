# Aether UI Design Spec — Theme-Safe Fintech Refresh

Scope: replace the hardcoded-hex global stylesheet in `app.py` (lines 16-31)
and the duplicate stylesheet in `pages/trading.py` (lines 46-57) with one
theme-aware stylesheet, defined once in `app.py`, consumed everywhere. No new
libraries, no build step — this is a `st.markdown(<style>...)` block plus
class names applied via small Python helper functions that return HTML
strings (the same pattern `_signal_html()` in `trading.py` already uses).

This is a spec, not code — concrete enough to implement without further
design decisions, but the implementer writes the actual CSS/Python.

---

## 1. Streamlit's theme-aware CSS variables (use these, not hex)

Streamlit injects these CSS custom properties on `<html>`/`:root` and they
flip automatically when the user toggles light/dark in the app menu
(Settings → Theme). Confirmed available in current Streamlit versions:

| Variable | Light mode resolves to | Dark mode resolves to | Use for |
|---|---|---|---|
| `--background-color` | `#ffffff` | `#0e1117` | page background |
| `--secondary-background-color` | `#f0f2f6` | `#262730` | card/widget surfaces, sidebar |
| `--text-color` | `#31333f` | `#fafafa` | body text |
| `--primary-color` | app accent (default `#ff4b4b`, but themeable) | same | focus rings, primary buttons, active nav |
| `--font` | configured font stack | same | rarely needed directly; Streamlit applies it to body already |

Streamlit does **not** expose a variable for borders, shadows, or muted/secondary
text — we define our own tokens for those (section 2) derived *from* the
variables above via `color-mix()`, so they still flip correctly with theme
and don't need a separate light/dark override block.

Do not use `[data-theme="dark"]` selectors anywhere in this spec — that
pattern (seen in generic Streamlit theming tutorials) requires manually
maintaining two palettes and is exactly the bug we're removing. Everything
below is theme-agnostic by construction.

---

## 2. Derived design tokens (define once, in `app.py`)

Place this at the top of the single global `<style>` block, inside `:root`.
`color-mix()` has been supported in all evergreen browsers (and therefore
Streamlit's Chromium-based renderer) since 2023, so this is safe to rely on.

```css
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
```

Why `color-mix()` with `var(--text-color)` and `var(--background-color)` as
inputs instead of fixed alpha-on-black/alpha-on-white: `--text-color` and
`--background-color` are *already* the correct value for the active theme,
so mixing toward them automatically produces a light-mode-appropriate tint in
light mode and a dark-mode-appropriate tint in dark mode from one rule. This
is what satisfies requirement 2 (green/red stay recognizable but are tuned
per-theme) without writing two palettes.

If the implementer's Streamlit version renders `color-mix()` unreliably
(older embedded Chromium), the fallback is to hand-write two blocks gated on
`@media (prefers-color-scheme: dark)` — but Streamlit's in-app theme toggle
is a JS/localStorage mechanism, not the OS-level `prefers-color-scheme`, so
that fallback will not track Streamlit's Settings menu correctly. Test
`color-mix()` support first; it should work.

**Correction after first implementation:** the original draft mixed 55% base
hue with 45% `--text-color` for `--aeth-bull`/`--aeth-bear`/`--aeth-warn`. In
dark mode `--text-color` resolves to near-white (`#fafafa`), so 45% white
washed the vivid teal/red down to pale mint/pink pastels — the signal cards
lost their bull/bear punch. Reduced to 85% base hue / 15% `--text-color` for
all three; still shifts slightly per theme for contrast, but stays
recognizably saturated in both. If retuning further, keep this ratio low
(≤20%) — it exists for contrast nudging, not for tinting the hue itself.

---

## 3. Component classes

All classes below go in the same global `<style>` block in `app.py`. Pages
apply them via HTML strings returned from small helper functions (same
pattern as the existing `_signal_html()` in `trading.py` — keep that
approach, just point it at these classes instead of inline hex).

### 3.1 Metric / stat card

Replaces the current `div[data-testid="metric-container"]` override (which
duplicates styling in both `app.py` and `trading.py` today — consolidate to
one definition in `app.py`).

```css
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
```

Note: `div[data-testid="metric-container"]` is a legacy selector from older
Streamlit; `div[data-testid="stMetric"]` is current. Keep both selectors
(comma-joined, as above) so the rule survives a Streamlit version bump
either direction — this is defensive, not redundant.

### 3.2 Signal card (bull / bear / neutral)

Replaces `.signal-card` / `.signal-bull` / `.signal-bear` / `.signal-neutral`
in `trading.py` lines 48-51. Same class names, new variable-based values —
existing `_signal_html()` call sites in `trading.py` need zero changes
beyond removing the now-redundant local `<style>` block.

```css
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
```

Implementer note: `trading.py`'s current `_signal_html()` emits
`<small style="color:#aaa;">` for the interpretation line — change that
`<small>` to `<span class="signal-note">` so it inherits the theme-safe muted
color instead of a hardcoded gray that disappears in light mode.

### 3.3 Badge / pill (confidence, reliability, IV rank, etc.)

New reusable class — currently ad hoc inline in `trading.py`
(`_render_prediction_card`, confidence badge around line 1066-1072) and via
emoji-in-metric-label elsewhere. Consolidate all "small labeled status chip"
uses into this one class + 4 modifiers.

```css
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
```

Mapping from current usage:
- Confidence HIGH → `aeth-badge--bull`, MODERATE → `aeth-badge--warn`, LOW → `aeth-badge--neutral`
- Model reliability ✅ → `aeth-badge--bull`, ⚠️ → `aeth-badge--warn`
- IV Rank sell-premium/buy-premium/neutral chips (currently plain-text
  🔴/🟢/🟡 prefixes in `_render_options`, line ~812) → same three-way mapping

Keep the emoji glyph inside the badge as a leading character in the text
content (e.g. `▲ HIGH`) — emoji are theme-agnostic and reinforce the
semantic color for colorblind users; don't drop them when moving to badges.

### 3.4 Disclaimer / warning box

Replaces the inline-styled div in `trading.py`
`_render_predictions_disclaimer()` (lines 1414-1438), which is currently the
single largest hardcoded-hex block in the app outside the top-of-file style
tags.

```css
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
```

Usage (replaces the `<b style='color:#ef5350'>IMPORTANT DISCLAIMER</b>` block):

```html
<div class="aeth-disclaimer">
  <strong class="aeth-disclaimer__title">Important Disclaimer</strong>
  ...body text, no inline styles needed...
</div>
```

Note the semantic shift from red (`--aeth-bear`) to amber (`--aeth-warn`) —
the current implementation colors a *methodology disclaimer* the same red as
"bearish," which conflates "this is a warning about limitations" with "this
is bearish data." Amber is the correct semantic for a disclaimer; reserve
red/green strictly for directional bull/bear signals so the color vocabulary
stays unambiguous across the app.

### 3.5 Status strip (market status / VIX / regime bar)

Replaces the inline-styled div in `trading.py` `_render_daytrading()`
(lines 152-160, the `background:#1a1f2e` status bar).

```css
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
```

The current code picks `status_color` in Python (`"#26a69a" if status ==
"MARKET OPEN" else "#f59e0b"`) and inlines it. Change that to select between
the two class names above instead of hex, e.g.
`f'<span class="aeth-status-open">{status}</span>'`.

### 3.6 Sidebar

Replaces `[data-testid="stSidebar"] { background-color: #0a0f1a; }`
(`app.py` line 22).

```css
[data-testid="stSidebar"] {
    background-color: var(--aeth-surface);
    border-right: 1px solid var(--aeth-border);
}

[data-testid="stSidebar"] hr {
    border-color: var(--aeth-border);
}
```

Do not darken the sidebar further than `--aeth-surface` (i.e. do not
`color-mix` it toward black) — in light mode that reintroduces a dark
sidebar against a light main panel, which is the exact "dark-on-light /
light-on-dark seam" bug this spec exists to eliminate. Sidebar and metric
cards should share the same surface tone in both themes.

### 3.7 Page background / root override

Replaces `.main { background-color: #0e1117; }` (`app.py` line 18).

```css
.main {
    background-color: var(--background-color);
}
```

This line is close to redundant once the hardcoded hex is gone (Streamlit
already paints `.main` with the theme background by default) — keep it only
if the implementer finds a specific Streamlit version where `.main` needs an
explicit repaint; otherwise it's safe to delete entirely rather than port it
forward as `var(--background-color)`. Flag this to the implementer as a
"verify, don't blindly port."

---

## 4. Typography & hierarchy (app-wide, not currently defined anywhere)

No font sizes are currently specified outside Plotly chart configs and one
inline `font-size:1.15em` in `_signal_html`. Add these baseline rules to the
same global stylesheet so heading hierarchy is consistent across all six
pages instead of relying on Streamlit's defaults + ad hoc inline sizing:

```css
h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 1.375rem; font-weight: 700; }
h3 { font-size: 1.125rem; font-weight: 600; }

/* st.caption renders as small/muted already; reinforce the muted token
   rather than leaving it to Streamlit's own default gray, which does not
   reliably meet 4.5:1 contrast in light mode */
[data-testid="stCaptionContainer"] {
    color: var(--aeth-text-muted);
}
```

Remove the single hardcoded `font-size:1.15em` in `_signal_html()`
(`trading.py` line 109) — that's now covered by `.signal-card .signal-value`
(section 3.2) at `1.1rem`/600 weight.

---

## 5. What NOT to touch in this pass

- **Plotly chart colors** (`#26a69a`, `#ef5350`, `#2196f3`, `#9c27b0`,
  `#ff9800`, etc. across `trading.py`, `research.py`, `home.py`,
  `portfolio.py`) are Plotly trace/marker colors, not CSS — they render
  inside an SVG/canvas Plotly owns, not the Streamlit DOM, so CSS variables
  don't reach them. They also currently hardcode `template="plotly_dark"`
  everywhere, which is a separate, larger problem (charts will stay
  dark-styled even after this CSS spec ships and the user switches to light
  mode). That's a follow-up: pass `template="plotly_dark" if
  st.context.theme.type == "dark" else "plotly_white"` (or equivalent) at
  each `fig.update_layout()` call site. Out of scope for this spec, but the
  UI refresh isn't complete without it — flag it to whoever picks this up
  next.
- **research.py, home.py, portfolio.py, screener.py, watchlist.py** page
  bodies need no changes for light/dark correctness — they use only
  Streamlit-native widgets (`st.metric`, `st.success/warning/error/info`,
  `st.container(border=True)`, markdown color spans like `:green[...]`),
  which are already theme-aware. They should, however, adopt `.aeth-badge`
  (3.3) where they currently hand-roll emoji-prefixed metric labels
  (e.g. research.py's `score_color()` traffic-light scorecard,
  portfolio.py's ⭐ conviction rating) if the implementer wants full visual
  consistency — optional, not required for the light-mode bug fix.

---

## 6. Accessibility checklist for the implementer

- `--aeth-bull` / `--aeth-bear` against `--aeth-bull-bg` / `--aeth-bear-bg`
  must be verified at ≥ 4.5:1 in both themes after `color-mix()` resolves —
  spot-check with browser devtools contrast checker once implemented, since
  `color-mix()` output can't be hand-calculated reliably.
- Every bull/bear color pairing must remain distinguishable without color
  alone (icons `▲`/`▼`/`◆` already do this in `SIGNAL_CONFIG` — keep them
  when migrating to badges, don't strip to color-only).
- Hover states (3.1 metric card) must not be the only affordance for
  interactivity on non-interactive cards — metric cards aren't clickable, so
  the hover lift is decorative only; don't add hover-only information.
- Focus outlines: Streamlit's native widgets already carry visible focus
  rings via `--primary-color`; nothing in this spec overrides that, and
  nothing added here should either.
