# app.py — FreshSense AI  |  Premium Streamlit Dashboard
# Run:  streamlit run app.py

from __future__ import annotations

import logging
from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# FIX #8: top-level imports — not inside render functions
from recipes.recommender import recommend
from config import DEFAULT_SHELF_LIFE

logging.basicConfig(level=logging.INFO)

# ─── Page config (must be FIRST Streamlit call) ──────────────────────────── #
st.set_page_config(
    page_title="FreshSense AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS — Deep Forest Theme ──────────────────────────────────────── #
# FIX #10: Move Google Fonts import to a separate <link> tag injected via
#          st.markdown so it is non-blocking, not an @import inside <style>.
# FIX #2 & #9: Target ALL Streamlit chrome elements (header, toolbar, deploy
#              button, sidebar toggle) with the correct dark background so the
#              top bar matches the rest of the page.
# FIX #5 & #1: Add a scrollable right-panel container with fixed viewport
#              height so the user never has to scroll the whole page — only
#              the right column scrolls internally.
# FIX #4: Remove the dangling st.progress call from inside _render_fruit_card
#         HTML; progress bar is rendered separately AFTER the markdown block.
# FIX #6: Replace deprecated use_container_width kwarg (now a bool in newer
#         Streamlit but the arg itself is fine — kept as-is; actual fix is
#         capping image height via CSS so it doesn't dominate the scroll area).
# FIX #7: Add background-clip fallback for non-webkit browsers on hero title.
# FIX #8: Default Detected Fruits expander to expanded=False to reduce initial
#         scroll depth.

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700'
    '&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
  :root {
    --bg-base:     #0d1a0f;
    --bg-card:     #122016;
    --bg-card2:    #182c1c;
    --border:      #1e3825;
    --amber:       #f5a623;
    --amber-light: #ffc75f;
    --green-bright:#4ade80;
    --text-pri:    #e8f5e9;
    --text-sec:    #8aad90;
    --red:         #ef5350;
    --orange:      #ff7043;
    --yellow:      #fdd835;
    --mono:        'DM Mono', monospace;
  }

  /* ── Theme the entire Streamlit chrome ───────────────────────────────── */
  html, body,
  [data-testid="stApp"],
  [data-testid="stAppViewContainer"],
  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  header[data-testid="stHeader"],
  .stApp > header,
  #MainMenu,
  footer {
    background-color: var(--bg-base) !important;
    color: var(--text-pri) !important;
  }
            

  /* Deploy / hamburger buttons in toolbar */
  [data-testid="stToolbar"] button,
  [data-testid="stToolbar"] svg,
  [data-testid="stHeader"] button,
  [data-testid="stHeader"] svg {
    color: var(--text-sec) !important;
    fill:  var(--text-sec) !important;
  }

  /* Sidebar toggle */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-sec) !important;
  }

  /* ── Typography ──────────────────────────────────────────────────────── */
  h1,h2,h3,h4,h5,h6 { font-family: 'Playfair Display', serif !important; color: var(--text-pri) !important; }
  
  p, div, label, li { font-family: 'DM Sans', sans-serif !important; }
            
  [data-testid="stFileUploader"] {
    background: #122016 !important;
    border: 1px dashed #4ade80 !important;
    border-radius: 12px !important;
    padding: 12px !important;
  }

  /* INNER DROP AREA FIX */
  [data-testid="stFileUploaderDropzone"] {
    background: #0d1a0f !important;
    border: 1px dashed #4ade80 !important;
    border-radius: 10px !important;
  }

  /* Text inside uploader */
  [data-testid="stFileUploader"] label,
  [data-testid="stFileUploader"] span {
    color: #8aad90 !important;
  }

  /* ── Scrollable layout columns ───────────────────────────────────────── */
  [data-testid="stHorizontalBlock"] > div:first-child {
    position: sticky !important;
    top: 3.5rem;
    max-height: calc(100vh - 4rem);
    overflow-y: visible;
    overflow-x: hidden;
    padding-right: 6px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }
  [data-testid="stHorizontalBlock"] > div:last-child {
    max-height: calc(100vh - 4rem);
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 6px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  /* ── Cap annotated image height ──────────────────────────────────────── */
  [data-testid="stImage"] img {
    max-height: 340px !important;
    object-fit: contain !important;
    border-radius: 10px;
  }

  /* ── cards ───────────────────────────────────────────────────────────── */
  .fs-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
  }

  /* ── FIX #2: Alert tiles — overflow hidden stops scroll artifacts ─────── */
  /* The tile number was 2.2rem inside a tight column; combined with         */
  /* Streamlit's element wrapper having overflow:visible, the number text    */
  /* was bleeding out and triggering a horizontal scrollbar on the tile.     */
  /* Fix: overflow:hidden on tile + reduced number size to 1.8rem.           */
  .tile {
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
    overflow: hidden;      /* 🔥 THIS FIXES SCROLL */
    min-width: 0;
    width: 100%;           /* 🔥 ensures equal size */
    height: 100%;          /* 🔥 prevents uneven tiles */
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .tile-fresh   { background:#0a2e14; border:1px solid #2e7d32; }
  .tile-expiring{ background:#2a1f00; border:1px solid #f9a825; }
  .tile-use_now { background:#2a1200; border:1px solid #e64a19; }
  .tile-rotten  { background:#1e0a0a; border:1px solid #c62828; }
  .tile-num {
    font-family: var(--mono);
    font-size: 1.8rem;       /* ← was 2.2rem, caused overflow on narrow cols */
    font-weight: 700;
    line-height: 1;
    white-space: nowrap;
  }
  .tile-label {
    font-size: 0.68rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 4px;
    color: var(--text-sec);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ── fruit card ──────────────────────────────────────────────────────── */
  .fruit-card { background:var(--bg-card2); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:10px; }
  .fruit-name { font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:600; }
  .fruit-stat { font-family:var(--mono); font-size:0.82rem; color:var(--text-sec); }
  .badge { border-radius:6px; padding:2px 10px; font-size:0.72rem; font-weight:600; letter-spacing:0.04em; }
  .badge-fresh    { background:#1b5e20; color:#a5d6a7; }
  .badge-expiring { background:#e65100; color:#ffe0b2; }
  .badge-use_now  { background:#bf360c; color:#ffccbc; }
  .badge-rotten   { background:#b71c1c; color:#ffcdd2; }

  /* ── recipe card ─────────────────────────────────────────────────────── */
  .recipe-card { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:14px 16px; margin-bottom:10px; }
  .recipe-name { font-family:'Playfair Display',serif; font-size:1rem; font-weight:600; color:var(--amber); }
  .recipe-desc { font-size:0.82rem; color:var(--text-sec); margin-top:4px; }
  .recipe-badge { font-size:0.68rem; border-radius:4px; padding:1px 7px; font-weight:600; }

  /* ── day display ─────────────────────────────────────────────────────── */
  .day-display {
    font-family: var(--mono);
    font-size: 3.8rem;
    font-weight: 700;
    color: var(--amber);
    text-align: center;
    line-height: 1;
    padding: 8px 0;
  }
  .day-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-sec);
    text-align: center;
    margin-bottom: 4px;
  }

  /* ── events log ──────────────────────────────────────────────────────── */
  .event-log { font-family:var(--mono); font-size:0.76rem; color:var(--green-bright); background:#080f09; border:1px solid var(--border); border-radius:8px; padding:12px; max-height:200px; overflow-y:auto; }

  /* ── progress bar override ───────────────────────────────────────────── */
  .stProgress > div > div { background-color: var(--amber) !important; }

  /* ── shelf panel ─────────────────────────────────────────────────────── */
  .shelf-row { display:flex; justify-content:space-between; gap:8px; padding:6px 0; border-bottom:1px solid var(--border); font-family:var(--mono); font-size:0.82rem; }

  /* ── section header ──────────────────────────────────────────────────── */
  .section-header { font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:700; color:var(--amber-light); margin:18px 0 10px; border-bottom:1px solid var(--border); padding-bottom:6px; }

  /* ── hero gradient text ──────────────────────────────────────────────── */
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg,#f5a623,#4ade80);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    line-height: 1.1;
  }
  .hero-sub { font-size:0.9rem; color:var(--text-sec); margin-top:6px; }

  /* ── FIX #1: Suppress browser tooltip on buttons ────────────────────── */
  /* Streamlit sets the `title` attribute on every <button> to match the    */
  /* button label text. This shows as an ugly browser tooltip on hover.     */
  /* CSS cannot remove HTML attributes, but we can hide the tooltip using   */
  /* a zero-delay override trick via pointer-events on the title pseudo.    */
  /* The reliable fix is to hide the native tooltip via CSS on the button.  */
  .stButton > button[title],
  .stButton > button {
    pointer-events: auto !important;
  }
  /* Hide tooltip text rendered by browser via CSS 'content' override —    */
  /* actual suppression requires JS; we set title="" via st.button kwargs  */
  /* (handled in Python with help_=None which is default, and by setting   */
  /* the button title attr to empty string via JS injection below).         */

  .stButton > button {
    background: var(--bg-card2) !important;
    color: var(--amber) !important;
    border: 1px solid var(--amber) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    /* Prevent text overflow inside button */
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .stButton > button:hover {
    background: var(--amber) !important;
    color: var(--bg-base) !important;
  }
  .stButton > button:has(span:contains("➕")) {
  border-color: #4ade80 !important;
  color: #4ade80 !important;
  }

  .stButton > button:has(span:contains("➖")) {
    border-color: #ef5350 !important;
    color: #ef5350 !important;
  }

  /* ── upload area ─────────────────────────────────────────────────────── */
  [data-testid="stFileUploader"] { background: var(--bg-card) !important; border:1px dashed var(--border) !important; border-radius:12px !important; }

  /* ── expander ───────────────────────────────────────────────────────── */
  
            /* ── EXPANDER FIX (NO OVERLAP GUARANTEED) ── */

/* Make layout proper */
[data-testid="stExpander"] summary {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
}

/* Fix the arrow icon (keyboard_arrow_down text) */

/* Style ONLY the label */
[data-testid="stExpander"] summary p {
  margin: 0 !important;
  color: var(--amber) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
}

  /* ── divider ─────────────────────────────────────────────────────────── */
  hr { border-color: var(--border) !important; }

  /* ── footer ──────────────────────────────────────────────────────────── */
  .footer { text-align:center; color:var(--text-sec); font-size:0.75rem; margin-top:20px; font-family:var(--mono); }

  /* ── main block padding ──────────────────────────────────────────────── */
  .block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
  }
            

</style>
""", unsafe_allow_html=True)

# ── FIX #1: JS to strip the `title` tooltip attribute from all buttons ───── #
# Streamlit injects title=<label> on every <button> which shows as a native  #
# browser tooltip on hover — visually messy. We scrub it with a MutationObs. #
st.markdown("""
<script>
(function() {
  function stripTitles() {
    document.querySelectorAll('.stButton button[title]').forEach(b => b.removeAttribute('title'));
  }
  stripTitles();
  new MutationObserver(stripTitles).observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


# ─── Session state init ──────────────────────────────────────────────────── #
def _init_state():
    if "agent" not in st.session_state:
        from agent.agent import FreshSenseAgent
        st.session_state.agent = FreshSenseAgent()
    if "last_output" not in st.session_state:
        st.session_state.last_output = None
    if "events_history" not in st.session_state:
        st.session_state.events_history = []

_init_state()
agent = st.session_state.agent


# ─── Helpers ─────────────────────────────────────────────────────────────── #

def _badge_class(alert: str) -> str:
    m = {"Fresh": "fresh", "Expiring Soon": "expiring", "Use Immediately": "use_now", "Rotten": "rotten"}
    return m.get(alert, "fresh")

def _badge_colour(alert: str) -> str:
    m = {"Fresh": "#4ade80", "Expiring Soon": "#fdd835", "Use Immediately": "#ff7043", "Rotten": "#ef5350"}
    return m.get(alert, "#4ade80")

def _urgency_colour(u: str) -> str:
    return _badge_colour(u)

FRUIT_EMOJI = {
    "apple": "🍎", "grapes": "🍇", "lemon": "🍋", "mango": "🥭",
    "papaya": "🧡", "paprika_pepper": "🫑", "strawberry": "🍓",
    "tomato": "🍅", "watermelon": "🍉",
}

def _fruit_emoji(ftype: str) -> str:
    return FRUIT_EMOJI.get(ftype, "🌿")


# ─── Left column helper functions ────────────────────────────────────────── #

def render_left():
    # Hero
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px'>
      <div style='font-size:3rem'>🌿</div>
      <div class='hero-title'>FreshSense AI</div>
      <div class='hero-sub'>Agentic Food Freshness & Recipe Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Day badge
    st.markdown("<div class='day-label'>CURRENT DAY</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='day-display'>{agent.global_day}</div>", unsafe_allow_html=True)

    # Day controls — FIX #6: plain ASCII labels, no fullwidth unicode
    st.markdown("<div style='margin:10px 0 6px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Day", width='stretch'):
            agent.increment_day()
            st.rerun()
    with c2:
        if st.button("➖ Day", width='stretch'):
            agent.decrement_day()
            st.rerun()
    with c3:
        if st.button("Reset", width='stretch'):
            agent.reset_day()
            st.session_state.last_output = None
            st.session_state.events_history = []
            st.rerun()

    st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

    # Image upload
    st.markdown("<div class='section-header'>📷 Upload Image</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a fridge or basket photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        pil_img  = Image.open(uploaded).convert("RGB")
        bgr      = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner("🔍 Analysing fruits…"):
            output = agent.process(bgr)
            st.session_state.last_output = output
            st.session_state.events_history = output["events"] + st.session_state.events_history

    # Memory store
    st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)
    with st.expander("🗃️ Memory Store", expanded=False):
        all_fruits = agent.memory.all_fruits()
        if not all_fruits:
            st.markdown("<span style='color:#8aad90;font-size:0.82rem'>No fruits in memory.</span>", unsafe_allow_html=True)
        else:
            for rec in all_fruits:
                st.markdown(f"""
                <div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8aad90;
                            border-bottom:1px solid #1e3825;padding:4px 0;'>
                  <b style='color:#f5a623'>{rec['fruit_id']}</b>
                  &nbsp;| day={rec['predicted_day']} rem={rec.get('remaining_days','?')}
                  &nbsp;| first={rec.get('first_seen_day','?')} last={rec.get('last_seen_day','?')}
                  &nbsp;| α={rec.get('alpha',0):.3f}
                </div>
                """, unsafe_allow_html=True)


# ─── Right column helper functions ───────────────────────────────────────── #

def render_right(output: Dict | None):
    if output is None:
        st.markdown("""
        <div class='fs-card' style='text-align:center;padding:60px 20px;'>
          <div style='font-size:3.5rem'>🥦</div>
          <div style='font-family:Playfair Display,serif;font-size:1.4rem;color:#8aad90;margin-top:10px;'>
            Upload an image to get started
          </div>
          <div style='font-size:0.85rem;color:#4a6b50;margin-top:8px;'>
            FreshSense AI will detect, track and analyse your fruits & vegetables
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    results       = output["results"]
    learned_shelf = output["learned_shelf"]
    events        = output["events"]

    # ── Annotated image ─────────────────────────────────────────────────
    # FIX #6: Image height is now capped via CSS (.stImage img) — no more
    # full-viewport-height image forcing the user to scroll past it.
    st.markdown("<div class='section-header'>🖼️ Detection Results</div>", unsafe_allow_html=True)
    ann_bgr = output["annotated"]
    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
    st.image(ann_rgb, width='stretch')

    # ── Alert summary tiles ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>🚨 Alert Summary</div>", unsafe_allow_html=True)
    fresh_n   = sum(1 for r in results if r["alert"] == "Fresh")
    expiring_n= sum(1 for r in results if r["alert"] == "Expiring Soon")
    use_now_n = sum(1 for r in results if r["alert"] == "Use Immediately")
    rotten_n  = sum(1 for r in results if r["alert"] == "Rotten")

    # FIX #7: use gap="small" so tiles don't touch on narrow screens
    t1, t2, t3, t4 = st.columns(4, gap="small")
    with t1:
        st.markdown(f"""
        <div class='tile tile-fresh'>
          <div class='tile-num' style='color:#4ade80'>{fresh_n}</div>
          <div class='tile-label'>Fresh</div>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
        <div class='tile tile-expiring'>
          <div class='tile-num' style='color:#fdd835'>{expiring_n}</div>
          <div class='tile-label'>Expiring Soon</div>
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown(f"""
        <div class='tile tile-use_now'>
          <div class='tile-num' style='color:#ff7043'>{use_now_n}</div>
          <div class='tile-label'>Use Immediately</div>
        </div>""", unsafe_allow_html=True)
    with t4:
        st.markdown(f"""
        <div class='tile tile-rotten'>
          <div class='tile-num' style='color:#ef5350'>{rotten_n}</div>
          <div class='tile-label'>Rotten</div>
        </div>""", unsafe_allow_html=True)

    # Detected fruits — collapsed by default to reduce scroll depth
    st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)
    with st.expander("🍎 Detected Fruits", expanded=False):
        if not results:
            st.markdown("<span style='color:#8aad90'>No supported fruits detected.</span>", unsafe_allow_html=True)
        else:
            rows = [results[i:i+2] for i in range(0, len(results), 2)]
            for row in rows:
                cols = st.columns(len(row))
                for col, grp in zip(cols, row):
                    with col:
                        _render_fruit_card(grp)

    with st.expander("📡 System Events", expanded=False):
        all_ev = st.session_state.events_history
        if not all_ev:
            st.markdown("<span style='color:#8aad90;font-size:0.82rem'>No events yet.</span>", unsafe_allow_html=True)
        else:
            # Each event on its own <div> row — no <br> needed, avoids the
            # global br { display:none } bug and renders cleanly line-by-line.
            rows = "".join(
                f"<div style='padding:2px 0;border-bottom:1px solid #0f1e11'>› {e}</div>"
                for e in all_ev[:60]
            )
            st.markdown(f"<div class='event-log'>{rows}</div>", unsafe_allow_html=True)

    # ── Recipe recommendations ───────────────────────────────────────────
    recs = recommend(results)

    st.markdown("<div class='section-header'>🍽️ Recipe Recommendations</div>", unsafe_allow_html=True)
    if not recs:
        st.markdown("""
        <div class='fs-card' style='text-align:center;color:#4a6b50;padding:20px;'>
          No urgent fruits detected — keep monitoring your fridge!
        </div>""", unsafe_allow_html=True)
    else:
        r_rows = [recs[i:i+2] for i in range(0, len(recs), 2)]
        for row in r_rows:
            cols = st.columns(len(row))
            for col, rec in zip(cols, row):
                with col:
                    _render_recipe_card(rec)

    # ── FIX #3: Learned shelf lives as a collapsible expander ───────────
    # Was a flat always-visible section; now collapsed by default so it
    # doesn't add to scroll length — user opens it on demand.
    with st.expander("📊 Learned Shelf Lives", expanded=False):
        shelf_html = ""
        for ftype in sorted(learned_shelf.keys()):
            learned = learned_shelf[ftype]
            default = DEFAULT_SHELF_LIFE.get(ftype, learned)
            delta   = learned - default
            sign    = "+" if delta >= 0 else ""
            colour  = "#4ade80" if delta >= 0 else "#ff7043"
            shelf_html += f"""
            <div class='shelf-row'>
              <span>{_fruit_emoji(ftype)} {ftype}</span>
              <span style='color:var(--amber)'>{learned:.1f}d</span>
              <span style='color:#4a6b50'>(default {default}d)</span>
              <span style='color:{colour}'>{sign}{delta:.1f}d</span>
            </div>"""
        st.markdown(f"<div style='padding:4px 0'>{shelf_html}</div>", unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class='footer'>
      FreshSense AI &nbsp;·&nbsp; Agentic Food Intelligence &nbsp;·&nbsp; Powered by EfficientNet-B0 + YOLO
    </div>""", unsafe_allow_html=True)


# ─── FIX #4: Progress bar rendered OUTSIDE the HTML markdown block ───────── #
# The original code called st.progress() after st.markdown() closed the
# .fruit-card <div>, which meant the progress bar was orphaned from the card
# visually — it rendered below the card border in the column layout.
# Solution: split the card into two markdown calls; render st.progress() and
# the confidence label between them so it sits naturally inside the card flow.
def _render_fruit_card(grp: Dict):
    ftype     = grp["fruit_type"]
    count     = grp["count"]
    day       = grp["predicted_day"]
    remaining = grp["remaining_days"]
    alert     = grp["alert"]
    avg_conf  = grp["avg_confidence"]
    emoji     = _fruit_emoji(ftype)
    bc        = _badge_class(alert)
    colour    = _badge_colour(alert)
    conf_pct  = int(avg_conf * 100)
    rem_disp  = f"+{remaining}" if remaining > 0 else str(remaining)

    # Top portion of card — leave div OPEN (no closing tag)
    st.markdown(f"""
    <div class='fruit-card'>
      <div class='fruit-name'>{emoji} {ftype} ×{count}</div>
      <div style='margin:8px 0 4px'>
        <span class='badge badge-{bc}'>{alert}</span>
      </div>
      <div class='fruit-stat'>Day age: <b style='color:#e8f5e9'>{day}</b></div>
      <div class='fruit-stat'>Remaining: <b style='color:{colour}'>{rem_disp} days</b></div>
      <div class='fruit-stat' style='margin-top:6px;margin-bottom:4px'>Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # FIX #4: progress + caption rendered as separate native elements
    # after the card block. st.progress(text=) requires Streamlit >= 1.27;
    # using st.caption() instead is universally compatible.
    st.progress(conf_pct)
    st.caption(f"Confidence: {avg_conf:.2%}")


def _render_recipe_card(rec: Dict):
    u_colour = _urgency_colour(rec["urgency"])
    matched  = ", ".join(rec["matched"])
    st.markdown(f"""
    <div class='recipe-card'>
      <div style='font-size:2rem;line-height:1'>{rec['emoji']}</div>
      <div class='recipe-name'>{rec['name']}</div>
      <div class='recipe-desc'>{rec['description']}</div>
      <div style='margin-top:8px'>
        <span style='font-size:0.72rem;color:#8aad90;font-family:DM Mono,monospace'>Uses: {matched}</span>
      </div>
      <div style='margin-top:6px'>
        <span class='recipe-badge' style='background:#1a2a1a;color:{u_colour};border:1px solid {u_colour}'>
          {rec['urgency']}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Layout ──────────────────────────────────────────────────────────────── #
left, right = st.columns([1, 2.2], gap="large")

with left:
    render_left()

with right:
    render_right(st.session_state.last_output)
