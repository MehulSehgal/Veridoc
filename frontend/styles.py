"""
Visual identity for ArgusRAG's Streamlit UI.

Design concept: a research investigator's case board. The agent is
literally gathering evidence, cross-examining it, and reaching a verdict —
so the interface borrows from a case file rather than a generic chat
widget: a paper-toned evidence card, a connected trace timeline instead of
a flat log, and a verdict ring instead of a plain confidence number.

Palette
  bg        #14181C  deep charcoal, cool undertone (not pure black)
  panel     #1B2127  raised panel surface
  paper     #EDE6D6  aged-paper tone, used only for evidence cards
  ink       #4C8577  muted teal — primary accent, used sparingly
  rust      #B5541F  warm rust — reserved for low-confidence / retry states
  text      #E7E2D3  primary text on dark surfaces
  text-dim  #8B9198  secondary/muted text

Type
  Display: "Fraunces" (serif, editorial, has real texture at large sizes)
  UI/body: "IBM Plex Sans"
  Data/tags: "IBM Plex Mono" — used only for scores, sources, page refs
"""

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #14181C;
    --panel: #1B2127;
    --panel-border: #2A323A;
    --paper: #EDE6D6;
    --paper-text: #2B2620;
    --ink: #4C8577;
    --ink-soft: #6BA093;
    --rust: #B5541F;
    --text: #E7E2D3;
    --text-dim: #8B9198;
}

/* base */
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--bg);
}
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
}

/* headline */
.case-title {
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 2.4rem;
    letter-spacing: -0.01em;
    color: var(--text);
    margin-bottom: 0.1rem;
    line-height: 1.1;
}
.case-subtitle {
    color: var(--text-dim);
    font-size: 0.95rem;
    max-width: 62ch;
    margin-bottom: 1.6rem;
}

/* sidebar case-file label */
.sidebar-label {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    color: var(--text);
    border-bottom: 1px solid var(--panel-border);
    padding-bottom: 0.5rem;
    margin-bottom: 0.9rem;
}

/* question input row */
div[data-testid="stTextInput"] input {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    color: var(--text);
    border-radius: 6px;
    font-size: 1.02rem;
    padding: 0.65rem 0.9rem;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--ink);
    box-shadow: 0 0 0 1px var(--ink);
}

/* primary button */
.stButton button {
    background: var(--ink);
    color: #0F1512;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    padding: 0.55rem 1.3rem;
    transition: background 0.15s ease;
}
.stButton button:hover {
    background: var(--ink-soft);
    color: #0F1512;
}

/* panel headings for each column */
.panel-heading {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    color: var(--text);
    margin-bottom: 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--panel-border);
}

/* ===== trace timeline ===== */
.trace-list {
    position: relative;
    padding-left: 1.4rem;
    border-left: 2px solid var(--panel-border);
}
.trace-step {
    position: relative;
    margin-bottom: 1.15rem;
    padding-left: 0.9rem;
}
.trace-step::before {
    content: '';
    position: absolute;
    left: -1.62rem;
    top: 0.35rem;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--panel);
    border: 2px solid var(--ink);
}
.trace-step.retry::before { border-color: var(--rust); }
.trace-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-soft);
    text-transform: none;
    letter-spacing: 0.02em;
    margin-bottom: 0.15rem;
}
.trace-step.retry .trace-label { color: var(--rust); }
.trace-content {
    font-size: 0.92rem;
    color: var(--text);
    line-height: 1.5;
    white-space: pre-wrap;
}

/* ===== evidence cards (paper tone) ===== */
.evidence-card {
    background: var(--paper);
    color: var(--paper-text);
    border-radius: 3px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.28);
    transform: rotate(-0.2deg);
}
.evidence-source {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #6B6152;
    margin-bottom: 0.3rem;
}
.evidence-text {
    font-size: 0.87rem;
    line-height: 1.45;
}

.figure-card {
    background: var(--paper);
    border-radius: 3px;
    padding: 0.5rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.28);
}
.figure-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #6B6152;
    margin-top: 0.35rem;
    text-align: center;
}

/* ===== verdict / confidence ring ===== */
.verdict-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 1.3rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1.3rem;
}
.verdict-text {
    font-size: 1.02rem;
    line-height: 1.55;
    color: var(--text);
}
.verdict-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: 0.5rem;
}

/* misc */
hr { border-color: var(--panel-border); }
[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 6px; }
</style>
"""


def confidence_ring_svg(confidence: float, size: int = 88) -> str:
    """A small ring gauge: filled arc proportional to confidence, rust below
    threshold, ink teal above it — same visual language as the trace dots."""
    radius = size / 2 - 6
    circumference = 2 * 3.14159 * radius
    filled = circumference * max(0.0, min(confidence, 1.0))
    color = "#4C8577" if confidence >= 0.6 else "#B5541F"
    center = size / 2
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
                stroke="#2A323A" stroke-width="6"/>
        <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
                stroke="{color}" stroke-width="6"
                stroke-dasharray="{filled:.1f} {circumference:.1f}"
                stroke-linecap="round"
                transform="rotate(-90 {center} {center})"/>
        <text x="{center}" y="{center + 5}" text-anchor="middle"
              font-family="IBM Plex Mono, monospace" font-size="15"
              fill="{color}">{int(confidence * 100)}%</text>
    </svg>
    """
