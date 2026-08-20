"""
app.py
Streamlit demo app for the QML Invoice Risk Classification pipeline.
Upload an invoice PDF (or pick a sample) and see it scored live by
Logistic Regression, RBF-SVM, and a quantum kernel SVM — then routed for
approval, queued for review, and recorded in a persistent audit trail.

Run locally with:  streamlit run src/app.py
"""

import json
import os
import sys
import types
import numpy as np
import pandas as pd
import streamlit as st
import pennylane as qml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_utils import extract_fields_from_pdf, compute_features_for_new_invoice, FEATURE_COLS
import approval_rules
import audit_store
import confidence as conf
import dashboard
import notifications as notif
import retrain
from approval_rules import ROUTE_META, ROUTE_ORDER
from audit_store import STATUS_META, OPEN_STATUSES, WorkflowError
from pipeline_utils import assess_duplicate

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
GT_PATH = os.path.join(BASE_DIR, "data", "ground_truth.csv")
EXTRACTED_PATH = os.path.join(BASE_DIR, "data", "processed", "extracted_fields.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
INVOICES_DIR = os.path.join(BASE_DIR, "data", "invoices")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics_comparison.csv")

N_QUBITS = 5

st.set_page_config(page_title="Ledger: QML Invoice Risk", page_icon="◈", layout="wide")

# ---------------------------------------------------------------------------
# Design system: "Audit Ledger meets Quantum Circuit"
# graphite surfaces, amber for risk, teal for clean, mono for data
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

:root {
    /* Base surfaces — light theme. Elevation still reads as "lighter":
       bg-0 is the page, bg-1 a raised card, bg-2 the most-elevated chip. */
    --bg-0: #E8E8E8;
    --bg-1: #F5F5F3;
    --bg-2: #FFFFFF;
    --border: #D3D3CE;
    --text-0: #111111;
    --text-1: #5C5C57;
    --ink: #0A0A0A;

    /* Risk-semantic accents. Reserved for actual risk badges/bands, plus
       a small set of audit-status uses (chain verification, rule flags)
       that share the same domain. Everything else uses the panel accents
       below. The base hues are tuned as bright accents (dots, glows,
       borders, dim tint backgrounds); they don't have enough contrast to
       use directly as text on this light page, so each has a darker
       "ink" twin below, used only where the color carries running text. */
    --amber: #E8A33D;
    --amber-dim: rgba(232, 163, 61, 0.14);
    --amber-ink: #8F5C11;
    --teal: #2DD4BF;
    --teal-dim: rgba(45, 212, 191, 0.14);
    --teal-ink: #177065;
    --violet: #8C7CF0;
    --violet-dim: rgba(140, 124, 240, 0.14);
    --violet-ink: #604AEA;
    --rose: #F2647B;
    --rose-dim: rgba(242, 100, 123, 0.14);
    --rose-ink: #CB112F;

    /* Landing-page panel accents, solid full-block fills, not tints. */
    --panel-yellow: #F5F566;
    --panel-orange: #D97F3D;
    --panel-blue: #A9CAD9;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-0); }

/* Streamlit renders its own default text (tab labels, widget labels,
   captions, checkbox/radio text, the file uploader's instructions) using
   a color it picks from the visitor's OS/browser color-scheme preference,
   independently of the page background we set above. A visitor whose
   system is in dark mode gets Streamlit's dark-theme text color, near
   white, sitting on this light page, unreadable, even though nothing
   here has a dark background anymore. These elements are pinned to the
   app's own text colors explicitly so legibility doesn't depend on the
   visitor's OS setting. Custom HTML rendered through render_html()/
   st.markdown() never uses a bare <p>, so this can't clobber it. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stTab"] p,
[data-testid="stCheckbox"] p,
[data-testid="stRadio"] p {
    color: var(--text-0) !important;
}
[data-testid="stCaptionContainer"] p {
    color: var(--text-1) !important;
}
[data-testid="stFileUploaderDropzone"] p {
    color: var(--text-0) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--text-1) !important; opacity: 1 !important;
}
[data-testid="stMultiSelectTagsContainer"] span[data-tag] {
    background: var(--bg-2) !important; border: 1px solid var(--border) !important;
}
[data-testid="stMultiSelectTagsContainer"] span[data-tag] * {
    color: var(--text-0) !important;
}

/* Buttons Streamlit renders with its own native (unstyled-by-us) dark
   pill, like the file uploader's "Browse files" and st.download_button,
   already carry light text meant for that dark fill. The broad text-0
   rule above would otherwise reach their inner label paragraph too and
   force near-black text onto that same dark background. Handing color
   back to `inherit` lets each button's own color, ours or Streamlit's
   native default, apply uncontested. */
button [data-testid="stMarkdownContainer"] p,
[data-testid="stFileUploaderDropzone"] button * {
    color: inherit !important;
}

[data-testid="stSidebar"] { background: var(--bg-1) !important; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text-0) !important; letter-spacing: -0.01em; }

/* Respect reduced-motion preferences before anything else so every rule
   below that adds transition/animation can be neutralized in one place. */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important; }
}

.ledger-sub {
    font-family: 'Inter', sans-serif; font-size: 15px; color: var(--text-1);
    max-width: 720px; margin-top: 12px; line-height: 1.6;
}

/* --- Landing hero (Section 1) -------------------------------------------
   Nav row (logomark + wordmark, plain nav labels, a pill CTA), a huge
   mono display headline sitting tight under it, one line of supporting
   copy with its own CTA, and a full-height split band below: an abstract
   CSS circuit panel standing in for photography, paired with a solid
   panel-accent block carrying a centered mark. No photography, no risk
   colors — this section is pure base palette + panel accents. */
.landing-nav {
    display: flex; justify-content: space-between; align-items: center;
    padding: 4px 2px 8px 2px; flex-wrap: wrap; gap: 14px;
}
.landing-logomark { display: flex; align-items: center; gap: 10px; }
.logomark-shape {
    width: 16px; height: 16px; display: inline-block; border-radius: 3px;
    background: conic-gradient(var(--ink) 0deg 90deg, var(--panel-orange) 90deg 180deg,
                var(--ink) 180deg 270deg, var(--panel-orange) 270deg 360deg);
    transform: rotate(45deg);
}
.logomark-word {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.14em;
    color: var(--text-0); text-transform: uppercase; font-weight: 500;
}
.landing-nav-right { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.landing-nav-links {
    display: flex; gap: 26px; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    letter-spacing: 0.1em; color: var(--text-1); text-transform: uppercase;
}
.landing-nav-cta {
    display: inline-flex; align-items: center; gap: 10px; width: fit-content;
    padding: 8px 8px 8px 20px; border-radius: 100px; background: var(--ink);
    text-decoration: none !important; cursor: pointer; flex-shrink: 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.landing-nav-cta:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.25); }
.landing-nav-cta-text {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 13px; color: #FAFAF8;
}
.landing-nav-cta-icon {
    width: 24px; height: 24px; border-radius: 50%; background: var(--panel-orange);
    display: flex; align-items: center; justify-content: center; color: var(--ink);
    font-size: 12px; font-weight: 700; transition: transform 0.2s ease; flex-shrink: 0;
}
.landing-nav-cta:hover .landing-nav-cta-icon { transform: rotate(45deg); }

.landing-headline {
    font-family: 'Space Mono', 'IBM Plex Mono', monospace; font-weight: 700;
    font-size: 84px; line-height: 0.96; letter-spacing: -0.03em; color: var(--ink);
    text-transform: uppercase; margin-top: -10px;
}

.landing-hero-body {
    display: flex; justify-content: space-between; align-items: flex-end;
    flex-wrap: wrap; gap: 22px; margin-top: 26px; padding-bottom: 30px;
    border-bottom: 1px solid var(--border);
}
.landing-desc {
    font-family: 'Inter', sans-serif; font-size: 16px; color: var(--text-1);
    line-height: 1.6; max-width: 480px;
}
.landing-cta {
    display: inline-flex; align-items: center; gap: 14px; width: fit-content;
    padding: 6px 6px 6px 26px; border-radius: 100px; flex-shrink: 0;
    background: var(--ink); text-decoration: none !important; cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.landing-cta:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(0,0,0,0.30); }
.landing-cta-text {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14px; color: #FAFAF8;
}
.landing-cta-icon {
    width: 34px; height: 34px; border-radius: 50%; background: var(--amber);
    display: flex; align-items: center; justify-content: center; color: var(--ink);
    font-size: 16px; font-weight: 700; transition: transform 0.2s ease; flex-shrink: 0;
}
.landing-cta:hover .landing-cta-icon { transform: rotate(45deg); }

.landing-split {
    display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
    margin-top: 28px; min-height: 420px;
}
.landing-split-panel {
    position: relative; border-radius: 14px; overflow: hidden; border: 1px solid var(--border);
}

/* Left panel: abstract circuit visual — grid lines, a soft glow, a slow
   rotating diamond, accent nodes. Deliberately dark against the light
   page so the split reads as two distinct materials, not two tints of
   the same surface. */
.landing-split-panel--circuit { background: var(--ink); }
.landing-panel-grid {
    position: absolute; inset: 0; opacity: 0.5;
    background-image:
        linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px);
    background-size: 34px 34px;
}
.landing-panel-glow {
    position: absolute; width: 320px; height: 320px; border-radius: 50%; top: -70px; right: -70px;
    background: radial-gradient(circle, rgba(217,127,61,0.45), transparent 70%); filter: blur(8px);
}
.landing-panel-diamond {
    position: absolute; width: 180px; height: 180px; border: 1.5px solid rgba(169,202,217,0.6);
    bottom: 16%; left: 50%; margin-left: -90px; transform: rotate(45deg);
    animation: panelDrift 40s linear infinite;
}
.landing-panel-diamond::before {
    content: ''; position: absolute; inset: 24px; border: 1px solid rgba(245,245,102,0.5);
}
@keyframes panelDrift { to { transform: rotate(405deg); } }
.landing-panel-node {
    position: absolute; width: 9px; height: 9px; border-radius: 50%;
    background: var(--panel-orange); box-shadow: 0 0 12px var(--panel-orange);
}
.landing-panel-node--1 { top: 20%; left: 16%; }
.landing-panel-node--2 { top: 64%; left: 74%; background: var(--panel-blue); box-shadow: 0 0 12px var(--panel-blue); }
.landing-panel-node--3 { top: 80%; left: 24%; background: var(--panel-yellow); box-shadow: 0 0 12px var(--panel-yellow); }

/* Right panel: solid yellow block, centered mark, thin divider, caption. */
.landing-split-panel--yellow {
    background: var(--panel-yellow); display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 18px; padding: 24px;
}
.split-mark {
    font-family: 'Space Grotesk', sans-serif; font-size: 64px; line-height: 1; color: var(--ink);
}
.split-divider { width: 64px; height: 1.5px; background: var(--ink); opacity: 0.55; }
.split-caption {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--ink); opacity: 0.75; text-align: center;
}

@media (max-width: 900px) {
    .landing-headline { font-size: 42px; }
    .landing-split { grid-template-columns: 1fr; min-height: auto; }
    .landing-split-panel { min-height: 260px; }
}

/* --- Capabilities section (Section 2) ------------------------------------
   Bracketed label, a large mono headline, one line of copy, and a
   three-column grid of numbered blocks — real phases of the pipeline,
   not marketing copy. */
.cap-section { padding: 64px 0 12px 0; }
.cap-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.14em;
    color: var(--text-1); text-transform: uppercase; margin-bottom: 18px;
}
.cap-headline {
    font-family: 'Space Mono', 'IBM Plex Mono', monospace; font-weight: 700;
    font-size: 48px; line-height: 1.05; letter-spacing: -0.02em; color: var(--ink);
    text-transform: uppercase;
}
.cap-desc {
    font-family: 'Inter', sans-serif; font-size: 15px; color: var(--text-1);
    max-width: 520px; margin-top: 16px; line-height: 1.6;
}
.cap-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 32px; margin-top: 44px;
}
.cap-col-divider { height: 1.5px; background: var(--ink); opacity: 0.5; margin-bottom: 22px; }
.cap-num {
    font-family: 'Space Mono', 'IBM Plex Mono', monospace; font-weight: 700; font-size: 46px;
    line-height: 1; color: transparent; -webkit-text-stroke: 1.5px var(--ink); text-stroke: 1.5px var(--ink);
}
.cap-name {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 17px;
    letter-spacing: -0.01em; color: var(--ink); text-transform: uppercase; margin-top: 16px;
}
.cap-col-desc {
    font-family: 'Inter', sans-serif; font-size: 14px; color: var(--text-1);
    line-height: 1.6; margin-top: 10px;
}

@media (max-width: 900px) {
    .cap-grid { grid-template-columns: 1fr; gap: 36px; }
    .cap-headline { font-size: 32px; }
}

.circuit-divider {
    height: 20px; margin: 22px 0 26px 0; position: relative;
    background-image: repeating-linear-gradient(90deg, var(--border) 0px, var(--border) 6px, transparent 6px, transparent 14px);
    background-position: center; background-size: 100% 1px; background-repeat: no-repeat;
}
.circuit-divider::before, .circuit-divider::after {
    content: ''; position: absolute; top: 50%; width: 6px; height: 6px; border-radius: 50%;
    background: var(--teal); transform: translateY(-50%); box-shadow: 0 0 8px var(--teal);
}
.circuit-divider::before { left: 0; }
.circuit-divider::after { right: 0; background: var(--amber); box-shadow: 0 0 8px var(--amber); }

[data-testid="stTab"] {
    font-family: 'Space Grotesk', sans-serif !important; font-size: 14px !important;
    transition: color 0.2s ease !important;
}
[data-testid="stTab"] p { font-size: 14px !important; transition: color 0.2s ease !important; }
[data-testid="stTab"]:hover p { color: var(--ink) !important; }
[data-testid="stTab"][aria-selected="true"] p { color: var(--ink) !important; font-weight: 600; }
[data-testid="stTab"] .react-aria-SelectionIndicator { background-color: var(--ink) !important; }
[data-testid="stTabListBorder"], [data-baseweb="tab-border"] { background-color: var(--border) !important; }

.stButton > button, [data-testid="stFormSubmitButton"] button {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; border-radius: 6px;
    background: var(--teal) !important; color: #06110F !important; border: none !important;
    transition: background 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
    background: #24BBA8 !important; transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(45,212,191,0.22);
}
.stButton > button:active, [data-testid="stFormSubmitButton"] button:active { transform: translateY(0); }

/* Reviewer actions other than the primary one stay quiet until you reach for them. */
.stButton > button[kind="secondary"], [data-testid="stFormSubmitButton"] button[kind="secondary"] {
    background: transparent !important; color: var(--text-0) !important;
    border: 1px solid var(--border) !important;
    transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease !important;
}
.stButton > button[kind="secondary"]:hover,
[data-testid="stFormSubmitButton"] button[kind="secondary"]:hover {
    background: var(--bg-2) !important; border-color: var(--text-1) !important; transform: translateY(-1px);
    box-shadow: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-1) !important; border: 1px dashed var(--border) !important; border-radius: 8px;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--teal) !important; background: var(--bg-2) !important;
}

/* --- 3D tilt on hover: a light, GPU-cheap perspective effect shared by every
   "card" surface (metric tiles, route cards, prediction cards, notification
   cards, table rows). transform-style + perspective on the parent give the
   tilted child real depth instead of a flat skew. --- */
[data-testid="stMetric"], .route-card, .pred-card, .notif-card, .ledger-table tbody tr {
    transition: transform 0.22s cubic-bezier(0.2, 0.7, 0.3, 1), box-shadow 0.22s ease, border-color 0.22s ease;
    transform-style: preserve-3d;
    will-change: transform;
}
[data-testid="stMetric"] {
    background: var(--bg-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px;
    perspective: 800px;
}
[data-testid="stMetric"]:hover {
    transform: perspective(800px) rotateX(3deg) rotateY(-3deg) translateY(-2px);
    border-color: rgba(45,212,191,0.35);
    box-shadow: 0 12px 28px rgba(0,0,0,0.35), 0 0 0 1px rgba(45,212,191,0.08);
}
[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important;
    letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-1) !important; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif !important; color: var(--text-0) !important; }

.ledger-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
.ledger-table th {
    text-align: left; font-family: 'Inter', sans-serif; font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--text-1); padding: 10px 12px; border-bottom: 1px solid var(--border);
}
.ledger-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text-0); }
.ledger-table tbody tr:hover {
    transform: perspective(1000px) rotateX(1deg) scale(1.003);
}
.ledger-table tr:hover td { background: var(--bg-1); }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 100px; font-size: 11px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 500;
}
.badge-risky { background: var(--amber-dim); color: var(--amber-ink); border: 1px solid rgba(232,163,61,0.35); }
.badge-clean { background: var(--teal-dim); color: var(--teal-ink); border: 1px solid rgba(45,212,191,0.35); }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.dot-risky { background: var(--amber); }
.dot-clean { background: var(--teal); }

.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--violet-ink); margin: 4px 0 10px 0;
}

.pred-card {
    background: var(--bg-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;
    perspective: 800px;
}
.pred-card:hover {
    transform: perspective(800px) rotateX(2.5deg) rotateY(-2.5deg) translateY(-2px);
    border-color: rgba(140,124,240,0.35);
    box-shadow: 0 12px 26px rgba(0,0,0,0.32);
}
.pred-model { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: var(--text-0); font-size: 14px; }
.pred-score { font-family: 'IBM Plex Mono', monospace; color: var(--text-1); font-size: 12px; margin-top: 2px; }

/* --- Approval routing ------------------------------------------------- */
.badge-auto { background: var(--teal-dim); color: var(--teal-ink); border: 1px solid rgba(45,212,191,0.35); }
.badge-manager { background: var(--violet-dim); color: var(--violet-ink); border: 1px solid rgba(140,124,240,0.35); }
.badge-controller { background: var(--amber-dim); color: var(--amber-ink); border: 1px solid rgba(232,163,61,0.35); }
.badge-dual { background: var(--rose-dim); color: var(--rose-ink); border: 1px solid rgba(242,100,123,0.40); }
.badge-neutral { background: rgba(92,92,87,0.12); color: var(--text-1); border: 1px solid var(--border); }

.route-card {
    background: var(--bg-1); border: 1px solid var(--border); border-left: 3px solid var(--text-1);
    border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; perspective: 800px;
}
.route-card:hover {
    transform: perspective(800px) rotateX(2.5deg) rotateY(-2.5deg) translateY(-2px);
    box-shadow: 0 12px 26px rgba(0,0,0,0.32);
}
.route-card.tier-routine  { border-left-color: var(--teal); }
.route-card.tier-elevated { border-left-color: var(--violet); }
.route-card.tier-high     { border-left-color: var(--amber); }
.route-card.tier-critical { border-left-color: var(--rose); }
.route-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; }
.route-name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 17px; color: var(--text-0); }
.route-meta {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-1);
    margin-top: 8px; letter-spacing: 0.03em;
}
.route-meta span { margin-right: 16px; }

.rule-chip {
    background: var(--bg-2); border: 1px solid var(--border); border-left: 2px solid var(--amber);
    border-radius: 6px; padding: 9px 12px; margin-top: 8px;
}
.rule-code {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--amber-ink);
    letter-spacing: 0.06em; margin-right: 8px;
}
.rule-name { font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 600; color: var(--text-0); }
.rule-detail { font-family: 'Inter', sans-serif; font-size: 12px; color: var(--text-1); margin-top: 3px; line-height: 1.5; }
.rule-empty {
    font-family: 'Inter', sans-serif; font-size: 13px; color: var(--text-1);
    padding: 9px 0 2px 0;
}

/* --- Confidence bands --------------------------------------------------- */
/* A band is a position on a scale, so it's drawn as one: the meter shows where
   the ensemble score landed, the ticks show the band boundaries it crossed. */
.band-meter {
    position: relative; height: 6px; border-radius: 100px; margin: 10px 0 6px 0;
    background: linear-gradient(90deg, var(--teal) 0%, var(--violet) 33%, var(--amber) 66%, var(--rose) 100%);
    opacity: 0.85;
}
.band-marker {
    position: absolute; top: 50%; width: 3px; height: 16px; border-radius: 2px;
    background: var(--text-0); transform: translate(-50%, -50%);
    box-shadow: 0 0 0 2px var(--bg-1);
}
.band-tick {
    position: absolute; top: 50%; width: 1px; height: 8px;
    background: var(--bg-0); transform: translate(-50%, -50%); opacity: 0.7;
}
.band-scale {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--text-1);
    letter-spacing: 0.04em;
}
.band-spread {
    position: absolute; top: 50%; height: 6px; border-radius: 100px;
    background: rgba(232, 236, 239, 0.28); transform: translateY(-50%);
}
.model-row {
    display: flex; justify-content: space-between; align-items: center;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--text-0);
    padding: 6px 0; border-bottom: 1px solid var(--border);
}
.model-row:last-child { border-bottom: none; }
.model-prob { color: var(--text-0); }
.match-line {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-1);
    margin-top: 8px; line-height: 1.6;
}

/* --- Notification preview ----------------------------------------------- */
/* Styled like a mail client, deliberately: the reviewer needs to judge the
   tone and urgency of the alert, and that reads differently in a table row
   than it does in something shaped like the message itself. */
.notif-card {
    background: var(--bg-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 0; margin-bottom: 12px; overflow: hidden; perspective: 800px;
}
.notif-card:hover {
    transform: perspective(800px) rotateX(2deg) rotateY(-2deg) translateY(-2px);
    box-shadow: 0 12px 26px rgba(0,0,0,0.32);
}
.notif-card.notif-inactive { border-style: dashed; opacity: 0.85; }
.notif-meta-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-bottom: 1px solid var(--border); background: var(--bg-2);
}
.notif-meta-row .notif-field {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-1);
}
.notif-meta-row .notif-field b { color: var(--text-0); font-weight: 500; }
.notif-subject {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 15px;
    color: var(--text-0); padding: 14px 16px 4px 16px;
}
.notif-summary {
    font-family: 'Inter', sans-serif; font-size: 13px; color: var(--text-1);
    padding: 0 16px 12px 16px; line-height: 1.55;
}
.notif-sla {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--amber-ink);
    padding: 0 16px 12px 16px;
}
.notif-reasons { padding: 0 16px 4px 16px; }
.notif-footer {
    font-family: 'Inter', sans-serif; font-style: italic; font-size: 11px; color: var(--text-1);
    padding: 10px 16px; border-top: 1px solid var(--border); background: var(--bg-2);
}

/* --- Dashboard ----------------------------------------------------------- */
.trend-row { margin-bottom: 14px; }
.trend-head {
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-1);
    margin-bottom: 5px;
}
.trend-head b { color: var(--text-0); font-weight: 500; }
.trend-bar {
    position: relative; height: 16px; border-radius: 4px; background: var(--bg-2);
    overflow: hidden; display: flex;
}
.trend-seg-cleared { background: var(--teal); height: 100%; }
.trend-seg-flagged { background: var(--violet); height: 100%; }
.trend-legend {
    display: flex; gap: 18px; margin: 4px 0 16px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-1);
}
.trend-legend span { display: inline-flex; align-items: center; gap: 6px; }
.trend-dot { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
.kpi-note {
    font-family: 'Inter', sans-serif; font-size: 12px; color: var(--text-1);
    line-height: 1.6; margin-top: 4px;
}

/* --- Audit trail ------------------------------------------------------- */
.chain-status {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; border-radius: 8px;
    padding: 11px 14px; margin-bottom: 14px; border: 1px solid var(--border);
}
.chain-ok { background: var(--teal-dim); color: var(--teal-ink); border-color: rgba(45,212,191,0.35); }
.chain-bad { background: var(--rose-dim); color: var(--rose-ink); border-color: rgba(242,100,123,0.40); }
.hash-cell { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-1); }
.event-type { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--violet-ink); }
.empty-state {
    background: var(--bg-1); border: 1px dashed var(--border); border-radius: 10px;
    padding: 26px 20px; text-align: center; color: var(--text-1);
    font-family: 'Inter', sans-serif; font-size: 14px;
}

footer, #MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Quantum kernel
# ---------------------------------------------------------------------------
dev = qml.device("default.qubit", wires=N_QUBITS)


def feature_map(x):
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2 * x[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]


def build_kernel_matrix(X1, X2):
    return np.array([[quantum_kernel(x1, x2) for x2 in X2] for x1 in X1])


@st.cache_resource
def train_models():
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    X_angles = X * np.pi

    X_train, X_test, y_train, y_test = train_test_split(
        X_angles, y, test_size=0.3, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    svm = SVC(kernel="rbf", C=10, probability=True).fit(X_train, y_train)

    K_train = build_kernel_matrix(X_train, X_train)
    qsvm = SVC(kernel="precomputed", C=10, probability=True).fit(K_train, y_train)

    return {"lr": lr, "svm": svm, "qsvm": qsvm, "X_train": X_train, "y_train": y_train}


@st.cache_data
def load_reference_data():
    return pd.read_csv(EXTRACTED_PATH)


def render_html(html):
    """
    Render a hand-indented HTML fragment through st.markdown without it being
    misread as plain text.

    Every multi-line HTML string in this file is written as a triple-quoted
    f-string nested inside the surrounding Python code, so it inherits that
    code's indentation — often 8 or 12 spaces before the first tag. Markdown
    treats a line with four or more leading spaces as the start of an
    indented code block, rendered as literal escaped text rather than parsed
    as HTML; a line that's whitespace-only (which the closing triple-quote of
    each nested chunk produces) counts as blank, which ends an HTML block that
    *did* get parsed, mid-way through, and dumps everything after it as text
    too. Both are exactly the bug this fixes: a results table showing up as
    raw <tr><td> tags on the page instead of an actual table.

    Collapsing the fragment to one whitespace-normalized line sidesteps both
    rules — there is no indentation left to misread and no line left to be
    blank — and changes nothing about how it renders, since browsers already
    collapse the whitespace between tags.
    """
    compact = " ".join(line.strip() for line in html.strip().splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def animate_kpi_counters():
    """
    Count up every st.metric value rendered so far from 0 to its final number
    on load, instead of the number simply appearing.

    st.markdown can't run <script> tags reliably — Streamlit inserts markdown
    via innerHTML, and scripts assigned that way never execute. Rendering the
    script inside an iframe does execute it, so the animation reaches into
    the parent document (`window.parent.document`, same-origin inside a
    normal Streamlit session) to find and animate the metric tiles that were
    just drawn above this call. Each tile is marked with a data attribute
    once animated so a later Streamlit rerun never restarts a count that's
    already finished.

    st.iframe is only available from Streamlit 1.4x onward; requirements.txt
    pins a wider range than that (>=1.30 covers releases that predate it),
    so a deployment can end up on a version without it. st.components.v1.html
    has done the same job since Streamlit's earliest releases and stays
    available across that whole range, so it's the fallback here rather than
    the primary path.
    """
    script = """
        <script>
        (function () {
            var doc = window.parent.document;
            var nodes = doc.querySelectorAll('[data-testid="stMetricValue"]:not([data-counted])');
            nodes.forEach(function (node) {
                var raw = node.textContent.trim();
                var match = raw.match(/^(-?[\\d.,]+)(.*)$/);
                if (!match) { node.setAttribute('data-counted', '1'); return; }
                var target = parseFloat(match[1].replace(/,/g, ''));
                var suffix = match[2] || '';
                var decimals = (match[1].split('.')[1] || '').length;
                if (isNaN(target)) { node.setAttribute('data-counted', '1'); return; }
                node.setAttribute('data-counted', '1');
                var duration = 700;
                var start = null;
                function step(ts) {
                    if (start === null) { start = ts; }
                    var progress = Math.min((ts - start) / duration, 1);
                    var eased = 1 - Math.pow(1 - progress, 3);
                    var current = target * eased;
                    node.textContent = current.toFixed(decimals) + suffix;
                    if (progress < 1) {
                        window.requestAnimationFrame(step);
                    } else {
                        node.textContent = raw;
                    }
                }
                window.requestAnimationFrame(step);
            });
        })();
        </script>
        """
    if hasattr(st, "iframe"):
        st.iframe(script, height=1)
    else:
        st.components.v1.html(script, height=0)


def cell(value, dash="N/A"):
    """Render a possibly-missing DB value. NaN is truthy, so `value or dash` won't do."""
    return dash if value is None or pd.isna(value) else value


def route_badge(route):
    meta = ROUTE_META.get(route, {})
    return f'<span class="badge {meta.get("badge", "badge-neutral")}">{meta.get("label", route)}</span>'


def status_badge(status):
    meta = STATUS_META.get(status, {})
    return f'<span class="badge {meta.get("badge", "badge-neutral")}">{meta.get("label", status)}</span>'


def rules_html(rules):
    """Render the fired approval rules as evidence chips."""
    if not rules:
        return '<div class="rule-empty">No approval rules triggered.</div>'
    out = ""
    for r in rules:
        out += (
            f'<div class="rule-chip">'
            f'<span class="rule-code">{r["code"]}</span>'
            f'<span class="rule-name">{r["name"]}</span>'
            f'<div class="rule-detail">{r["detail"]}</div>'
            f"</div>"
        )
    return out


def route_card(route, tier, approver, sla_hours, policy_version, rules):
    """The full 'who signs this off, and why' panel."""
    label = ROUTE_META.get(route, {}).get("label", route)
    sla = "immediate" if not sla_hours else f"{sla_hours}h"
    return f"""
    <div class="route-card tier-{tier}">
        <div class="route-head">
            <div>
                <div class="section-label" style="margin:0 0 4px 0;">APPROVAL ROUTE</div>
                <div class="route-name">{label}</div>
            </div>
            {route_badge(route)}
        </div>
        <div class="route-meta">
            <span>APPROVER &middot; {approver}</span>
            <span>SLA &middot; {sla}</span>
            <span>POLICY &middot; v{policy_version}</span>
        </div>
        {rules_html(rules)}
    </div>
    """


def band_badge(band_key):
    if not band_key:
        return '<span class="badge badge-neutral">Not scored</span>'
    return f'<span class="badge {conf.band_badge(band_key)}">{conf.band_label(band_key)}</span>'


def confidence_meter(assessment):
    """
    A band is a position on a scale, so draw it as one: the marker is the
    ensemble mean, the lighter bar behind it is the spread across the three
    models, and the ticks are the band boundaries.
    """
    values = list(assessment.probabilities.values())
    lo, hi = min(values) * 100, max(values) * 100
    ticks = "".join(f'<div class="band-tick" style="left:{t}%;"></div>' for t in (25, 50, 75))
    return f"""
    <div class="band-meter">
        <div class="band-spread" style="left:{lo:.1f}%; width:{max(hi - lo, 0.6):.1f}%;"></div>
        {ticks}
        <div class="band-marker" style="left:{assessment.mean * 100:.1f}%;"></div>
    </div>
    <div class="band-scale">
        <span>0%</span><span>CLEARED</span><span>LOW</span><span>ELEVATED</span><span>HIGH</span>
    </div>
    """


def confidence_card(assessment, duplicate_match=None):
    """Ensemble band, the spread behind it, and each model's own score."""
    if assessment is None:
        return (
            '<div class="route-card"><div class="route-name">Not scored</div>'
            '<div class="rule-detail">Extraction failed, so no model score is available.</div></div>'
        )

    rows = "".join(
        f'<div class="model-row"><span>{name}</span>'
        f'<span class="model-prob">{p:.0%} &nbsp;{band_badge(assessment.per_model_bands[name])}</span>'
        f"</div>"
        for name, p in sorted(assessment.probabilities.items())
    )

    if assessment.is_split:
        agreement = (
            f'<div class="rule-detail" style="color:var(--amber-ink);">'
            f"Split decision, {assessment.spread:.0%} spread between the most and least "
            f"concerned model.</div>"
        )
    else:
        agreement = (
            f'<div class="rule-detail">Models {assessment.agreement}, '
            f"{assessment.spread:.0%} spread.</div>"
        )

    match_html = ""
    if duplicate_match is not None and duplicate_match.matched_invoice_id:
        match_html = f'<div class="match-line">CLOSEST MATCH &middot; {duplicate_match.describe()}</div>'

    return f"""
    <div class="route-card tier-{assessment.band if assessment.band != 'cleared' else 'routine'}">
        <div class="route-head">
            <div>
                <div class="section-label" style="margin:0 0 4px 0;">ENSEMBLE CONFIDENCE</div>
                <div class="route-name">{assessment.label} &middot; {assessment.mean:.0%}</div>
            </div>
            {band_badge(assessment.band)}
        </div>
        {confidence_meter(assessment)}
        {agreement}
        <div style="margin-top:10px;">{rows}</div>
        {match_html}
    </div>
    """


def score_and_route(record, reference_df, extraction_ok=True):
    """
    Run one extracted invoice through the models and the approval policy.

    Returns (features, duplicate_match, assessment, decision). When extraction
    failed there's nothing to score, so the models are skipped and routing runs
    on that fact alone.
    """
    if not extraction_ok:
        return {}, None, None, approval_rules.evaluate(
            record.get("amount"), {}, None, extraction_ok=False
        )

    features = compute_features_for_new_invoice(record, reference_df)
    duplicate_match = assess_duplicate(record, reference_df)
    x_new = np.array([features[c] for c in FEATURE_COLS]) * np.pi

    k_vec = build_kernel_matrix([x_new], models["X_train"])
    assessment = conf.assess({
        "LogReg": float(models["lr"].predict_proba([x_new])[0][1]),
        "RBF-SVM": float(models["svm"].predict_proba([x_new])[0][1]),
        "Quantum": float(models["qsvm"].predict_proba(k_vec)[0][1]),
    })

    # The matched invoice travels with the features so R-03 can cite it.
    features["duplicate_match_id"] = duplicate_match.matched_invoice_id
    decision = approval_rules.evaluate(
        record.get("amount"), features, assessment, extraction_ok=True
    )
    return features, duplicate_match, assessment, decision


def notification_reasons_html(reasons):
    if not reasons:
        return ""
    chips = "".join(
        f'<div class="rule-chip"><span class="rule-code">{r["code"]}</span>'
        f'<span class="rule-name">{r["name"]}</span>'
        f'<div class="rule-detail">{r["detail"]}</div></div>'
        for r in reasons
    )
    return f'<div class="notif-reasons">{chips}</div>'


def notification_card(n):
    """
    Render the alert a routing decision would generate, styled as an email
    preview so its tone and urgency can be judged the way an approver would
    actually see them — not just read off as a table row.

    `n` is a plain dict shaped like Notification.to_dict() — either a freshly
    built preview or one read back from a stored "notification.generated"
    audit event, so both render through the same code path.
    """
    inactive_class = "" if n["is_actionable"] else " notif-inactive"
    to_line = ", ".join(n["recipients"]) if n["recipients"] else "no approver, not sent"

    return f"""
    <div class="notif-card{inactive_class}">
        <div class="notif-meta-row">
            <span class="notif-field">TO &middot; <b>{to_line}</b></span>
            <span class="notif-field">PRIORITY &middot; <b>{n["priority"]}</b></span>
        </div>
        <div class="notif-subject">{n["subject"]}</div>
        <div class="notif-summary">{n["summary_line"]}</div>
        {notification_reasons_html(n["reasons"])}
        <div class="notif-sla">{n["sla_line"]}</div>
        <div class="notif-footer">Simulated preview. No message has actually been sent.</div>
    </div>
    """


def decision_view_from_row(row):
    """
    A duck-typed stand-in for a RoutingDecision, built from a persisted queue
    row rather than recomputed from the model. notif.build_notification only
    needs a handful of attributes, and the row already carries the *current*
    route — which may have moved since intake if the invoice was escalated —
    so this reflects what's true now rather than replaying the original
    routing event from the audit trail.
    """
    route = row["route"]
    meta = ROUTE_META[route]
    return types.SimpleNamespace(
        is_auto=route == "auto_approve",
        route=route,
        tier=meta["tier"],
        label=meta["label"],
        sla_hours=meta["sla_hours"],
        policy_version=row["policy_version"],
        rules_as_dicts=lambda: json.loads(row["rules_json"] or "[]"),
    )


def trend_bar_row(label, cleared, flagged, right_text=""):
    """One run's cleared/flagged split as a proportional two-segment bar."""
    total = max(cleared + flagged, 1)
    cleared_pct = cleared / total * 100
    flagged_pct = flagged / total * 100
    return f"""
    <div class="trend-row">
        <div class="trend-head"><b>{label}</b><span>{right_text}</span></div>
        <div class="trend-bar">
            <div class="trend-seg-cleared" style="width:{cleared_pct:.2f}%;"
                 title="{cleared} cleared"></div>
            <div class="trend-seg-flagged" style="width:{flagged_pct:.2f}%;"
                 title="{flagged} flagged"></div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Landing hero — Section 1 of the scroll-page structure. Nav row (logomark,
# plain nav labels, a pill CTA), a huge mono headline tight under it, one
# line of supporting copy with its own CTA, and a full-height split band:
# an abstract circuit panel standing in for photography, paired with a
# solid panel-accent block. Both CTAs scroll to the #workspace anchor
# placed just above the tool tabs.
# ---------------------------------------------------------------------------
render_html(
    """
    <div class="landing-nav">
        <div class="landing-logomark">
            <span class="logomark-shape"></span>
            <span class="logomark-word">Invoice Risk Ledger</span>
        </div>
        <div class="landing-nav-right">
            <div class="landing-nav-links">
                <span>Pipeline</span><span>Policy</span><span>Audit Trail</span>
            </div>
            <a href="#workspace" class="landing-nav-cta">
                <span class="landing-nav-cta-text">Get started</span>
                <span class="landing-nav-cta-icon">&#8599;</span>
            </a>
        </div>
    </div>
    <div class="landing-headline">Automated<br>Invoice Risk<br>Classification</div>
    <div class="landing-hero-body">
        <div class="landing-desc">
            Every invoice is extracted, scored across three risk models, classical
            and quantum kernel, and routed to the right approver, with every step
            written to a hash-chained audit trail.
        </div>
        <a href="#workspace" class="landing-cta">
            <span class="landing-cta-text">Get started</span>
            <span class="landing-cta-icon">&#8599;</span>
        </a>
    </div>
    <div class="landing-split">
        <div class="landing-split-panel landing-split-panel--circuit">
            <div class="landing-panel-grid"></div>
            <div class="landing-panel-glow"></div>
            <div class="landing-panel-diamond"></div>
            <div class="landing-panel-node landing-panel-node--1"></div>
            <div class="landing-panel-node landing-panel-node--2"></div>
            <div class="landing-panel-node landing-panel-node--3"></div>
        </div>
        <div class="landing-split-panel landing-split-panel--yellow">
            <div class="split-mark">&#9670;</div>
            <div class="split-divider"></div>
            <div class="split-caption">5-Qubit Quantum Kernel</div>
        </div>
    </div>
    """
)

# ---------------------------------------------------------------------------
# Capabilities — Section 2. Bracketed label, a large mono headline, and a
# three-column grid of numbered blocks covering the three real phases of
# the pipeline: intake, scoring, routing.
# ---------------------------------------------------------------------------
render_html(
    """
    <div class="cap-section">
        <div class="cap-label">[ 01 / CAPABILITIES ]</div>
        <div class="cap-headline">What It Does</div>
        <div class="cap-desc">
            Three phases run on every invoice, in order, without a human touching
            the ones that don't need one.
        </div>
        <div class="cap-grid">
            <div class="cap-col">
                <div class="cap-col-divider"></div>
                <div class="cap-num">01</div>
                <div class="cap-name">Automated Intake</div>
                <div class="cap-col-desc">
                    PDF invoices are parsed with pdfplumber and normalized into five
                    audit-relevant risk features the moment they land.
                </div>
            </div>
            <div class="cap-col">
                <div class="cap-col-divider"></div>
                <div class="cap-num">02</div>
                <div class="cap-name">Risk Scoring</div>
                <div class="cap-col-desc">
                    Logistic regression, an RBF-kernel SVM, and a quantum kernel SVM
                    each score the invoice, combined into one banded confidence.
                </div>
            </div>
            <div class="cap-col">
                <div class="cap-col-divider"></div>
                <div class="cap-num">03</div>
                <div class="cap-name">Approval Routing</div>
                <div class="cap-col-desc">
                    Nine deterministic policy rules route the invoice to auto-approval
                    or the right human reviewer, logged to a hash-chained audit trail.
                </div>
            </div>
        </div>
    </div>
    """
)

st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)

audit_store.init_db()
models = train_models()
reference_df = load_reference_data()

# ---------------------------------------------------------------------------
# Reviewer identity — every queue action is attributed, so it has to be set
# before anyone can act on an invoice.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="section-label">ACTING AS</div>', unsafe_allow_html=True)
    reviewer = st.text_input("Reviewer", value="", placeholder="name@company.com",
                             label_visibility="collapsed")
    st.caption(
        "Approvals, rejections and escalations are recorded against this name in the "
        "audit trail. Dual-control invoices need two different reviewers."
    )

    st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">POLICY</div>', unsafe_allow_html=True)
    render_html(
        f"""
        <div class="route-meta" style="margin:0;">
            <div>VERSION &middot; v{approval_rules.POLICY_VERSION}</div>
            <div style="margin-top:6px;">AUTO-APPROVE &middot; &lt; ${approval_rules.MANAGER_LIMIT:,.0f}</div>
            <div style="margin-top:6px;">MANAGER &middot; ${approval_rules.MANAGER_LIMIT:,.0f} &ndash; ${approval_rules.CONTROLLER_LIMIT:,.0f}</div>
            <div style="margin-top:6px;">CONTROLLER &middot; ${approval_rules.CONTROLLER_LIMIT:,.0f} &ndash; ${approval_rules.DUAL_CONTROL_LIMIT:,.0f}</div>
            <div style="margin-top:6px;">DUAL CONTROL &middot; &ge; ${approval_rules.DUAL_CONTROL_LIMIT:,.0f}</div>
        </div>
        """
    )

    st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">CONFIDENCE BANDS</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(
            f'<div class="model-row"><span>{b["label"]}</span>'
            f'<span class="model-prob">{b["lower"]:.0%}&ndash;{min(b["upper"], 1.0):.0%}</span></div>'
            for b in conf.BANDS
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Bands apply to the mean of the three model scores. A spread of "
        f"{conf.SPLIT_SPREAD:.0%} or more is reported as a split decision. "
        f"Invoice similarity at or above {approval_rules.NEAR_DUPLICATE:.0%} is "
        f"screened as a possible duplicate."
    )

stats = audit_store.queue_stats()

st.markdown('<div id="workspace"></div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["BATCH INTAKE", "SINGLE INVOICE", f"REVIEW QUEUE ({stats['open']})",
     "DASHBOARD", "AUDIT TRAIL", "MODEL COMPARISON"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Batch Intake
# ---------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-label">AUTOMATED PROCESSING</div>', unsafe_allow_html=True)
    st.caption(
        "Drop in a batch of invoice PDFs. Every invoice is extracted, feature-engineered, "
        "classified, and routed for approval automatically, cleared for payment or pushed "
        "to the review queue, with the whole run written to the audit trail."
    )

    batch_files = st.file_uploader(
        "Upload invoice PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )
    use_samples = st.checkbox("Run on all 40 sample invoices instead")
    run_batch = st.button("Run automated batch", type="primary")

    if run_batch:
        sources = []
        if use_samples:
            sample_files = sorted(os.listdir(INVOICES_DIR))
            sources = [(f, os.path.join(INVOICES_DIR, f)) for f in sample_files]
        elif batch_files:
            sources = [(f.name, f) for f in batch_files]

        if not sources:
            st.warning("Upload at least one PDF, or check the sample-invoices box.")
        else:
            progress = st.progress(0, text="Starting batch run...")
            batch_results = []

            for i, (fname, src) in enumerate(sources):
                progress.progress((i + 1) / len(sources), text=f"Processing {fname}...")
                try:
                    record = extract_fields_from_pdf(src)
                    extraction_ok = not any(v is None for v in record.values())
                except Exception:
                    record, extraction_ok = {"invoice_id": None}, False

                features, duplicate_match, assessment, decision = score_and_route(
                    record, reference_df, extraction_ok
                )
                invoice_id, status, _ = audit_store.submit_invoice(
                    record, features, assessment, decision, source_file=fname, actor="batch-intake"
                )

                batch_results.append({
                    "file": fname,
                    "invoice_id": invoice_id,
                    "vendor": record.get("vendor") or "-",
                    "amount": record.get("amount"),
                    "confidence": assessment.mean if assessment else None,
                    "band": assessment.band if assessment else None,
                    "spread": assessment.spread if assessment else None,
                    "split": assessment.is_split if assessment else None,
                    "duplicate_score": features.get("duplicate_score"),
                    "duplicate_match": duplicate_match.matched_invoice_id if duplicate_match else None,
                    "route": decision.route,
                    "route_label": decision.label,
                    "rules_fired": ", ".join(r.code for r in decision.rules) or "-",
                    "status": status,
                    "failed": not extraction_ok,
                })

            progress.empty()

            summary = dashboard.summarize_results(batch_results)
            source = "sample_set" if use_samples else "batch_intake"
            audit_store.record_batch_run(source, "batch-intake", summary)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Processed", summary["total"])
            c2.metric("Auto-approved", summary["auto_approved"])
            c3.metric("Routed for review", summary["routed"])
            c4.metric("Touch rate", f"{summary['routed'] / summary['total']:.0%}")

            c5, c6, c7 = st.columns(3)
            c5.metric("High confidence band", summary["high_band"])
            c6.metric("Split decisions", summary["split"])
            c7.metric("Possible duplicates", summary["duplicates"])
            st.caption(
                f"{summary['total']} invoice(s) written to the review queue and audit trail, "
                f"{summary['auto_approved']} cleared automatically, {summary['routed']} routed "
                f"for review. This run is now part of the trend on the Dashboard tab."
            )
            if summary["failed"]:
                st.warning(
                    f"{summary['failed']} invoice(s) had extraction issues, routed for manual "
                    f"review rather than cleared, since an unreadable document can't be "
                    f"auto-approved."
                )

            # Highest approval authority first, then most-confident within a route:
            # what needs a person surfaces above what doesn't.
            sorted_results = sorted(
                batch_results,
                key=lambda r: (-ROUTE_ORDER.index(r["route"]), -(r["confidence"] or 0)),
            )

            rows_html = ""
            for r in sorted_results:
                if r["failed"]:
                    rows_html += (
                        f"<tr><td>{r['file']}</td><td>{r['invoice_id']}</td>"
                        f"<td colspan='4'><span class='badge badge-neutral'>extraction failed</span></td>"
                        f"<td>{route_badge(r['route'])}</td><td class='hash-cell'>"
                        f"{r['rules_fired']}</td></tr>"
                    )
                    continue

                dup = r["duplicate_score"] or 0
                dup_cell = f"{dup:.0%}"
                if dup >= approval_rules.NEAR_DUPLICATE and r["duplicate_match"]:
                    dup_cell = f"<span style='color:var(--amber-ink);'>{dup:.0%} {r['duplicate_match']}</span>"

                split_mark = (
                    ' <span style="color:var(--amber-ink);" title="models disagree">&#9663;</span>'
                    if r["split"] else ""
                )

                rows_html += f"""
                <tr>
                    <td>{r['file']}</td>
                    <td>{r['invoice_id']}</td>
                    <td style="font-family:'Inter',sans-serif;">{r['vendor']}</td>
                    <td>${r['amount']:,.2f}</td>
                    <td>{r['confidence']:.0%}{split_mark}</td>
                    <td>{band_badge(r['band'])}</td>
                    <td>{dup_cell}</td>
                    <td>{route_badge(r['route'])}</td>
                    <td class="hash-cell">{r['rules_fired']}</td>
                </tr>
                """

            table_html = f"""
            <table class="ledger-table">
                <thead><tr>
                    <th>File</th><th>Invoice</th><th>Vendor</th><th>Amount</th>
                    <th>Risk score</th><th>Band</th><th>Closest match</th>
                    <th>Route</th><th>Rules</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """
            render_html(table_html)
            st.caption(
                "Risk score is the mean of the three models; ▽ marks a split decision where "
                "they disagree by more than 35 points. Closest match is the highest invoice "
                "similarity on file, amber above the 80% review threshold."
            )

            results_df = pd.DataFrame(sorted_results)
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download report (.csv)", csv, "invoice_risk_report.csv", "text/csv")

# ---------------------------------------------------------------------------
# TAB 2 — Single Invoice
# ---------------------------------------------------------------------------
with tab2:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">01 &middot; SOURCE</div>', unsafe_allow_html=True)
        source = st.radio("Source", ["Upload a PDF", "Try a sample invoice"], horizontal=True, label_visibility="collapsed")

        pdf_source = None
        if source == "Upload a PDF":
            uploaded = st.file_uploader("Upload invoice PDF", type=["pdf"], label_visibility="collapsed")
            if uploaded:
                pdf_source = uploaded
        else:
            sample_files = sorted(os.listdir(INVOICES_DIR))
            chosen = st.selectbox("Pick a sample invoice", sample_files, label_visibility="collapsed")
            if chosen:
                pdf_source = os.path.join(INVOICES_DIR, chosen)

        record = None
        if pdf_source:
            record = extract_fields_from_pdf(pdf_source)
            st.markdown('<div class="section-label" style="margin-top:22px;">02 &middot; EXTRACTED FIELDS</div>', unsafe_allow_html=True)
            st.json(record)

    with col2:
        if record and all(v is not None for v in record.values()):
            features, duplicate_match, assessment, decision = score_and_route(record, reference_df)

            st.markdown('<div class="section-label">03 &middot; RISK FEATURES</div>', unsafe_allow_html=True)
            feat_df = pd.DataFrame([{c: features[c] for c in FEATURE_COLS}])
            st.dataframe(feat_df, width='stretch', hide_index=True)

            if duplicate_match.matched_invoice_id:
                st.markdown(
                    f'<div class="match-line">DUPLICATE SCREEN &middot; {duplicate_match.describe()}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-label" style="margin-top:22px;">04 &middot; VERDICT</div>',
                        unsafe_allow_html=True)
            render_html(confidence_card(assessment, duplicate_match))

            # --- 05 · Approval routing -------------------------------------
            st.markdown('<div class="section-label" style="margin-top:22px;">05 &middot; APPROVAL ROUTING</div>',
                        unsafe_allow_html=True)
            render_html(
                route_card(decision.route, decision.tier, decision.approver,
                           decision.sla_hours, decision.policy_version, decision.rules_as_dicts())
            )

            # --- 06 · Notification preview ----------------------------------
            st.markdown('<div class="section-label" style="margin-top:22px;">06 &middot; NOTIFICATION PREVIEW</div>',
                        unsafe_allow_html=True)
            st.caption(
                "What the approver would see, before anything is actually submitted. "
                "Nothing here is sent, this is a preview of the alert this routing "
                "decision would generate."
            )
            preview_notification = notif.build_notification(
                record.get("invoice_id"), record.get("vendor"), record.get("amount"), decision,
            )
            render_html(notification_card(preview_notification.to_dict()))

            if st.button("Submit to review queue", key="submit_single"):
                invoice_id, status, was_new = audit_store.submit_invoice(
                    record, features, assessment, decision,
                    source_file=getattr(pdf_source, "name", os.path.basename(str(pdf_source))),
                    actor="single-intake",
                )
                label = STATUS_META.get(status, {}).get("label", status)
                if not was_new and status in audit_store.TERMINAL_STATUSES:
                    st.info(f"{invoice_id} was already decided ({label.lower()}). "
                            f"The re-submission was logged but the decision stands.")
                elif decision.is_auto:
                    st.success(f"{invoice_id} cleared for payment automatically. No rules triggered.")
                else:
                    st.success(f"{invoice_id} queued for {decision.label.lower()}. "
                               f"See the Review Queue tab.")
        elif pdf_source:
            st.error("Could not extract all required fields from this PDF. Try another invoice.")

# ---------------------------------------------------------------------------
# TAB 3 — Review Queue
# ---------------------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-label">PENDING APPROVALS</div>', unsafe_allow_html=True)
    st.caption(
        "Everything the policy engine wouldn't clear on its own, ordered by the authority "
        "it needs. Approve, reject, ask for more information, or escalate: each action is "
        "attributed and written to the audit trail."
    )

    queue_df = audit_store.load_queue()

    if queue_df.empty:
        st.markdown(
            '<div class="empty-state">Nothing in the queue yet. Run a batch from the '
            '<strong>Batch Intake</strong> tab to populate it.</div>',
            unsafe_allow_html=True,
        )
    else:
        by_status = stats["by_status"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open", stats["open"])
        c2.metric("Awaiting 2nd approval", by_status.get(audit_store.STATUS_AWAITING_SECOND, 0))
        c3.metric("Approved", by_status.get(audit_store.STATUS_APPROVED, 0)
                  + by_status.get(audit_store.STATUS_AUTO_APPROVED, 0))
        c4.metric("Rejected", by_status.get(audit_store.STATUS_REJECTED, 0))

        f1, f2 = st.columns(2)
        present_statuses = [s for s in STATUS_META if s in set(queue_df["status"])]
        default_statuses = [s for s in present_statuses if s in OPEN_STATUSES] or present_statuses
        status_filter = f1.multiselect(
            "Status", present_statuses, default=default_statuses,
            format_func=lambda s: STATUS_META[s]["label"],
        )
        present_routes = [r for r in ROUTE_ORDER if r in set(queue_df["route"])]
        route_filter = f2.multiselect(
            "Route", present_routes, default=present_routes,
            format_func=lambda r: ROUTE_META[r]["label"],
        )

        view = queue_df[queue_df["status"].isin(status_filter) & queue_df["route"].isin(route_filter)]

        if view.empty:
            st.markdown('<div class="empty-state">No invoices match these filters.</div>',
                        unsafe_allow_html=True)
        else:
            rows_html = ""
            for _, r in view.iterrows():
                codes = ", ".join(x["code"] for x in json.loads(r["rules_json"] or "[]")) or "-"
                amount = f"${r['amount']:,.2f}" if pd.notna(r["amount"]) else "N/A"
                score = f"{float(r['confidence_mean']):.0%}" if pd.notna(r["confidence_mean"]) else "N/A"
                band = band_badge(r["confidence_band"]) if pd.notna(r["confidence_band"]) else ""
                rows_html += f"""
                <tr>
                    <td>{r['invoice_id']}</td>
                    <td style="font-family:'Inter',sans-serif;">{cell(r['vendor'], '-')}</td>
                    <td>{amount}</td>
                    <td>{score}</td>
                    <td>{band}</td>
                    <td>{route_badge(r['route'])}</td>
                    <td>{status_badge(r['status'])}</td>
                    <td class="hash-cell">{codes}</td>
                    <td class="hash-cell">{str(cell(r['submitted_at'], ''))[:16].replace('T', ' ')}</td>
                </tr>
                """
            render_html(
                f"""
                <table class="ledger-table">
                    <thead><tr>
                        <th>Invoice</th><th>Vendor</th><th>Amount</th><th>Risk</th><th>Band</th>
                        <th>Route</th><th>Status</th><th>Rules</th><th>Submitted</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """
            )

        # --- Reviewer actions ------------------------------------------------
        st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">REVIEWER ACTION</div>', unsafe_allow_html=True)

        actionable = queue_df[queue_df["status"].isin(OPEN_STATUSES)]
        if actionable.empty:
            st.markdown(
                '<div class="empty-state">Queue is clear. Every invoice has been decided.</div>',
                unsafe_allow_html=True,
            )
        else:
            selected_id = st.selectbox(
                "Invoice to action",
                actionable["invoice_id"].tolist(),
                format_func=lambda i: (
                    f"{i} · {ROUTE_META[actionable.loc[actionable['invoice_id'] == i, 'route'].iloc[0]]['label']}"
                ),
            )
            row = actionable[actionable["invoice_id"] == selected_id].iloc[0]

            d1, d2 = st.columns([1.15, 1], gap="large")
            with d1:
                render_html(
                    route_card(
                        row["route"], ROUTE_META[row["route"]]["tier"],
                        ROUTE_META[row["route"]]["approver"], ROUTE_META[row["route"]]["sla_hours"],
                        row["policy_version"], json.loads(row["rules_json"] or "[]"),
                    )
                )
            with d2:
                amount = f"${row['amount']:,.2f}" if pd.notna(row["amount"]) else "not extracted"
                dup_score = row["duplicate_score"]
                dup_line = "N/A"
                if pd.notna(dup_score):
                    match_id = cell(row["duplicate_match_id"], "no match")
                    dup_line = f"{float(dup_score):.0%} vs {match_id}"
                render_html(
                    f"""
                    <div class="route-card">
                        <div class="route-name">{cell(row['vendor'], 'Unknown vendor')}</div>
                        <div class="route-meta">
                            <div>AMOUNT &middot; {amount}</div>
                            <div style="margin-top:6px;">POSTED &middot; {cell(row['posting_date'])}
                                &nbsp;&nbsp; DUE &middot; {cell(row['due_date'])}</div>
                            <div style="margin-top:6px;">SOURCE &middot; {cell(row['source_file'])}</div>
                            <div style="margin-top:6px;">STATUS &middot; {STATUS_META[row['status']]['label']}</div>
                            <div style="margin-top:6px;">CLOSEST MATCH &middot; {dup_line}</div>
                        </div>
                    </div>
                    """
                )

                stored = json.loads(row["confidence_json"]) if pd.notna(row["confidence_json"]) else None
                if stored:
                    render_html(confidence_card(conf.assess(stored["probabilities"])))

            st.markdown('<div class="section-label" style="margin-top:6px;">NOTIFICATION PREVIEW</div>',
                        unsafe_allow_html=True)
            queue_notification = notif.build_notification(
                row["invoice_id"], row["vendor"], row["amount"],
                decision_view_from_row(row), submitted_at=row["submitted_at"],
            )
            render_html(notification_card(queue_notification.to_dict()))

            if row["status"] == audit_store.STATUS_AWAITING_SECOND:
                st.info(
                    f"Dual control: {row['first_approver']} gave the first approval. "
                    f"A second, different reviewer must approve to release this invoice."
                )

            with st.form("reviewer_action", clear_on_submit=True):
                note = st.text_area(
                    "Reviewer note",
                    placeholder="What did you check, and what did you conclude?",
                    height=80,
                )
                b1, b2, b3, b4 = st.columns(4)
                clicked = {
                    "approve": b1.form_submit_button("Approve", type="primary"),
                    "request_info": b2.form_submit_button("Request info", type="secondary"),
                    "escalate": b3.form_submit_button("Escalate", type="secondary"),
                    "reject": b4.form_submit_button("Reject", type="secondary"),
                }

            action = next((a for a, was_clicked in clicked.items() if was_clicked), None)
            if action:
                try:
                    _, message = audit_store.apply_action(selected_id, action, reviewer, note)
                    st.session_state["queue_flash"] = ("success", message)
                    st.rerun()
                except WorkflowError as exc:
                    st.error(str(exc))

        flash = st.session_state.pop("queue_flash", None)
        if flash:
            st.success(flash[1])


# ---------------------------------------------------------------------------
# TAB 4 — Dashboard
# ---------------------------------------------------------------------------
with tab4:
    st.markdown('<div class="section-label">TOUCHLESS PROCESSING RATE</div>', unsafe_allow_html=True)
    st.caption(
        "How much of the intake volume never needs a human, the automation "
        "payoff this whole pipeline exists to deliver."
    )

    full_queue = audit_store.load_queue()
    overall_touchless = dashboard.touchless_rate(full_queue)
    runs_df = dashboard.annotate_run_history(audit_store.load_batch_runs())

    d1, d2, d3 = st.columns(3)
    d1.metric("Touchless rate (all-time)",
              f"{overall_touchless:.0%}" if overall_touchless is not None else "N/A")
    d2.metric("Invoices processed", 0 if full_queue.empty else len(full_queue))
    d3.metric("Batch runs recorded", 0 if runs_df is None or runs_df.empty else len(runs_df))

    st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">FLAGGED VS. CLEARED, BATCH RUN TREND</div>',
                unsafe_allow_html=True)
    st.caption(
        "Each run from the Batch Intake tab, in order. The bar is the split within that "
        "run; the cumulative rate is the trend line that actually matters, whether "
        "automation is holding up as more volume runs through it."
    )

    if runs_df is None or runs_df.empty:
        st.markdown(
            '<div class="empty-state">No batch runs recorded yet. Run a batch from the '
            '<strong>Batch Intake</strong> tab to start the trend.</div>',
            unsafe_allow_html=True,
        )
    else:
        render_html(
            """
            <div class="trend-legend">
                <span><span class="trend-dot" style="background:var(--teal);"></span>Cleared (auto-approved)</span>
                <span><span class="trend-dot" style="background:var(--violet);"></span>Flagged (routed for review)</span>
            </div>
            """
        )
        for _, r in runs_df.iterrows():
            right = (
                f"{int(r['auto_approved_count'])}/{int(r['invoice_count'])} cleared &middot; "
                f"{r['touchless_rate']:.0%} this run &middot; {r['cumulative_touchless_rate']:.0%} cumulative"
            )
            render_html(trend_bar_row(r["run_label"], r["auto_approved_count"], r["flagged_count"], right))

        runs_table = runs_df[[
            "run_label", "started_at", "source", "invoice_count",
            "auto_approved_count", "flagged_count", "touchless_rate",
        ]].copy()
        runs_table["started_at"] = runs_table["started_at"].astype(str).str[:19].str.replace("T", " ")
        runs_table["touchless_rate"] = (runs_table["touchless_rate"] * 100).round(1).astype(str) + "%"
        runs_table.columns = ["Run", "Started (UTC)", "Source", "Total", "Cleared", "Flagged", "Touchless rate"]
        st.dataframe(runs_table, width='stretch', hide_index=True)

    st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">FALSE-POSITIVE RATE</div>', unsafe_allow_html=True)
    st.caption(
        "Of the invoices the policy sent for review, how many did a reviewer approve rather "
        "than reject, a flag that turned out not to be a problem. This uses the reviewer's "
        "own decision, not the dataset's known labels: a real invoice never carries a label, "
        "so a real deployment has no other signal to track this against."
    )

    fp = dashboard.false_positive_summary(full_queue)
    if fp["decided"] == 0:
        st.markdown(
            '<div class="empty-state">No flagged invoice has been decided yet. Approve or '
            'reject items in the <strong>Review Queue</strong> tab to start tracking this.</div>',
            unsafe_allow_html=True,
        )
    else:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("False-positive rate", f"{fp['false_positive_rate']:.0%}")
        e2.metric("Decided", fp["decided"])
        e3.metric("False positives", fp["false_positives"])
        e4.metric("Coverage", f"{fp['coverage']:.0%}" if fp["coverage"] is not None else "N/A")

        if fp["decided"] < 5:
            st.markdown(
                f'<div class="kpi-note">Based on only {fp["decided"]} decision(s), too few '
                f"to read as a stable rate yet. Treat this as a placeholder until more of the "
                f"queue has been worked.</div>",
                unsafe_allow_html=True,
            )

        trend = dashboard.false_positive_trend(full_queue)
        if len(trend) > 1:
            st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
            trend_table = trend.copy()
            trend_table["false_positive_rate"] = (
                (trend_table["false_positive_rate"] * 100).round(1).astype(str) + "%"
            )
            trend_table.columns = ["Date", "Decided", "False positives", "False-positive rate"]
            st.dataframe(trend_table, width='stretch', hide_index=True)
        else:
            st.markdown(
                '<div class="kpi-note">All decisions fall on a single day so far. A trend '
                "needs decisions spread across more than one day to show anything.</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">VENDOR RISK PROFILES</div>', unsafe_allow_html=True)
    st.caption(
        "Per-vendor risk tracking: total invoices submitted, count flagged for review, "
        "false-positive rate among flagged, and recent activity (flagged count in last 5 invoices)."
    )

    vendor_risks = dashboard.vendor_risk_summary(full_queue)
    if vendor_risks.empty:
        st.markdown(
            '<div class="empty-state">No vendors in the queue yet. Submit a batch or single invoice '
            'to start tracking vendor risk.</div>',
            unsafe_allow_html=True,
        )
    else:
        vendor_risks_display = vendor_risks[
            ["vendor", "total", "flagged", "flagged_rate", "approved_flagged", "recent_flagged"]
        ].copy()
        vendor_risks_display.columns = [
            "Vendor", "Total invoices", "Flagged", "Flag rate", "False positives", "Recent (5)"
        ]
        vendor_risks_display["Flag rate"] = (vendor_risks_display["Flag rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(vendor_risks_display, width='stretch', hide_index=True)

    st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">MODEL RETRAINING</div>', unsafe_allow_html=True)
    st.caption(
        "Human-in-the-loop learning: reviewer decisions feed back into model improvement. "
        "When enough feedback accumulates, retrain the models to learn from corrections."
    )

    readiness = retrain.retraining_readiness()
    r1, r2, r3 = st.columns(3)
    r1.metric("Feedback collected", readiness["total_feedback"])
    r2.metric("Retraining threshold", readiness["threshold"])
    r3.metric("Progress", f"{readiness['progress']:.0%}")

    if readiness["ready"]:
        st.success(f"✓ Ready to retrain ({readiness['total_feedback']} feedback samples collected)")
        if st.button("Retrain models"):
            with st.spinner("Training models on reviewer feedback..."):
                result = retrain.train_retrained_models()
                if result:
                    path = retrain.save_retrained_models(result)
                    st.success(
                        f"Models retrained successfully!\n\n"
                        f"- Samples: {result['metadata']['training_samples']}\n"
                        f"- Models: {', '.join(result['metadata']['models_trained'])}\n"
                        f"- Saved to: {path}"
                    )
                else:
                    st.error("Failed to retrain models. Insufficient or invalid feedback data.")
    else:
        remaining = readiness["threshold"] - readiness["total_feedback"]
        st.info(
            f"Collect {remaining} more reviewer decision(s) to enable retraining. "
            f"Approve or reject items in the Review Queue to provide feedback."
        )

    animate_kpi_counters()


# ---------------------------------------------------------------------------
# TAB 5 — Audit Trail
# ---------------------------------------------------------------------------
with tab5:
    st.markdown('<div class="section-label">APPEND-ONLY EVENT LOG</div>', unsafe_allow_html=True)
    st.caption(
        "Every intake, routing decision and reviewer action, in order. Each entry is "
        "hash-chained to the one before it, so an edited or deleted row breaks the chain "
        "and is reported below rather than passing silently."
    )

    chain_ok, chain_msg = audit_store.verify_chain()
    st.markdown(
        f'<div class="chain-status {"chain-ok" if chain_ok else "chain-bad"}">'
        f'{"&#10003;" if chain_ok else "&#9888;"} {chain_msg}</div>',
        unsafe_allow_html=True,
    )

    events_df = audit_store.load_events(limit=1000)

    if events_df.empty:
        st.markdown(
            '<div class="empty-state">No events recorded yet. Run a batch to start the trail.</div>',
            unsafe_allow_html=True,
        )
    else:
        g1, g2 = st.columns(2)
        invoice_ids = ["All invoices"] + sorted(events_df["invoice_id"].dropna().unique().tolist())
        chosen_invoice = g1.selectbox("Invoice", invoice_ids)
        event_types = sorted(events_df["event_type"].unique().tolist())
        chosen_types = g2.multiselect("Event type", event_types, default=event_types)

        view = events_df[events_df["event_type"].isin(chosen_types)]
        if chosen_invoice != "All invoices":
            view = view[view["invoice_id"] == chosen_invoice]

        st.caption(f"Showing {len(view)} of {len(events_df)} event(s), newest first.")

        rows_html = ""
        for _, e in view.head(200).iterrows():
            transition = ""
            if e["to_status"]:
                if e["from_status"] and e["from_status"] != e["to_status"]:
                    transition = (f"{STATUS_META.get(e['from_status'], {}).get('label', e['from_status'])}"
                                  f" &rarr; {STATUS_META.get(e['to_status'], {}).get('label', e['to_status'])}")
                else:
                    transition = STATUS_META.get(e["to_status"], {}).get("label", e["to_status"])
            detail = json.loads(e["detail_json"] or "{}")
            note = detail.get("note") or detail.get("reason") or ""
            rows_html += f"""
            <tr>
                <td class="hash-cell">#{e['seq']}</td>
                <td class="hash-cell">{str(e['ts'])[:19].replace('T', ' ')}</td>
                <td>{cell(e['invoice_id'])}</td>
                <td class="event-type">{e['event_type']}</td>
                <td style="font-family:'Inter',sans-serif;">{cell(e['actor'])}</td>
                <td class="hash-cell">{transition or 'N/A'}</td>
                <td style="font-family:'Inter',sans-serif;color:var(--text-1);">{note or 'N/A'}</td>
                <td class="hash-cell">{e['entry_hash'][:10]}…</td>
            </tr>
            """
        render_html(
            f"""
            <table class="ledger-table">
                <thead><tr>
                    <th>Seq</th><th>Timestamp (UTC)</th><th>Invoice</th><th>Event</th>
                    <th>Actor</th><th>Transition</th><th>Note</th><th>Hash</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """
        )
        if len(view) > 200:
            st.caption("Table truncated to the 200 most recent events. Download for the full log.")

        st.download_button(
            "Download audit log (.csv)",
            view.to_csv(index=False).encode("utf-8"),
            "audit_trail.csv",
            "text/csv",
        )


# ---------------------------------------------------------------------------
# TAB 6 — Model Comparison
# ---------------------------------------------------------------------------
with tab6:
    st.markdown('<div class="section-label">CLASSICAL VS. QUANTUM KERNEL</div>', unsafe_allow_html=True)

    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics_df, width='stretch', hide_index=True)

        fig_path = os.path.join(BASE_DIR, "outputs", "figures", "metrics_comparison.png")
        if os.path.exists(fig_path):
            st.image(fig_path, caption="F1 score and training time by model")

        cm_path = os.path.join(BASE_DIR, "outputs", "figures", "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion matrices on the held-out test set")

    st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
    render_html(
        """
        <div class="ledger-sub">
        <strong style="color: var(--text-0);">Honest takeaway:</strong>
        the classical RBF-SVM still edges the quantum kernel SVM on F1 while training roughly
        675x faster; they tie on accuracy, trading precision against recall. With a 12-invoice
        test set a single invoice moves accuracy by 8 points, so differences this size sit
        inside the noise. Quantum kernel matrices cost O(n&sup2;) circuit evaluations, and
        near-term simulators carry heavy per-circuit overhead classical kernels don't. This is
        reported as-is rather than adjusted to favor either approach.
        </div>
        """
    )

st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)
st.caption(
    f"QML mini-project · risk classification, approval routing and audit trail for invoice "
    f"workflows · policy v{approval_rules.POLICY_VERSION} · {stats['events']} audit event(s) on file"
)
