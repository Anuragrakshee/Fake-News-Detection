import html
import json
import os
import re
from datetime import datetime

import joblib
import streamlit as st

# =========================================================
# SETTINGS
# =========================================================
CONFIDENCE_THRESHOLD = 70.0  # below this, the app says "Not sure"
METRICS_FILE = "metrics.json"  # created by evaluate_model.py

st.set_page_config(page_title="TruthLens AI", page_icon="🛡️", layout="centered")


# =========================================================
# LOAD MODEL + METRICS (cached)
# =========================================================
@st.cache_resource
def load_artifacts():
    return joblib.load("fake_news_model.pkl"), joblib.load("tfidf_vectorizer.pkl")


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE) as f:
        return json.load(f)


model, vectorizer = load_artifacts()
metrics = load_metrics()


# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@400;500;700&display=swap');

:root {
    --bg: #0b1220; --panel: #121b2e; --line: #243049; --text: #e8edf6; --muted: #8a97b1;
    --fake: #ff6b6b; --real: #3ddc97; --warn: #f5b84b; --accent: #7aa2ff;
}

html, body, .stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 820px; padding-top: 2.5rem; padding-bottom: 4rem; }

.brand { display:flex; align-items:center; justify-content:space-between; color: var(--muted); font-size: 14px; }
.brand b { color: var(--text); font-size: 16px; }
.status { display:inline-flex; align-items:center; gap:8px; }
.dot { width:8px; height:8px; border-radius:50%; background: var(--real); box-shadow: 0 0 0 4px rgba(61,220,151,.15); }
.hero { font-family:'Instrument Serif', serif; font-size: 50px; line-height: 1.1; margin: 48px 0 12px; letter-spacing: -0.5px; }
.hero em { font-style: italic; color: var(--accent); }
.hero-sub { color: var(--muted); font-size: 17px; max-width: 100%; line-height: 1.55; margin-bottom: 28px; }

.stTextArea textarea {
    background: var(--panel) !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 14px !important;
    font-size: 16px !important; line-height: 1.6 !important; padding: 18px !important;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(122,162,255,.18) !important; }

.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: 1px solid var(--line); background: var(--panel);
    color: var(--text); font-weight: 500; padding: 10px 18px; transition: border-color .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--text); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: var(--accent); color: #0b1220; border: none; font-weight: 700; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color:#0b1220; }

.verdict { background: var(--panel); border:1px solid var(--line); border-radius: 18px; padding: 28px; margin-top: 36px; }
.verdict.fake { border-left: 6px solid var(--fake); }
.verdict.real { border-left: 6px solid var(--real); }
.verdict.unsure { border-left: 6px solid var(--warn); }
.verdict-label { color: var(--muted); font-size: 14px; }
.verdict-text { font-family:'Instrument Serif', serif; font-size: 52px; line-height:1.1; margin: 4px 0 6px; }
.verdict.fake .verdict-text { color: var(--fake); }
.verdict.real .verdict-text { color: var(--real); }
.verdict.unsure .verdict-text { color: var(--warn); }
.verdict-conf { color: var(--muted); }
.verdict-conf b { color: var(--text); }
.unsure-note { margin-top: 14px; padding: 12px 14px; border-radius: 10px; background: rgba(245,184,75,.10);
    color: #f6d9a0; font-size: 14px; line-height: 1.5; }

.split { display:flex; height: 14px; border-radius: 999px; overflow:hidden; margin-top: 22px; background: var(--line); }
.split .f { background: var(--fake); } .split .r { background: var(--real); }
.split-legend { display:flex; justify-content:space-between; margin-top:8px; font-size:14px; color: var(--muted); }
.split-legend b.f { color: var(--fake); } .split-legend b.r { color: var(--real); }

.h2 { font-family:'Instrument Serif', serif; font-size: 30px; margin: 44px 0 6px; }
.h2-sub { color: var(--muted); font-size: 15px; margin-bottom: 16px; line-height:1.5; }

.article { background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:20px; line-height:1.9; color:#c6cfe0; }
.article mark.fake { background: rgba(255,107,107,.22); color:#ffd0d0; padding:1px 4px; border-radius:4px; }
.article mark.real { background: rgba(61,220,151,.20); color:#c9f6e1; padding:1px 4px; border-radius:4px; }

.col-title { font-weight:700; margin-bottom:10px; }
.wrow { margin-bottom: 10px; }
.wrow-top { display:flex; justify-content:space-between; font-size:14px; margin-bottom:4px; }
.wrow-top span:last-child { color: var(--muted); }
.wbar { height:6px; background: var(--line); border-radius:999px; overflow:hidden; }
.wbar div { height:100%; border-radius:999px; }
.wbar .f { background: var(--fake); } .wbar .r { background: var(--real); }

.stat { background: var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.stat b { font-size: 26px; display:block; }
.stat span { color: var(--muted); font-size: 13px; }

.cm { width:100%; border-collapse: separate; border-spacing: 6px; text-align:center; margin-top: 8px; }
.cm th { color: var(--muted); font-weight: 500; font-size: 13px; padding: 4px; }
.cm td { border-radius: 10px; padding: 16px 8px; font-size: 24px; font-weight: 700; }
.cm td small { display:block; font-size: 12px; font-weight: 400; color: var(--muted); margin-top: 2px; }
.cm td.ok  { background: rgba(61,220,151,.14); color: var(--real); }
.cm td.bad { background: rgba(255,107,107,.12); color: var(--fake); }

.note { margin-top: 36px; padding: 16px 18px; border:1px dashed var(--line); border-radius:12px; color: var(--muted); font-size:14px; line-height:1.6; }

div[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; margin-top: 40px; }

@media (max-width: 640px) { .hero { font-size: 40px; } .verdict-text { font-size: 40px; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def word_rows(words, scores, css):
    if not words:
        return '<div class="h2-sub">No strong signals found.</div>'
    top = max(scores) or 1
    rows = ""
    for w, s in zip(words, scores):
        rows += (
            f'<div class="wrow"><div class="wrow-top"><span>{html.escape(w)}</span>'
            f"<span>{s:.3f}</span></div>"
            f'<div class="wbar"><div class="{css}" style="width:{s / top * 100:.0f}%"></div></div></div>'
        )
    return rows


def highlight(text, fake_words, real_words):
    out = []
    for token in text.split():
        key = re.sub(r"[^a-z]", "", token.lower())
        safe = html.escape(token)
        if key in fake_words:
            out.append(f'<mark class="fake">{safe}</mark>')
        elif key in real_words:
            out.append(f'<mark class="real">{safe}</mark>')
        else:
            out.append(safe)
    return " ".join(out)


def build_report(r) -> str:
    """Plain-text report for the download button."""
    lines = [
        "TRUTHLENS AI - ANALYSIS REPORT",
        f"Generated: {r['time']}",
        "=" * 50,
        f"Verdict:        {r['verdict']}",
        f"Confidence:     {r['confidence']:.2f}%",
        f"Fake prob.:     {r['fake_p']:.2f}%",
        f"Real prob.:     {r['real_p']:.2f}%",
        "",
        "Words pushing toward FAKE:",
        *[f"  - {w} ({s:.4f})" for w, s in r["fake_words"]],
        "",
        "Words pushing toward REAL:",
        *[f"  - {w} ({s:.4f})" for w, s in r["real_words"]],
        "",
        f"Article stats: {r['words']} words, {r['chars']} characters, {r['sentences']} sentences",
        "",
        "Article (first 600 characters):",
        r["text"][:600],
        "",
        "-" * 50,
        "Note: this tool does not fact-check claims. It shows how the wording",
        "compares with patterns the model learned from its training data.",
    ]
    return "\n".join(lines)


def analyze_article(news: str):
    """Run the model and return a dict of results, or an error string."""
    cleaned = clean_text(news)
    if not cleaned:
        return "The text has no readable words. Paste a longer article."

    X = vectorizer.transform([cleaned])
    fake_p, real_p = model.predict_proba(X)[0] * 100
    confidence = max(fake_p, real_p)
    leans_fake = fake_p > real_p

    names = vectorizer.get_feature_names_out()
    values = X.toarray()[0]
    contrib = values * model.coef_[0]  # negative -> fake, positive -> real
    present = values > 0
    order = contrib.argsort()
    fake_idx = [i for i in order if present[i] and contrib[i] < 0][:8]
    real_idx = [i for i in order[::-1] if present[i] and contrib[i] > 0][:8]

    if confidence < CONFIDENCE_THRESHOLD:
        verdict = "Not sure"
    else:
        verdict = "Likely fake" if leans_fake else "Likely real"

    return {
        "text": news,
        "time": datetime.now().strftime("%d %b %Y, %H:%M"),
        "verdict": verdict,
        "state": "unsure" if confidence < CONFIDENCE_THRESHOLD else ("fake" if leans_fake else "real"),
        "leans": "fake" if leans_fake else "real",
        "confidence": confidence,
        "fake_p": fake_p,
        "real_p": real_p,
        "fake_words": [(names[i], abs(contrib[i])) for i in fake_idx],
        "real_words": [(names[i], abs(contrib[i])) for i in real_idx],
        "words": len(news.split()),
        "chars": len(news),
        "sentences": len(re.findall(r"[.!?]+", news)),
    }


# ---------------------- callbacks ----------------------
SAMPLES = {
    "Try a sample": "",
    "Sample 1": "Government officials announced a new infrastructure bill on Tuesday after weeks of negotiations "
    "between lawmakers, with funding set to be distributed to state transportation departments over five years.",
    "Sample 2": "SHOCKING!!! Secret document LEAKED, they don't want you to know the truth about what is hidden in "
    "your tap water. Share this before it gets deleted!",
}


def load_sample():
    st.session_state["news"] = SAMPLES[st.session_state["sample"]]
    st.session_state["result"] = None


def clear_all():
    st.session_state["news"] = ""
    st.session_state["sample"] = "Try a sample"
    st.session_state["result"] = None
    st.session_state["error"] = None


def run_analysis():
    text = st.session_state["news"]
    st.session_state["error"] = None
    if not text.strip():
        st.session_state["result"] = None
        st.session_state["error"] = "Paste an article above, then select Analyze article."
        return
    out = analyze_article(text)
    if isinstance(out, str):
        st.session_state["result"] = None
        st.session_state["error"] = out
    else:
        st.session_state["result"] = out


st.session_state.setdefault("result", None)
st.session_state.setdefault("error", None)


# =========================================================
# HEADER + HERO
# =========================================================
st.markdown(
    '<div class="brand"><b>🛡️ TruthLens AI</b>'
    '<span class="status"><span class="dot"></span>Model online</span></div>'
    '<div class="hero">Detect misinformation <em>with AI.</em></div>'
    '<div class="hero-sub">Analyze news articles using Natural Language Processing '
    "and Machine Learning.</div>",
    unsafe_allow_html=True,
)

# =========================================================
# INPUT
# =========================================================
st.selectbox("Sample", list(SAMPLES.keys()), key="sample", on_change=load_sample, label_visibility="collapsed")
st.text_area(
    "News article",
    key="news",
    height=220,
    placeholder="Paste a headline or the full article here...",
    label_visibility="collapsed",
)

btn_analyze, btn_clear = st.columns([4, 1])
btn_analyze.button("Analyze article", type="primary", use_container_width=True, on_click=run_analysis)
btn_clear.button("Clear", use_container_width=True, on_click=clear_all)

if st.session_state["error"]:
    st.warning(st.session_state["error"])

# =========================================================
# RESULTS
# =========================================================
r = st.session_state["result"]

if r:
    # ---- Verdict --------------------------------------------------
    if r["state"] == "unsure":
        sub = f"Leans {r['leans']} at {r['confidence']:.1f}%, below the {CONFIDENCE_THRESHOLD:.0f}% needed for a firm verdict."
        note = (
            '<div class="unsure-note">The model is unsure about this one. '
            "Try pasting the full article, or check other sources before trusting this result.</div>"
        )
    else:
        sub = f"Confidence <b>{r['confidence']:.1f}%</b>"
        note = ""

    st.markdown(
        f"""<div class="verdict {r['state']}">
<div class="verdict-label">The model's verdict</div>
<div class="verdict-text">{r['verdict']}</div>
<div class="verdict-conf">{sub}</div>
<div class="split"><div class="f" style="width:{r['fake_p']:.2f}%"></div><div class="r" style="width:{r['real_p']:.2f}%"></div></div>
<div class="split-legend"><span><b class="f">{r['fake_p']:.1f}%</b> fake</span><span><b class="r">{r['real_p']:.1f}%</b> real</span></div>
{note}
</div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.download_button(
        "Download report",
        data=build_report(r),
        file_name=f"truthlens_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    # ---- Explanation ----------------------------------------------
    st.markdown(
        '<div class="h2">Why this verdict</div>'
        '<div class="h2-sub">Words that pushed the score toward fake or real. '
        "Longer bars mean a stronger push.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="col-title">Pushes toward fake</div>'
            + word_rows([w for w, _ in r["fake_words"]], [s for _, s in r["fake_words"]], "f"),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="col-title">Pushes toward real</div>'
            + word_rows([w for w, _ in r["real_words"]], [s for _, s in r["real_words"]], "r"),
            unsafe_allow_html=True,
        )

    # ---- Highlighted text -----------------------------------------
    st.markdown(
        '<div class="h2">Your article, highlighted</div>'
        '<div class="h2-sub">Red leans fake. Green leans real.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="article">{highlight(r["text"], {w for w, _ in r["fake_words"]}, {w for w, _ in r["real_words"]})}</div>',
        unsafe_allow_html=True,
    )

    # ---- Stats ------------------------------------------------------
    st.markdown('<div class="h2">Article details</div>', unsafe_allow_html=True)
    for col, (label, val) in zip(
        st.columns(3),
        [("Words", r["words"]), ("Characters", r["chars"]), ("Sentences", r["sentences"])],
    ):
        col.markdown(f'<div class="stat"><b>{val:,}</b><span>{label}</span></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="note"><b>Keep in mind:</b> TruthLens does not fact-check claims. It uses TF-IDF and '
        "Logistic Regression to compare the article's wording with patterns in its training data. "
        "A high score does not prove a claim is true or false.</div>",
        unsafe_allow_html=True,
    )

# =========================================================
# ABOUT THE MODEL (metrics + confusion matrix)
# =========================================================
with st.expander("About the model: accuracy and confusion matrix"):
    if not metrics:
        st.info("No metrics found yet. Run `python evaluate_model.py` once, then refresh this page.")
    else:
        st.markdown(
            f'<div class="h2-sub">{html.escape(metrics["model"])}, tested on '
            f'{metrics["test_samples"]:,} articles the model did not train on.</div>',
            unsafe_allow_html=True,
        )
        for col, (label, key) in zip(
            st.columns(4),
            [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1")],
        ):
            col.markdown(
                f'<div class="stat"><b>{metrics[key] * 100:.2f}%</b><span>{label}</span></div>',
                unsafe_allow_html=True,
            )

        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        total = tn + fp + fn + tp
        pct = lambda v: f"{v / total * 100:.1f}% of tests"
        st.markdown(
            f"""<div class="h2-sub" style="margin-top:22px;">Confusion matrix: green cells are correct predictions, red cells are mistakes.</div>
<table class="cm">
<tr><th></th><th>Predicted fake</th><th>Predicted real</th></tr>
<tr><th>Actually fake</th><td class="ok">{tn:,}<small>{pct(tn)}</small></td><td class="bad">{fp:,}<small>{pct(fp)}</small></td></tr>
<tr><th>Actually real</th><td class="bad">{fn:,}<small>{pct(fn)}</small></td><td class="ok">{tp:,}<small>{pct(tp)}</small></td></tr>
</table>""",
            unsafe_allow_html=True,
        )