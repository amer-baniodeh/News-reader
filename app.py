import re
import time
import joblib
import streamlit as st
import feedparser

# ---------------------------------------------------------------------------
# Must match the cleaning function used in your training notebook EXACTLY.
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(
        r'^[A-Z\s\/\-]+\s*\(Reuters\)\s*[-–]\s*',
        '',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("svm_model.pkl")
    return vectorizer, model


def predict(title: str, text: str, vectorizer, model):
    combined = clean_text(title) + " " + clean_text(text)
    X = vectorizer.transform([combined])
    pred = model.predict(X)[0]
    score = model.decision_function(X)[0]
    return pred, score


def confidence_pct(score: float) -> int:
    """Squash the raw SVM margin into a 0-100 display bar. Not a real
    probability -- just a readable stand-in for 'how far from the line'."""
    import math
    squashed = 1 / (1 + math.exp(-abs(score)))
    return int(50 + squashed * 50) if abs(score) > 0 else 50


# ---------------------------------------------------------------------------
# Page config + global styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="WIRE-CHECK // Fake News Terminal", page_icon="\U0001F4E1", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg-void: #0B0E0C;
    --panel: #12160F;
    --panel-border: #2A3324;
    --phosphor: #B9D9A8;
    --phosphor-dim: #6C7E64;
    --ink-red: #C4432E;
    --ink-green: #4C8C5B;
}

.stApp {
    background-color: var(--bg-void);
    color: var(--phosphor);
    font-family: 'IBM Plex Sans', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 1.5rem;
    max-width: 900px;
}

.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: var(--panel);
    border-top: 1px solid var(--panel-border);
    border-bottom: 1px solid var(--panel-border);
    padding: 6px 0;
    margin-bottom: 1.5rem;
}
.ticker-text {
    display: inline-block;
    white-space: nowrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    color: var(--phosphor-dim);
    animation: scroll-left 22s linear infinite;
}
@keyframes scroll-left {
    0% { transform: translateX(100vw); }
    100% { transform: translateX(-100%); }
}

.masthead {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: 1px;
    color: var(--phosphor);
    border-bottom: 2px solid var(--phosphor);
    padding-bottom: 6px;
    margin-bottom: 2px;
}
.masthead .cursor {
    display: inline-block;
    width: 10px;
    background: var(--phosphor);
    animation: blink 1.1s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }
.subhead {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--phosphor-dim);
    letter-spacing: 2px;
    margin-bottom: 1.8rem;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1px;
    font-size: 13px;
    color: var(--phosphor-dim);
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 2px;
}
.stTabs [aria-selected="true"] {
    color: var(--bg-void) !important;
    background: var(--phosphor) !important;
}

.stTextInput input, .stTextArea textarea {
    background: var(--panel) !important;
    color: var(--phosphor) !important;
    border: 1px solid var(--panel-border) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--phosphor) !important;
    box-shadow: none !important;
}
label { color: var(--phosphor-dim) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; letter-spacing: 1px; }

.stButton>button {
    background: transparent;
    color: var(--phosphor);
    border: 1px solid var(--phosphor);
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 2px;
    font-size: 12px;
    padding: 0.5rem 1.2rem;
    transition: all 0.15s ease;
}
.stButton>button:hover {
    background: var(--phosphor);
    color: var(--bg-void);
    border-color: var(--phosphor);
}

.stamp-wrap { display: flex; align-items: center; gap: 24px; margin: 1.5rem 0; }
.stamp {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 1.6rem;
    letter-spacing: 3px;
    padding: 10px 22px;
    border: 4px double currentColor;
    border-radius: 6px;
    transform: rotate(-6deg);
    display: inline-block;
    text-transform: uppercase;
}
.stamp.fake { color: var(--ink-red); }
.stamp.real { color: var(--ink-green); }

.meter-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--phosphor-dim);
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.meter-track {
    width: 220px;
    height: 8px;
    background: var(--panel-border);
    border-radius: 4px;
    overflow: hidden;
}
.meter-fill { height: 100%; }
.meter-fill.fake { background: var(--ink-red); }
.meter-fill.real { background: var(--ink-green); }

.wire-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 10px 0;
    border-bottom: 1px dashed var(--panel-border);
    display: flex;
    gap: 12px;
    align-items: baseline;
}
.wire-tag {
    font-weight: 700;
    letter-spacing: 1px;
    min-width: 52px;
}
.wire-tag.fake { color: var(--ink-red); }
.wire-tag.real { color: var(--ink-green); }
.wire-headline { color: var(--phosphor); }
.wire-source { color: var(--phosphor-dim); font-size: 11px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ticker-wrap">
  <div class="ticker-text">
    TFIDF + LINEAR SVM &nbsp;//&nbsp; TRAINED ON REUTERS + LABELED FAKE CORPUS &nbsp;//&nbsp;
    DECISION MARGIN IS A DISTANCE FROM THE BOUNDARY, NOT A PROBABILITY &nbsp;//&nbsp;
    VERIFY AGAINST YOUR OWN JUDGEMENT &nbsp;//&nbsp;
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="masthead">WIRE&#8209;CHECK<span class="cursor">&nbsp;</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subhead">AUTOMATED COPY VERIFICATION TERMINAL &nbsp;\u00b7&nbsp; TF-IDF / LINEAR SVM</div>', unsafe_allow_html=True)

vectorizer, model = load_artifacts()

tab1, tab2 = st.tabs(["SUBMIT COPY", "LIVE WIRE"])

with tab1:
    title = st.text_input("HEADLINE")
    text = st.text_area("ARTICLE BODY", height=220)

    if st.button("RUN VERIFICATION", key="manual"):
        if not title and not text:
            st.warning("Enter a headline or article body first.")
        else:
            with st.spinner(""):
                time.sleep(0.4)
                pred, score = predict(title, text, vectorizer, model)

            is_fake = pred == 1  # flip this if your label encoding is reversed
            pct = confidence_pct(score)
            cls = "fake" if is_fake else "real"
            stamp_text = "FLAGGED" if is_fake else "CLEARED"

            st.markdown(f"""
            <div class="stamp-wrap">
                <div class="stamp {cls}">{stamp_text}</div>
                <div>
                    <div class="meter-label">SIGNAL STRENGTH &nbsp;\u00b7&nbsp; {pct}%</div>
                    <div class="meter-track">
                        <div class="meter-fill {cls}" style="width:{pct}%;"></div>
                    </div>
                    <div class="meter-label" style="margin-top:6px;">RAW MARGIN &nbsp;{score:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="meter-label" style="margin-bottom:14px;">PULL LIVE HEADLINES AND RUN EACH THROUGH THE MODEL</div>', unsafe_allow_html=True)

    feeds = {
        "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Reuters World": "https://feeds.reuters.com/Reuters/worldNews",
        "NPR News": "https://feeds.npr.org/1001/rss.xml",
    }
    chosen = st.multiselect("FEEDS", list(feeds.keys()), default=["BBC World"])

    if st.button("PULL WIRE", key="live"):
        for name in chosen:
            st.markdown(f'<div style="margin-top:1.2rem; font-family:\'IBM Plex Mono\'; letter-spacing:1px; color:var(--phosphor); font-size:13px;">\u2014 {name.upper()} \u2014</div>', unsafe_allow_html=True)
            parsed = feedparser.parse(feeds[name])
            for entry in parsed.entries[:8]:
                headline = entry.get("title", "")
                summary = entry.get("summary", "")
                pred, score = predict(headline, summary, vectorizer, model)
                is_fake = pred == 1
                cls = "fake" if is_fake else "real"
                tag = "FLAG" if is_fake else "CLEAR"
                st.markdown(f"""
                <div class="wire-line">
                    <span class="wire-tag {cls}">{tag}</span>
                    <span class="wire-headline">{headline}</span>
                </div>
                """, unsafe_allow_html=True)
                