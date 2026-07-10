import re
import joblib
import streamlit as st
import feedparser

# ---------------------------------------------------------------------------
# Must match the cleaning function used in your training notebook EXACTLY.
# If you changed clean_text() later in the notebook, mirror it here.
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
    # LinearSVC has no predict_proba by default; decision_function gives
    # a confidence-like margin instead.
    score = model.decision_function(X)[0]
    return pred, score


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Fake News Classifier", page_icon="📰")
st.title("📰 Fake News Classifier")
st.caption("TF-IDF + Linear SVM, trained in your notebook")

vectorizer, model = load_artifacts()

tab1, tab2 = st.tabs(["Paste an article", "Test on live headlines"])

with tab1:
    title = st.text_input("Headline / title")
    text = st.text_area("Article body", height=200)
    if st.button("Classify", key="manual"):
        if not title and not text:
            st.warning("Paste a title or article body first.")
        else:
            pred, score = predict(title, text, vectorizer, model)
            label = "🟥 FAKE" if pred == 1 else "🟩 REAL"
            # NOTE: flip the pred==1 check above if your label encoding
            # is the other way around (check y_train.value_counts() / the
            # label mapping cell in your notebook).
            st.subheader(label)
            st.write(f"Decision margin: `{score:.3f}` (further from 0 = more confident)")

with tab2:
    st.write("Pulls current headlines from a few RSS feeds and classifies each one.")
    feeds = {
        "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Reuters World": "https://feeds.reuters.com/Reuters/worldNews",
        "NPR News": "https://feeds.npr.org/1001/rss.xml",
    }
    chosen = st.multiselect("Feeds", list(feeds.keys()), default=["BBC World"])
    if st.button("Fetch & classify", key="live"):
        for name in chosen:
            st.markdown(f"### {name}")
            parsed = feedparser.parse(feeds[name])
            for entry in parsed.entries[:8]:
                headline = entry.get("title", "")
                summary = entry.get("summary", "")
                pred, score = predict(headline, summary, vectorizer, model)
                label = "🟥 FAKE" if pred == 1 else "🟩 REAL"
                st.write(f"{label}  ·  margin `{score:.2f}`  ·  {headline}")
