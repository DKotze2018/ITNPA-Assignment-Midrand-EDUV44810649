import re
import unicodedata
import joblib
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

extra_stop = {"abstract", "paper", "article", "study", "studies", "research", "work", "present", "presents",
              "propose", "proposed", "result", "results", "conclusion", "conclusions", "method", "methods",
              "background", "objective", "objectives", "purpose", "copyright", "wiley", "sons", "ltd",
              "elsevier", "springer", "rights", "reserved", "author", "authors", "journal", "doi"}
STOP = set(ENGLISH_STOP_WORDS) | extra_stop
KEEP = {"series", "species", "bias", "gas", "news", "analysis", "basis", "hypothesis", "economics", "statistics", "dynamics", "physics"}
IRREG = {"analyses": "analysis", "hypotheses": "hypothesis", "crises": "crisis", "bases": "basis"}


def lemma(w):
    if w in KEEP or len(w) <= 3:
        return w
    if w in IRREG:
        return IRREG[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "ics")):
        return w[:-1]
    return w


def clean_text(text):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"<[^>]+>|&[a-z]+;|&#\d+;", " ", text)
    t = re.sub(r"(copyright\s*)?(©|\(c\)).{0,80}?(ltd\.?|inc\.?|elsevier|springer|wiley|ieee|all rights reserved)\.?", " ", t, flags=re.I)
    t = re.sub(r"(https?://\S+|www\.\S+|\b10\.\d{4,}/\S+)", " ", t, flags=re.I)
    t = re.sub(r"\S+@\S+\.\S+", " ", t)
    t = re.sub(r"^\s*(abstract|summary)\s*[:.\-]?\s*", "", t, flags=re.I)
    t = re.sub(r"\b(background|objectives?|purpose|methods?|results?|conclusions?|aims?|findings)\s*:", " ", t, flags=re.I)
    t = re.sub(r"\$[^$]{0,200}\$|\\[a-zA-Z]+\{?[^}\s]*\}?", " ", t)
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii").lower()
    t = t.replace("-", " ").replace("/", " ")
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = [lemma(w) for w in t.split() if len(w) >= 3 and w not in STOP]
    return " ".join(tok for tok in tokens if tok not in STOP and len(tok) >= 3)


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("Researchly Research Abstract Classifier")
st.write("This app classifies research abstracts as either AI-related or classical research.")

abstract = st.text_area("Enter research abstract:", height=220)

if st.button("Classify Abstract"):
    cleaned = clean_text(abstract)
    if len(cleaned.split()) < 5:
        st.warning("Please enter a longer abstract.")
    else:
        p_ai = float(model.predict_proba([cleaned])[0, 1])
        st.subheader("Prediction")
        if p_ai >= 0.5:
            st.success("The abstract is classified as: artificial intelligence")
        else:
            st.success("The abstract is classified as: classical research")
        st.caption(f"Probability of AI-related: {p_ai:.1%}")
        st.progress(p_ai)
