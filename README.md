# Researchly: Research Papers Expert System

Text-mining project for SAnalytics. It has two parts:

- **Classification (Questions 1 and 2):** predicts whether a research abstract is
  **AI-related** or **classical research**, and serves the best model in a web page.
- **Topic analysis (Questions 3 and 4):** summarises five research articles and finds
  their main themes with Latent Dirichlet Allocation (LDA).

This README explains how to run the web app in this `deployment` folder, and how the
rest of the repository is organised.

---

## 1. Repository contents

| Item | What it is |
|---|---|
| `Q1 and Q2` notebook | Loading and denoising the abstracts, feature modelling, baseline model, five models, hyperparameter tuning, and model export |
| `Q3 and Q4` notebook | Article dataframe, preprocessing, TextRank and centroid/MMR summaries, LDA topic modelling, and the intertopic map |
| `lda_intertopic_map.html` | Interactive intertopic distance map (open it in a web browser) |
| `research_abstracts.json` | Dataset of 1,138 labelled research abstracts (`ai` or `classical`) |
| `research_corpus.zip` | The five research articles used in Questions 3 and 4 |
| Project documentation (Word file) | Questions and answers for Questions 1 to 4 |
| `deployment/` | The web app (this folder) |

---

## 2. The web app (Questions 1 and 2)

The app reads a research abstract and classifies it as **artificial intelligence** or
**classical research**, and shows the probability that the abstract is AI-related.

The model is a tuned Linear SVM trained on TF-IDF features of the cleaned abstract,
with calibrated probabilities. The page is built with Streamlit.

### Files in this folder

| File | What it is |
|---|---|
| `app.py` | The Streamlit web page and the text-cleaning function |
| `model.joblib` | The saved, trained model |
| `requirements.txt` | The Python packages the app needs |
| `README.md` | This file |

`model.joblib` and `app.py` must stay in the same folder.

### What you need

- Python 3.9 or newer
- An internet connection the first time, to install the packages

### How to run it

1. Open a terminal (Command Prompt on Windows) and go into this folder:
   ```bash
   cd deployment
   ```
2. Optional but tidy: create and activate a virtual environment.
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac and Linux
   ```
3. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```
5. The page opens in your browser at **http://localhost:8501**. If it does not open,
   copy that address into your browser.

To stop the app, go back to the terminal and press `Ctrl + C`.

### How to use it

1. Paste a research abstract into the text box.
2. Click **Classify Abstract**.
3. Read the prediction and the probability shown under the heading "Prediction".

### Examples to try

AI-style abstract:

> This paper forecasts monthly consumer price inflation in the United States using
> machine learning methods. We train long short-term memory neural networks, random
> forests and gradient boosting models on a large set of macroeconomic indicators, and
> compare their out-of-sample forecasts with a traditional ARIMA benchmark. The neural
> network models give the lowest forecast errors, especially during periods of high
> volatility.

Classical-style abstract:

> This paper studies the persistence of inflation using a Bayesian vector
> autoregression with stochastic volatility. We estimate a Phillips curve for the euro
> area and test whether the relationship between unemployment and inflation has changed
> over time. The results show that inflation expectations play a larger role than the
> output gap in explaining price movements.

---

## 3. How the model was built

1. **Denoising:** removed duplicates, five off-topic physics abstracts, copyright lines,
   web links, maths code, digits, punctuation and stopwords, then reduced plurals to
   their singular form.
2. **Features:** TF-IDF on the cleaned abstract. The `DOI` column was excluded because
   it shows the publisher and not the content.
3. **Baseline:** TF-IDF with Multinomial Naive Bayes.
4. **Models compared:** Logistic Regression, Linear SVM, Complement Naive Bayes, Random
   Forest and Gradient Boosting.
5. **Enhancement:** grid search with cross-validation over the n-gram range, minimum word
   frequency, term weighting, regularisation strength and class weighting.
6. **Deployment:** the best configuration, a tuned Linear SVM with class weights, was refitted on all the
   cleaned abstracts and saved as `model.joblib`.

### Held-out test results (AI class)

| Measure | Baseline (Naive Bayes) | Final model (tuned Linear SVM) |
|---|---|---|
| F1 | 0.30 | 0.82 |
| Recall | 0.17 | 0.78 |

---

## 4. Notes and limits

- The model was trained on abstracts about inflation, where the two classes are papers
  using AI methods and papers using classical econometric methods. It may be less
  reliable on abstracts from other subjects.
- Very short text is rejected. The abstract needs a few meaningful words.
- The test set contains only 46 AI abstracts, so small differences in scores can be
  chance.

---

## 5. Troubleshooting

- **`streamlit` is not recognised:** run `python -m streamlit run app.py` instead.
- **Model file not found:** check that `model.joblib` is in the same folder as `app.py`,
  and that you started the app from inside this folder.
- **Version warning when loading the model:** install the same scikit-learn version that
  was used to train it. The versions are listed in `requirements.txt`.
- **The page does not open:** copy `http://localhost:8501` into your browser by hand.

---

Author: Dian Kotze
