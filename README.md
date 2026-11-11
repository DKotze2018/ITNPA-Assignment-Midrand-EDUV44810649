# Researchly Abstract Classifier: Deployment

A small web app that reads a research abstract and classifies it as
**AI-related** or **classical research**. It is the deployment for Questions 1 and 2
of the Researchly project.

The model is a tuned Linear SVM trained on TF-IDF features of the cleaned abstract,
with calibrated probabilities. The web page is built with Streamlit.

## Files in this folder

| File | What it is |
|---|---|
| `app.py` | The Streamlit web page and the text-cleaning function |
| `model.joblib` | The saved, trained model |
| `requirements.txt` | The Python packages the app needs |
| `README.md` | This file |

## What you need

- Python 3.9 or newer
- An internet connection to install the packages the first time

## How to run it

1. Open a terminal (Command Prompt on Windows) and go into this folder:
   ```bash
   cd deployment
   ```
2. (Optional but tidy) create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac / Linux
   ```
3. Install the packages:
   ```
   import sklearn, numpy, joblib, streamlit
   print("scikit-learn==" + sklearn.__version__)
   print("numpy==" + numpy.__version__)
   print("joblib==" + joblib.__version__)
   print("streamlit==" + streamlit.__version__)
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```
5. Your browser should open at **http://localhost:8501**. If it does not, copy that
   address into your browser.

`model.joblib` and `app.py` must stay in the same folder.

## How to use it

1. Paste a research abstract into the text box.
2. Click **Classify Abstract**.
3. The page shows the prediction ("artificial intelligence" or "classical research")
   and the probability that the abstract is AI-related.

### Examples to try

AI-style abstract:
> This paper forecasts monthly consumer price inflation in the United States using
> machine learning methods. We train long short-term memory neural networks, random
> forests and gradient boosting models on a large set of macroeconomic indicators, and
> compare their out-of-sample forecasts with a traditional ARIMA benchmark.

Classical-style abstract:
> This paper studies the persistence of inflation using a Bayesian vector
> autoregression with stochastic volatility. We estimate a Phillips curve for the euro
> area and test whether the relationship between unemployment and inflation has
> changed over time.

## Model performance (held-out test set)

| Measure | Baseline (Naive Bayes) | Final model (tuned Linear SVM) |
|---|---|---|
| F1, AI class | 0.30 | 0.82 |
| Recall, AI class | 0.17 | 0.78 |

## Notes and limits

- The model was trained on abstracts about inflation, where the two classes are
  papers using AI methods and papers using classical econometric methods. It may be
  less reliable on abstracts from other subjects.
- The abstract needs at least a few meaningful words, and very short text is rejected.

## Troubleshooting

- **`streamlit` is not recognised:** run `python -m streamlit run app.py` instead.
- **Model file not found:** check that `model.joblib` is in the same folder as `app.py`
  and that you started the app from inside this folder.
- **Version warning when loading the model:** install the same scikit-learn version
  that was used to train it (see `requirements.txt`).
