# Product Review Sentiment Analyzer

An end-to-end machine-learning project that converts unstructured product reviews into sentiment predictions and presents the result in an interactive Streamlit application.

## Highlights

- input validation and duplicate removal;
- stratified and reproducible train/test split;
- TF-IDF features with word and bigram information;
- logistic-regression classifier with cross-validation;
- classification report and confusion matrix;
- exported preprocessing-and-model pipeline;
- Streamlit interface with prediction confidence.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run text_prediction.py
```

To retrain the model, run `Modelling.ipynb` before launching the app.

## Project structure

- `Modelling.ipynb` — data validation, training and evaluation
- `text_prediction.py` — Streamlit application
- `product_sentiment.xlsx` — demonstration dataset
- `model-aziz-2.pkl` — exported pipeline used by the app

## Limitations

The model is intended as a portfolio demonstration. Its predictions should not be used for operational decisions without dataset-specific validation and monitoring.

## Acknowledgement

Initially developed during a DTSense learning module and subsequently completed, restructured and documented by Muhammad Aziz.
