from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_PATH = Path(__file__).with_name("model-aziz-2.pkl")

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="📊")
st.title("Product Review Sentiment Analyzer")
st.caption("A TF-IDF text-classification demo for positive and negative product reviews.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run Modelling.ipynb to train and export it.")
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()

text = st.text_area("Review to analyse", placeholder="Enter a product review...", height=140)

if st.button("Analyse sentiment", type="primary"):
    if not text.strip():
        st.warning("Please enter a review before running the analysis.")
    else:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        classes = [str(label) for label in model.classes_]
        result = pd.DataFrame({"class": classes, "probability": probabilities})
        st.success(f"Predicted sentiment: {prediction}")
        fig = px.bar(result, x="class", y="probability", range_y=[0, 1], text_auto=".1%", title="Prediction confidence")
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Model scope and limitations"):
    st.write("The model is a portfolio demonstration trained on the included review dataset. Results may not generalise to other languages, products or platforms.")
