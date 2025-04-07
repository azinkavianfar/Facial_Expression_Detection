
import streamlit as st


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from feat import Detector
import cv2
from PIL import Image
import tempfile
from xgboost import XGBClassifier

from io import BytesIO

# Load AU detector
detector = Detector()

# Action Units description (with references)
# Reference: Ekman, P., & Friesen, W. V. (1978). Facial Action Coding System: Investigator’s Guide.
AU_MAP = {
    "AU01": "Inner Brow Raiser",
    "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser",
    "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler",
    "AU10": "Upper Lip Raiser",
    "AU11": "Nasolabial Deepener",
    "AU12": "Lip Corner Puller",
    "AU14": "Dimpler",
    "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser",
    "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener",
    "AU24": "Lip Pressor",
    "AU25": "Lips Part",
    "AU26": "Jaw Drop",
    "AU28": "Lip Suck",
    "AU43": "Eyes Closed",
}

# Emotion-specific AU groups (justified selections)
HAPPINESS_AUs = ["AU06", "AU12", "AU25", "AU26"]
SADNESS_AUs = ["AU01", "AU04", "AU15", "AU17", "AU24"]

# Helper to get AU description
def describe_aus(au_list):
    return [f"{au}: {AU_MAP.get(au, 'Unknown')}" for au in au_list]

# Streamlit App
st.title("Facial AU Detection and Emotional Analysis")

uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    st.markdown("**Image is being analyzed...**")

    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        result = detector.detect(tmp_file.name)

    # Display AU dataframe
    aus_df = result.aus
    if not aus_df.empty:
        st.subheader("Detected Action Units")
        st.dataframe(aus_df.round(3))

        # Check for average confidence
        avg_confidence = aus_df.values.mean()
        if avg_confidence == 0:
            st.warning("⚠️ Average AU confidence is 0. Please verify model compatibility.")
        else:
            st.markdown(f"**Average AU Confidence:** {avg_confidence:.2f}")

        # Visualize most prominent AUs
        top_aus = aus_df.loc[:, aus_df.mean().sort_values(ascending=False).head(5).index]
        st.subheader("Top AU Activations")
        st.bar_chart(top_aus.T)

        # Emotional interpretation
        st.subheader("Emotion-Specific AU Overview")
        st.markdown("**Happiness AUs:**")
        st.write(describe_aus(HAPPINESS_AUs))
        st.markdown("**Sadness AUs:**")
        st.write(describe_aus(SADNESS_AUs))
    else:
        st.error("No Action Units were detected.")

    st.markdown("---")
    st.caption("AU Selection Justification Reference: Ekman & Friesen (1978), FACS Manual.")



