# app.py

import streamlit as st
from model import VibeModel

st.set_page_config(page_title="Vibe Analyzer", page_icon="🔥")

st.title("🔥 Vibe Analyzer")
st.write("Type text and get the vibe instantly")

@st.cache_resource
def load_model():
    return VibeModel()

model = load_model()

text = st.text_area("Your text", placeholder="I feel amazing today...")

if st.button("Analyze vibe"):
    if text.strip() == "":
        st.warning("Type something first 👀")
    else:
        vibe, prob = model.predict(text)

        emoji = {
            "happy": "😄",
            "angry": "😡",
            "neutral": "😐"
        }

        st.success(f"**VIBE:** {vibe.upper()} {emoji.get(vibe, '')}")
        st.info(f"Confidence: **{prob:.2f}**")
