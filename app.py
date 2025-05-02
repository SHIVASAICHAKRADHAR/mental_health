import streamlit as st
from model_utils import predict_emotion
import tempfile
import os

st.set_page_config(page_title="Mental Health Detector", layout="centered")

st.title("🧠 Mental Health Detection from Voice + Text")
st.markdown("Analyze your **emotions** through text and voice input.")

st.markdown("### 💬 Enter a short sentence about how you're feeling:")
user_text = st.text_input("Example: 'I'm feeling a bit anxious today.'")

st.markdown("### 🎤 Upload your voice clip (.wav format):")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav","mp4","m4a"])

if st.button("🔍 Analyze"):
    if not user_text or not uploaded_file:
        st.warning("Please provide both text and audio input.")
    else:
        # Save uploaded audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name

        # Predict emotion
        with st.spinner("Analyzing..."):
            label, confidence = predict_emotion(user_text, audio_path)

        # Display results
        st.success(f"🧠 **Predicted Emotion:** `{label}`")
        st.info(f"📊 **Confidence:** `{confidence:.2f}`")

        # Optional: Color-coded emotion box
        st.markdown("### 🎨 Emotion Indicator")
        st.markdown(f"<div style='padding:1rem; background-color:#e0f7fa; border-radius:10px;'>You seem to be feeling <strong>{label}</strong>.</div>", unsafe_allow_html=True)

        # Clean up temp file
        os.remove(audio_path)

