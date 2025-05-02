import streamlit as st
from model_utils import predict_emotion
import os
import tempfile

st.set_page_config(page_title="Mental Health Detector", layout="centered")

st.title("🧠 Mental Health Detection from Voice + Text")
st.markdown("This app predicts your emotional state using both your speech and a short text description.")

# Text Input
user_text = st.text_input("💬 How are you feeling today? (Text input)")

# Audio Input
uploaded_audio = st.file_uploader("🎤 Upload your voice clip (WAV only)", type=["wav"])

# Prediction
if st.button("🔍 Analyze"):
    if not user_text or not uploaded_audio:
        st.warning("Please provide both text and audio input.")
    else:
        with st.spinner("Analyzing..."):
            # Save uploaded audio to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(uploaded_audio.read())
                tmp_audio_path = tmp_audio.name

            # Predict
            try:
                label, confidence = predict_emotion(user_text, tmp_audio_path)
                st.success(f"🧠 Predicted Emotion: **{label}**")
                st.progress(min(int(confidence * 100), 100))
                st.write(f"📊 Confidence Score: `{confidence:.2f}`")
            except Exception as e:
                st.error(f"Error: {e}")

            # Clean up temp file
            os.remove(tmp_audio_path)

st.markdown("---")
st.caption("Built with 💙 using Streamlit, TensorFlow, and Librosa")
