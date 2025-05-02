import streamlit as st
from model_utils import predict_emotion  # Import the prediction function
import tempfile
import os

# Set up Streamlit app configuration
st.set_page_config(page_title="Mental Health Detector", layout="centered")

# App title and description
st.title("🧠 Mental Health Detection from Voice + Text")
st.markdown("Analyze your **emotions** through text and voice input.")

# Text input field
st.markdown("### 💬 Enter a short sentence about how you're feeling:")
user_text = st.text_input("Example: 'I'm feeling a bit anxious today.'")

# Audio input field (upload)
st.markdown("### 🎤 Upload your voice clip (.wav, .m4a, .mp4 formats supported):")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "m4a", "mp4"])

# Analyze button
if st.button("🔍 Analyze"):
    if not user_text or not uploaded_file:
        st.warning("Please provide both text and audio input.")
    else:
        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name

        # Display a spinner while analysis is in progress
        with st.spinner("Analyzing..."):
            label, confidence = predict_emotion(user_text, audio_path)

        # Display results after prediction
        st.success(f"🧠 **Predicted Emotion:** `{label}`")
        st.info(f"📊 **Confidence:** `{confidence:.2f}`")

        # Optional: Display color-coded emotion indicator
        st.markdown("### 🎨 Emotion Indicator")
        emotion_color = {
            "happy": "#e0f7fa",
            "sad": "#ffcccb",
            "neutral": "#f5f5f5",
            "angry": "#ffeb3b"  # Change based on your labels
        }
        st.markdown(f"<div style='padding:1rem; background-color:{emotion_color.get(label, '#f5f5f5')}; border-radius:10px;'>You seem to be feeling <strong>{label}</strong>.</div>", unsafe_allow_html=True)

        # Clean up temp file
        os.remove(audio_path)
