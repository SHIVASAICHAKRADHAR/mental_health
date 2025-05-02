import streamlit as st
from model_utils import predict_emotion
import tempfile
import os

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="🧠 Mental Health Detector", layout="centered")

st.title("🧠 Mental Health Detection from Voice + Text")
st.markdown("Welcome to the **Mental Health Detector** app! This tool analyzes your **emotional state** based on **text** and **voice** input.")
st.markdown("---")

# ---------------------- Text Input ----------------------
st.markdown("### 💬 Step 1: Enter how you're feeling (optional)")
user_text = st.text_input("Example: 'I’m feeling hopeful today.'")

# ---------------------- Audio Upload ----------------------
st.markdown("### 🎤 Step 2: Upload your voice (optional)")
uploaded_file = st.file_uploader(
    "Choose an audio file (wav, mp3, mp4, m4a)", 
    type=["wav", "mp3", "mp4", "m4a"]
)

# ---------------------- Analyze Button ----------------------
st.markdown("### 🚀 Step 3: Click Analyze")
if st.button("🔍 Analyze"):
    if not user_text and not uploaded_file:
        st.warning("⚠️ Please provide at least **text** or **voice input**.")
    else:
        audio_path = None

        # Save uploaded audio to a temp file
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name

        # Predict emotion using available inputs
        with st.spinner("Analyzing your emotional state..."):
            try:
                label, confidence = predict_emotion(user_text if user_text else None, audio_path if audio_path else None)
            except Exception as e:
                st.error(f"Error in processing: {str(e)}")
                if audio_path:
                    os.remove(audio_path)
                st.stop()

        # Display results
        st.success(f"🧠 **Predicted Emotion:** `{label}`")
        st.info(f"📊 **Confidence Score:** `{confidence:.2f}`")

        # Optional: Color-coded highlight box
        st.markdown("### 🎨 Emotion Indicator")
        st.markdown(
            f"""
            <div style='padding:1rem; background-color:#e0f7fa; border-radius:10px; font-size:18px'>
                You seem to be feeling <strong>{label}</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Clean up
        if audio_path:
            os.remove(audio_path)

# ---------------------- Footer ----------------------
st.markdown("---")
st.caption("🔒 This app runs locally. No personal data is stored or transmitted.")
