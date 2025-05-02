import streamlit as st
from model_utils import predict_emotion
from pydub import AudioSegment
import tempfile
import os

st.set_page_config(page_title="🧠 Mental Health Detector", layout="centered")
st.title("🧠 Mental Health Detection from Voice + Text")
st.markdown("Predict emotions from your **voice** and/or **text** input.")

# Text input
st.markdown("### 💬 Enter a sentence about how you feel (optional):")
user_text = st.text_input("Example: 'I'm feeling great today!'")

# Audio upload
st.markdown("### 🎤 Upload a voice file (.wav, .mp4, .m4a) (optional):")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp4", "m4a"])

# Voice recording (optional component)
from streamlit_audiorecorder import audiorecorder
audio_bytes = audiorecorder("Click to record", "Recording...")
#You can implement this with a compatible plugin if needed.

if st.button("🔍 Analyze"):
    if not user_text and not uploaded_file:
        st.warning("Please enter some text or upload an audio file.")
    else:
        audio_path = None

        # Handle audio file if provided
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
                temp_input.write(uploaded_file.read())
                temp_input.close()

                # Convert to .wav using pydub
                try:
                    sound = AudioSegment.from_file(temp_input.name)
                    sound.export(tmp.name, format="wav")
                    audio_path = tmp.name
                except Exception as e:
                    st.error(f"Audio conversion failed: {e}")
                    os.remove(temp_input.name)
                    audio_path = None
                os.remove(temp_input.name)

        # Run prediction
        with st.spinner("Analyzing..."):
            try:
                label, confidence = predict_emotion(user_text if user_text else None, audio_path)
                st.success(f"🧠 **Predicted Emotion:** `{label}`")
                st.info(f"📊 **Confidence:** `{confidence:.2f}`")
                st.markdown(f"<div style='padding:1rem; background-color:#e0f7fa; border-radius:10px;'>You seem to be feeling <strong>{label}</strong>.</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            finally:
                if audio_path:
                    os.remove(audio_path)
