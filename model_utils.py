import numpy as np
import joblib
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------- Load Models & Artifacts -------------------
# Paths (adjust if necessary)
TEXT_MODEL_PATH = "models/text_model.h5"
AUDIO_MODEL_PATH = "models/audio_model.h5"
TOKENIZER_PATH = "artifacts/tokenizer.pkl"
TEXT_ENCODER_PATH = "artifacts/label_encoder_text.pkl"
AUDIO_ENCODER_PATH = "artifacts/label_encoder_audio.pkl"

# Load pretrained models and artifacts
text_model = load_model(TEXT_MODEL_PATH)
audio_model = load_model(AUDIO_MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
text_encoder = joblib.load(TEXT_ENCODER_PATH)
audio_encoder = joblib.load(AUDIO_ENCODER_PATH)

# Check label alignment
assert list(text_encoder.classes_) == list(audio_encoder.classes_), "Label mismatch between models."

# ------------------- Preprocessing -------------------
def preprocess_text(text, max_len=100):
    """
    Tokenizes and pads the input text using pre-fitted tokenizer.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

def preprocess_audio(audio_path, max_len=173):  # adjust max_len as per training
    """
    Loads audio file, extracts MFCC features, and reshapes for model input.
    """
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = librosa.util.fix_length(mfcc, size=max_len, axis=1)
    mfcc = mfcc.T  # Shape: (time, 13)
    return np.expand_dims(mfcc, axis=0)  # Shape: (1, time, 13)

# ------------------- Inference -------------------
def predict_emotion(text_input=None, audio_path=None):
    """
    Predict emotion using available inputs: text, audio, or both.
    Uses late fusion if both are available.
    """
    if not text_input and not audio_path:
        raise ValueError("At least text or audio input must be provided.")

    probs = []

    if text_input:
        text_seq = preprocess_text(text_input)
        text_probs = text_model.predict(text_seq, verbose=0)[0]
        probs.append(text_probs)

    if audio_path:
        audio_seq = preprocess_audio(audio_path)
        audio_probs = audio_model.predict(audio_seq, verbose=0)[0]
        probs.append(audio_probs)

    combined_probs = np.mean(probs, axis=0)  # Late fusion
    predicted_index = np.argmax(combined_probs)
    predicted_label = text_encoder.classes_[predicted_index]
    confidence = float(combined_probs[predicted_index])

    return predicted_label, confidence
