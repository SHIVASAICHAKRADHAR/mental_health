import numpy as np
import joblib
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants


# Load models and artifacts
text_model = load_model("models/text_model.h5")
audio_model = load_model("models/audio_model.h5")

tokenizer = joblib.load("artifacts/tokenizer.pkl")
text_encoder = joblib.load("artifacts/label_encoder_text.pkl")
audio_encoder = joblib.load("artifacts/label_encoder_audio.pkl")

# ⚠️ Ensure both encoders are trained on the same class set.
# Use text_encoder.classes_ to map back to final labels.

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq)
    return padded

def preprocess_audio(audio_path):
    audio, _ = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=13)
    mfcc = librosa.util.fix_length(mfcc, axis=1)
    return np.expand_dims(mfcc.T, axis=0)  # Shape: (1, time, 13)

def predict_emotion(text_input, audio_path):
    # Preprocess
    text_input_seq = preprocess_text(text_input)
    audio_input_seq = preprocess_audio(audio_path)

    # Predict
    text_probs = text_model.predict(text_input_seq)[0]
    audio_probs = audio_model.predict(audio_input_seq)[0]

    # Late fusion (average)
    combined_probs = (text_probs + audio_probs) / 2
    predicted_index = np.argmax(combined_probs)
    predicted_label = text_encoder.classes_[predicted_index]  # assuming same classes
    confidence = combined_probs[predicted_index]

    return predicted_label, float(confidence)

