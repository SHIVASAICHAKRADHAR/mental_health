import numpy as np
import joblib
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pretrained models and artifacts
text_model = load_model("models/text_model.h5")
audio_model = load_model("models/audio_model.h5")
tokenizer = joblib.load("artifacts/tokenizer.pkl")
text_encoder = joblib.load("artifacts/label_encoder_text.pkl")
audio_encoder = joblib.load("artifacts/label_encoder_audio.pkl")

# Check label alignment
assert list(text_encoder.classes_) == list(audio_encoder.classes_), "Label mismatch between models."

# Function to preprocess the text input
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, padding='post', maxlen=100)  # Adjust maxlen to your model's input length
    return padded

# Function to preprocess the audio input
def preprocess_audio(audio_path):
    audio, _ = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=13)
    mfcc = librosa.util.fix_length(mfcc, size=216)  # Adjust size based on your training
    return np.expand_dims(mfcc.T, axis=0)  # Shape: (1, time, 13)

# Function to predict the emotion from both text and audio input
def predict_emotion(text_input, audio_path):
    # Preprocess the inputs
    text_input_seq = preprocess_text(text_input)
    audio_input_seq = preprocess_audio(audio_path)

    # Predict with both models
    text_probs = text_model.predict(text_input_seq)[0]
    audio_probs = audio_model.predict(audio_input_seq)[0]

    # Late fusion: Combine both models' probabilities (average them)
    combined_probs = (text_probs + audio_probs) / 2  # Simple average
    predicted_index = np.argmax(combined_probs)
    predicted_label = text_encoder.classes_[predicted_index]  # Get label from the encoder
    confidence = combined_probs[predicted_index]  # Confidence score

    return predicted_label, float(confidence)
