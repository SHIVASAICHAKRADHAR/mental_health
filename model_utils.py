import numpy as np
import joblib
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and encoders
text_model = load_model("text_model.h5")
audio_model = load_model("audio_model.h5")
tokenizer = joblib.load("tokenizer.pkl")
text_encoder = joblib.load("label_encoder_text.pkl")
audio_encoder = joblib.load("label_encoder_audio.pkl")

# ⚠️ Assumption: text_encoder and audio_encoder use the same label set and order

def preprocess_text(text):
    """
    Tokenize and pad text input for the LSTM model.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    return padded  # Shape: (1, 100)

def preprocess_audio(audio_path):
    """
    Load audio and extract MFCC features for CNN model input.
    """
    max_pad_len = 174  # Ensure same as used during training
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Pad or truncate
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc.T  # Shape: (174, 13)
    mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel: (174, 13, 1)
    mfcc = np.expand_dims(mfcc, axis=0)   # Add batch: (1, 174, 13, 1)
    return mfcc

def predict_emotion(text_input, audio_path):
    """
    Predict emotion from both text and audio inputs using late fusion.
    """
    # Preprocess inputs
    text_input_seq = preprocess_text(text_input)
    audio_input_seq = preprocess_audio(audio_path)

    # Predict
    text_probs = text_model.predict(text_input_seq)[0]
    audio_probs = audio_model.predict(audio_input_seq)[0]

    # Late fusion (average softmax probabilities)
    combined_probs = (text_probs + audio_probs) / 2
    predicted_index = np.argmax(combined_probs)
    predicted_label = text_encoder.classes_[predicted_index]
    confidence = combined_probs[predicted_index]

    return predicted_label, float(confidence)

