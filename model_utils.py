import numpy as np
import joblib
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and artifacts
text_model = load_model("/content/text_model.h5")  # Update path as needed
audio_model = load_model("/content/audio_model.h5")  # Update path as needed
tokenizer = joblib.load("/content/tokenizer.pkl")  # Update path as needed
text_encoder = joblib.load("/content/label_encoder_text.pkl")  # Update path as needed
audio_encoder = joblib.load("/content/label_encoder_audio.pkl")  # Update path as needed

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, padding='post', maxlen=100)  # Ensure maxlen matches the model's input
    return padded

def preprocess_audio(audio_path):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)  # Load the audio file at its native sample rate
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCC features

    # Transpose to shape (time_steps, features)
    mfcc = mfcc.T

    # Pad or trim to a fixed time length (e.g., 500)
    desired_length = 500
    if mfcc.shape[0] > desired_length:
        mfcc = mfcc[:desired_length, :]
    else:
        mfcc = np.pad(mfcc, ((0, desired_length - mfcc.shape[0]), (0, 0)), mode='constant')

    # Final shape: (1, time_steps, features)
    return np.expand_dims(mfcc, axis=0)

def predict_emotion(text_input, audio_path):
    # Preprocess text and audio inputs
    text_input_seq = preprocess_text(text_input)
    audio_input_seq = preprocess_audio(audio_path)

    # Predict the probabilities of each class from both models
    text_probs = text_model.predict(text_input_seq)[0]  # Shape: (3,)
    audio_probs = audio_model.predict(audio_input_seq)[0]  # Shape: (24,)

    # Option 1: Align text model output to audio model output
    # Assuming the text model has 3 classes, map them to the 24 classes in some way
    # This is just an example: Repeat the text model predictions to match the 24 classes
    text_probs = np.concatenate([text_probs] * 8)  # Example of repeating (3,) -> (24,)

    # Ensure both have the same shape for fusion
    combined_probs = (text_probs + audio_probs) / 2  # Late fusion (average of probabilities)

    # Find the index of the highest confidence
    predicted_index = np.argmax(combined_probs)  # Get the index of the max probability
    predicted_label = audio_encoder.classes_[predicted_index]  # Get the corresponding label from audio encoder
    confidence = combined_probs[predicted_index]  # Get the confidence of the prediction

    return predicted_label, float(confidence)
