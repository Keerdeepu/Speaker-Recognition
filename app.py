from flask import Flask, render_template, request
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

MODEL_PATH = 'saved_models/audio_classification2.keras'
model = load_model(MODEL_PATH)

DATASET_PATH = 'dataset/16000_pcm_speeches'
speakers = sorted(os.listdir(DATASET_PATH))

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/about.html')
def about():
    return render_template('./about.html')

@app.route('/contact.html')
def contact():
    return render_template('./contact.html')

@app.route('/services.html')
def services():
    return render_template('./services.html')

@app.route('/rec.html', methods=["GET", "POST"])
def rec():
    if request.method == "POST":
        audio_file = request.files['file']

        features = features_extractor(audio_file)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        predicted_index = np.argmax(prediction, axis=1)[0]

        speaker_name = speakers[predicted_index]
        msg = f"The Speaker identified is {speaker_name}"

        return render_template('rec.html', msg=msg)

    return render_template('./rec.html')

if __name__ == "__main__":
    app.run(debug=True)