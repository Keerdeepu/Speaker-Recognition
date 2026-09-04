import os
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# used to create and train your neural network.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Dataset path (relative path)
DATASET_PATH = 'dataset/16000_pcm_speeches'

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

print("Extracting features...")

extracted_features = []
#loop through every speaker
for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)
    # Check whether it is actually a folder
    if os.path.isdir(speaker_path):
        # Find all WAV files
        for audio_file in glob.glob(os.path.join(speaker_path, "*.wav")):
            # Extract MFCC from each audio file
            data = features_extractor(audio_file)
            extracted_features.append([data, speaker])

# Convert to DataFrame
df = pd.DataFrame(extracted_features, columns=['features', 'class'])

# Save CSV
df.to_csv('clean2.csv', index=False)

X = np.array(df['features'].tolist())
y = np.array(df['class'].tolist())

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

num_labels = y.shape[1]

# Build model
model = Sequential()
model.add(Dense(100, input_shape=(40,), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

# Create saved_models folder if not exists
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

checkpointer = ModelCheckpoint(
    filepath='saved_models/audio_classification2.keras',
    verbose=1,
    save_best_only=True
)

print("Training started...")
model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[checkpointer]
)

print("Training complete!")

test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", test_accuracy[1])