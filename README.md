Speech Emotion Recognition using TESS Toronto Emotional Speech Set
This project demonstrates the process of speech emotion recognition using the TESS Toronto emotional speech set data. It involves loading and visualizing audio data, extracting features, and building a machine learning model to classify emotions based on speech signals.

Table of Contents
Installation
Dataset
Data Loading and Preprocessing
Visualization
Feature Extraction
Model Training
Evaluation
Results
License
Installation
To get started, clone the repository and install the necessary dependencies:

git clone https://github.com/Manohar-04/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt
Make sure to install librosa if it's not included in requirements.txt:


pip install librosa
Dataset
The dataset used in this project is the TESS Toronto emotional speech set. It contains audio recordings of various emotions, including anger, disgust, fear, happiness, neutral, pleasant surprise, and sadness.

Data Loading and Preprocessing
The audio files are loaded from the dataset directory, and the labels are extracted from the filenames. The data is then organized into a pandas DataFrame for further processing.

Visualization
The project includes functions to visualize the waveform and spectrogram of audio signals for different emotions:


def waveform(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
Feature Extraction
The Mel-frequency cepstral coefficients (MFCC) are extracted from the audio signals, which serve as features for the machine learning model:


def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
Model Training
A sequential LSTM model is built using Keras to classify the emotions based on the extracted MFCC features. The model architecture is as follows:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
The model is trained for 50 epochs with a batch size of 64.

Evaluation
The model's performance is evaluated using training and validation accuracy and loss metrics:


history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)

epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
Results
The plots of training and validation accuracy and loss over epochs are provided to illustrate the model's performance.

License
This project is licensed under the MIT License. See the LICENSE file for details.

dataset link:https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
