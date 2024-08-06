import librosa
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

model = joblib.load('model.joblib')

labelencoder = joblib.load('label_encoder.joblib')

def prediction(audio_input):
    audio, sample_rate = librosa.load(audio_input)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=np.argmax(model.predict(mfccs_scaled_features), axis=-1)
    prediction_class = labelencoder.inverse_transform(predicted_label) 
    return prediction_class[0]