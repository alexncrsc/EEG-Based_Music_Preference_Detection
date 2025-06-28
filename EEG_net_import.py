import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from eegmodels.EEG_Net import EEGNet  #repo
#imports
import os
import numpy as np
import mne
import pywt
import random
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


#filter+denoise
def bandpass_filter(eeg_data, sfreq=250, l_freq=4, h_freq=40):
    filtered_data = []
    for trial in eeg_data:
        trial_T = trial.T
        trial_filtered = mne.filter.filter_data(data=trial_T, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)
        filtered_data.append(trial_filtered.T)
    return np.array(filtered_data)

def wavelet_denoise_one_channel(eeg_signal, wavelet='db4', level=2):
    coeffs = pywt.wavedec(eeg_signal, wavelet, level=level)
    thresholded = [coeffs[0]]
    for c in coeffs[1:]:
        t = np.median(np.abs(c)) / 0.6745
        thresholded.append(pywt.threshold(c, t, mode='soft'))
    return pywt.waverec(thresholded, wavelet)

def wavelet_denoise_trials(eeg_data, wavelet='db4', level=2):
    denoised_data = []
    for trial in eeg_data:
        trial_denoised = []
        for ch in range(trial.shape[1]):
            denoised_ch = wavelet_denoise_one_channel(trial[:, ch], wavelet, level)
            trial_denoised.append(denoised_ch[:trial.shape[0]])
        denoised_data.append(np.array(trial_denoised).T)
    return np.array(denoised_data)

#data load
def load_eeg_data(path, label):
    data_list, labels = [], []
    for f in os.listdir(path):
        if f.endswith(".npy"):
            data = np.load(os.path.join(path, f)).astype(np.float64)
            if data.shape[1] > skip_samples:
                data = data[:, skip_samples:]
            else:
                continue
            if data.shape[0] != 8:
                print(f"Skipping {f} due to incorrect shape: {data.shape}")
                continue
            data_list.append(data)
            labels.append(label)
    return data_list, labels


liked_path = "eeg_recordings/user_3/toate/liked/"
disliked_path = "eeg_recordings/user_3/toate/disliked/"
fs = 250
skip_secs = 2.5
skip_samples = int(skip_secs * fs)

liked, y1 = load_eeg_data(liked_path, 1)
disliked, y0 = load_eeg_data(disliked_path, 0)
X_raw = np.array(liked + disliked, dtype=object)
y = np.array(y1 + y0)

# preproces
X = np.stack([x.T.astype(np.float64) for x in X_raw])

X = bandpass_filter(X, sfreq=fs)
X = wavelet_denoise_trials(X)



X = np.transpose(X, (0, 2, 1))
X = X[..., np.newaxis]
#model config
model = EEGNet(nb_classes=1, Chans=8, Samples=3875, dropoutRate=0.5, kernLength=64,
               F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')

model.compile(loss=BinaryCrossentropy(), optimizer=Adam(1e-3), metrics=['accuracy'])

#cross_calidation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\n=== Fold {fold+1} ===")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train, epochs=60, batch_size=16, verbose=1)
    preds = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
