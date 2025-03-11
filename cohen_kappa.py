from sklearn.metrics import cohen_kappa_score, log_loss
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
import pywt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.feature_selection import RFE
from sklearn.utils import shuffle
from joblib import dump

# ==============================
#       PARAMETERS
# ==============================
fs = 250  # Sampling Frequency
skip_secs = 2.5
skip_samples = int(skip_secs * fs)
lowcut, highcut = 4.0, 30.0  # Bandpass range

# ==============================
#       LOAD DATA
# ==============================
liked_path = "/Users/alexnicorescu/eeg_recordings/user_3/toate/liked/"
disliked_path = "/Users/alexnicorescu/eeg_recordings/user_3/toate/disliked/"

def load_eeg_data(path, label):
    eeg_data_list, labels = [], []
    for file in os.listdir(path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(path, file))
            if data.shape[1] > skip_samples:
                data = data[:, skip_samples:]  # Remove first few seconds
            eeg_data_list.append(data)
            labels.append(label)
    return eeg_data_list, labels

liked_data, liked_labels = load_eeg_data(liked_path, 1)
disliked_data, disliked_labels = load_eeg_data(disliked_path, 0)
all_data = np.array(liked_data + disliked_data, dtype=object)
all_labels = np.array(liked_labels + disliked_labels)

# ==============================
#       SIGNAL PROCESSING
# ==============================

def wavelet_denoise(trial, wavelet='db4', level=4):
    n_channels, n_samples = trial.shape
    denoised_trial = np.zeros_like(trial)
    for ch in range(n_channels):
        coeffs = pywt.wavedec(trial[ch], wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(n_samples))
        coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
        denoised_trial[ch] = np.real(pywt.waverec(coeffs, wavelet)[:n_samples])
    return denoised_trial

def bandpass_filter(trial, lowcut=4, highcut=40, fs=250, order=4):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, trial, axis=1)

def compute_bandpower(trial, sfreq=250):
    freq_bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    return np.hstack([
        np.sum(welch(trial[ch], fs=sfreq, nperseg=256)[1][
            (welch(trial[ch], fs=sfreq, nperseg=256)[0] >= f_low) &
            (welch(trial[ch], fs=sfreq, nperseg=256)[0] <= f_high)])
        for f_low, f_high in freq_bands.values() for ch in range(trial.shape[0])
    ])

def extract_features(trial):
    trial = np.real(trial)
    return np.hstack(np.real([compute_bandpower(trial)]))

def run_pipeline_cv(all_data, all_labels, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_results = {}
    
    for train_idx, test_idx in skf.split(all_data, all_labels):
        X_train_raw, X_test_raw = all_data[train_idx], all_data[test_idx]
        y_train, y_test = all_labels[train_idx], all_labels[test_idx]

        X_train_features = [extract_features(wavelet_denoise(bandpass_filter(trial))) for trial in X_train_raw]
        X_test_features = [extract_features(wavelet_denoise(bandpass_filter(trial))) for trial in X_test_raw]
        
      
  
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_test_scaled = scaler.transform(X_test_features)
        dump(scaler, 'scaler.joblib')

        selector = RFE(SVC(kernel='linear'), n_features_to_select=10)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        # Apply PCA (keep 95% variance)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95)  # Retains enough components to explain 95% of variance
        X_train_pca = pca.fit_transform(X_train_features)
        X_test_pca = pca.transform(X_test_features)
        
        for model_name in ["svm", "random_forest", "knn", "xgboost", "mlp"]:
            model = choose_model(model_name)
            model.fit(X_train_pca, y_train)

            y_pred = model.predict(X_test_pca)
            train_acc = model.score(X_train_pca, y_train)
            test_acc = model.score(X_test_pca, y_test)
            overfit_gap = train_acc - test_acc
            kappa = cohen_kappa_score(y_test, y_pred)

            train_loss, test_loss = None, None
            if hasattr(model, "predict_proba"):  # if the model support the predict_proba
                y_train_prob = model.predict_proba(X_train_pca)
                y_test_prob = model.predict_proba(X_test_pca)
                train_loss = log_loss(y_train, y_train_prob)
                test_loss = log_loss(y_test, y_test_prob)

            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append({
                "train_acc": train_acc,
                "test_acc": test_acc,
                "overfit_gap": overfit_gap,
                "kappa": kappa,
                "train_loss": train_loss,
                "test_loss": test_loss
            })
    
    return {k: {
                "mean_train_acc": np.mean([r['train_acc'] for r in v]),
                "mean_test_acc": np.mean([r['test_acc'] for r in v]),
                "mean_overfit_gap": np.mean([r['overfit_gap'] for r in v]),
                "mean_kappa": np.mean([r['kappa'] for r in v]),
                "mean_train_loss": np.mean([r['train_loss'] for r in v if r['train_loss'] is not None]),
                "mean_test_loss": np.mean([r['test_loss'] for r in v if r['test_loss'] is not None])
            } for k, v in model_results.items()}

def choose_model(model_type):
    model_dict = {
        "svm": SVC(kernel='linear', C=0.1, probability=True),
        "random_forest": RandomForestClassifier(n_estimators=10, max_depth=4, min_samples_leaf=10, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "xgboost": xgb.XGBClassifier(n_estimators=10, max_depth=2, learning_rate=0.05, reg_lambda=2, eval_metric='logloss'),
        "mlp": MLPClassifier(hidden_layer_sizes=(75, 25), activation='relu', solver='adam', alpha=0.005, max_iter=5000, learning_rate_init=0.005, random_state=42, early_stopping=True)
    }
    return model_dict.get(model_type, SVC(kernel='rbf', probability=True))

if __name__ == "__main__":
    results = run_pipeline_cv(all_data, all_labels)
    for model, stats in results.items():
        print(f"{model.upper()} => Train Acc: {stats['mean_train_acc']:.2f}, Test Acc: {stats['mean_test_acc']:.2f}, Overfit Gap: {stats['mean_overfit_gap']:.2f}, Kappa: {stats['mean_kappa']:.2f}, Train Loss: {stats['mean_train_loss']:.4f}, Test Loss: {stats['mean_test_loss']:.4f}")
