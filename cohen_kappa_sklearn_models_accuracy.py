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
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import entropy


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
                data = data[:, skip_samples:] 
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
def wavelet_denoise_one_channel(eeg_signal, wavelet='db4', level=2):

    coeffs = pywt.wavedec(eeg_signal, wavelet, level=level)
    thresholded_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            thresholded_coeffs.append(coeff)
        else:
            threshold = np.median(np.abs(coeff)) / 0.6745
            coeff_new = pywt.threshold(coeff, threshold, mode='soft')
            thresholded_coeffs.append(coeff_new)
    denoised_sig = pywt.waverec(thresholded_coeffs, wavelet)
    return np.real(denoised_sig).astype(np.float64) 


def wavelet_denoise(trial):

    n_channels, n_samples = trial.shape
    denoised_trial = np.zeros((n_channels, n_samples), dtype=np.float64)
    
    for ch in range(n_channels):
        denoised = wavelet_denoise_one_channel(trial[ch])
        denoised_trial[ch] = denoised[:n_samples]  # clip if too long
    
    return denoised_trial



def bandpass_filter(trial, lowcut=4, highcut=40, fs=250, order=4):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, trial, axis=1)



def zero_crossing_rate(signal_1d):
    zero_crosses = np.nonzero(np.diff(np.sign(signal_1d)))[0]
    return len(zero_crosses)

def compute_hjorth_parameters(signal_1d):
    diff_signal = np.diff(signal_1d)
    var_zero = np.var(signal_1d)
    var_diff = np.var(diff_signal)
    
    mobility = np.sqrt(var_diff / var_zero) if var_zero != 0 else 0
    diff2_signal = np.diff(diff_signal)
    var_diff2 = np.var(diff2_signal)
    
    complexity = 0
    if var_diff != 0:
        complexity = np.sqrt((var_diff2 / var_diff)) / mobility if mobility != 0 else 0
    return mobility, complexity

def extract_features(trial, fs=250):
    features = []

    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40)
    }

    for ch in trial:
        freqs, pxx = welch(ch, fs=fs, nperseg=fs)
        pxx = np.array([val.real for val in pxx], dtype=np.float64)
        pxx += 1e-12  # avoid log(0)
        total_power = np.sum(pxx)
        
        theta_power = np.sum(pxx[(freqs >= 4) & (freqs < 8)])
        alpha_power = np.sum(pxx[(freqs >= 8) & (freqs < 13)])
        beta_power  = np.sum(pxx[(freqs >= 13) & (freqs < 30)])
        gamma_power = np.sum(pxx[(freqs >= 30) & (freqs < 40)])

        features.append(theta_power)
        features.append(alpha_power)
        features.append(beta_power)
        features.append(gamma_power)
        
        # Shannon entropy
        pxx_norm = pxx / (total_power + 1e-8)
        shannon_ent = entropy(pxx_norm)
        features.append(shannon_ent)

        # Hjorth
        mobility, complexity = compute_hjorth_parameters(ch)
        features.append(mobility)
        features.append(complexity)

        # Band powers
       
        features.extend([
            theta_power/total_power,
            alpha_power/total_power,
            beta_power/total_power,
            gamma_power/total_power,
        ])



        features.append(theta_power / (alpha_power + 1e-8))
        features.append(beta_power / (alpha_power + 1e-8))
        features.append(beta_power / (theta_power + 1e-8))

    return np.array(features)



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

        selector = RFE(SVC(kernel='linear'), n_features_to_select=112)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95)  
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
            f1 = f1_score(y_test, y_pred)  
            conf_matrix = confusion_matrix(y_test, y_pred) 

            train_loss, test_loss = None, None
            if hasattr(model, "predict_proba"):  
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
                "test_loss": test_loss,
                "f1": f1,  
                "conf_matrix": conf_matrix.tolist()  
            })
    
    return {k: {
                "mean_train_acc": np.mean([r['train_acc'] for r in v]),
                "mean_test_acc": np.mean([r['test_acc'] for r in v]),
                "mean_overfit_gap": np.mean([r['overfit_gap'] for r in v]),
                "mean_kappa": np.mean([r['kappa'] for r in v]),
                "mean_train_loss": np.mean([r['train_loss'] for r in v if r['train_loss'] is not None]),
                "mean_test_loss": np.mean([r['test_loss'] for r in v if r['test_loss'] is not None]),
                "mean_f1": np.mean([r['f1'] for r in v]),
                "last_conf_matrix": v[-1]['conf_matrix']

            } for k, v in model_results.items()}

def choose_model(model_type):
    model_dict = {
        "svm": SVC(kernel='linear', C=1, probability=True),
        "random_forest": RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_leaf=3, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "xgboost": xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, reg_lambda=1, eval_metric='logloss'),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.0005, max_iter=10000, learning_rate_init=0.001, random_state=42, early_stopping=True, n_iter_no_change=25)
    }
    return model_dict.get(model_type, SVC(kernel='rbf', probability=True))

if __name__ == "__main__":
    results = run_pipeline_cv(all_data, all_labels)
    for model, stats in results.items():
        print(f"{model.upper()} => "
      f"Train Acc: {stats['mean_train_acc']:.2f}, "
      f"Test Acc: {stats['mean_test_acc']:.2f}, "
      f"Overfit Gap: {stats['mean_overfit_gap']:.2f}, "
      f"Kappa: {stats['mean_kappa']:.2f}, "
      f"Train Loss: {stats['mean_train_loss']:.4f}, "
      f"Test Loss: {stats['mean_test_loss']:.4f}, "
      f"F1 Score: {stats['mean_f1']:.2f}, "
      f"Confusion Matrix:\n{stats['last_conf_matrix']}")



