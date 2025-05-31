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
from tensorflow.keras.activations import gelu
from sklearn.model_selection import train_test_split





SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


os.environ["TF_DETERMINISTIC_OPS"] = "0"
os.environ["PYTHONHASHSEED"] = str(SEED)





###############################################################################
#                             1. PARAMETERS & PATHS
###############################################################################
fs = 250           
skip_secs = 2.5
skip_samples = int(skip_secs * fs)

liked_path = "eeg_recordings/user_3/toate/liked/"
disliked_path = "eeg_recordings/user_3/toate/disliked/"

###############################################################################
#               2. BANDPASS FILTER & WAVELET DENOISING
###############################################################################
def bandpass_filter(eeg_data, sfreq=250, l_freq=4, h_freq=40):
    filtered_data = []
    for trial in eeg_data:
        trial_T = trial.T  
        trial_filtered = mne.filter.filter_data(
            data=trial_T, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
        )
        filtered_data.append(trial_filtered.T)  
    return np.array(filtered_data)

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
    return denoised_sig

def wavelet_denoise_trials(eeg_data, wavelet='db4', level=2):
    """
    eeg_data shape = (num_trials, num_samples, num_channels).
    per channel
    """
    denoised_data = []
    for trial in eeg_data:  # (num_samples, num_channels)
        trial_denoised = []
        for ch_idx in range(trial.shape[1]):
            channel_data = trial[:, ch_idx]
            channel_denoised = wavelet_denoise_one_channel(
                channel_data, wavelet=wavelet, level=level
            )
            
            channel_denoised = channel_denoised[: trial.shape[0]]
            trial_denoised.append(channel_denoised)
        trial_denoised = np.array(trial_denoised).T  # (num_samples, num_channels)
        denoised_data.append(trial_denoised)
    return np.array(denoised_data)

###############################################################################
#               3. LOADING RAW EEG .NPY FILES
###############################################################################
def load_eeg_data(path, label):

    eeg_data_list, labels = [], []
    for file in os.listdir(path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(path, file))
            data = data.astype(np.float64)
            
            # skip
            if data.shape[1] > skip_samples:
                data = data[:, skip_samples:]
            else:
                continue
            
 
            if data.shape[0] != 8:
                print(f"Skipping {file} due to incorrect shape: {data.shape}")
                continue

            eeg_data_list.append(data)
            labels.append(label)
    
    return eeg_data_list, labels

###############################################################################
#               4. FEATURE EXTRACTION
###############################################################################
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

def zero_crossing_rate(signal_1d):

    zero_crosses = np.nonzero(np.diff(np.sign(signal_1d)))[0]
    return len(zero_crosses)

def compute_features_for_window(window_2d, fs=250):

    num_channels = window_2d.shape[1]
    all_feats = []
    
    for ch in range(num_channels):
        ch_data = window_2d[:, ch]
        #mean_val = np.mean(ch_data)
       # std_val  = np.std(ch_data)
       # zcr_val = zero_crossing_rate(ch_data)
        
  
        mob, comp = compute_hjorth_parameters(ch_data)
        
        # Welch PSD
        freqs, pxx = welch(ch_data, fs=fs, nperseg=fs)
        theta_power = np.sum(pxx[(freqs >= 4) & (freqs < 8)])
        alpha_power = np.sum(pxx[(freqs >= 8) & (freqs < 13)])
        beta_power  = np.sum(pxx[(freqs >= 13) & (freqs < 30)])
        gamma_power = np.sum(pxx[(freqs >= 30) & (freqs < 40)])

        #rel
        total_power = np.sum(pxx[(freqs >= 1) & (freqs < 40)]) + 1e-8  

        theta_rel = theta_power / total_power
        alpha_rel = alpha_power / total_power
        beta_rel  = beta_power  / total_power
        gamma_rel = gamma_power / total_power
        #ratio
        theta_alpha_ratio = theta_power / (alpha_power + 1e-8)
        beta_alpha_ratio  = beta_power  / (alpha_power + 1e-8)
        beta_theta_ratio  = beta_power  / (theta_power + 1e-8)

        # Entropy
        pxx_norm = pxx / (np.sum(pxx) + 1e-8)
        shannon_ent = entropy(pxx_norm)
        
     
        feats_ch = [
           # mean_val,
           #std_val,
           # zcr_val,

            mob,
            comp,
            theta_power,
            alpha_power,
            beta_power, 
            gamma_power,
            theta_rel,
            alpha_rel,
            beta_rel,
            gamma_rel,
            theta_alpha_ratio,
            beta_alpha_ratio,
            beta_theta_ratio,
            shannon_ent
        ]
        all_feats.extend(feats_ch)
    
    return np.array(all_feats)

def segment_and_extract_features(eeg_data, fs=250, window_size=3, step_size=1):
    """
    window splitting
    """
    num_trials, num_samples, num_channels = eeg_data.shape
    window_samples = int(window_size * fs)
    step_samples   = int(step_size * fs)
    
    all_trials_feature_sequences = []
    
    for i in range(num_trials):
        trial = eeg_data[i]  # (num_samples, num_channels)
        features_seq = []
        start = 0
        while start + window_samples <= num_samples:
            window_2d = trial[start : start + window_samples, :] 
            feats = compute_features_for_window(window_2d, fs=fs)
            features_seq.append(feats)
            start += step_samples
        
        features_seq = np.array(features_seq)  # shape = (num_windows, feat_dim)
        all_trials_feature_sequences.append(features_seq)
    
    return all_trials_feature_sequences

def pad_sequences_to_fixed(sequences, max_len, feat_dim):
    return np.stack(sequences)


from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

###############################################################################
#               5. BUILDING THE TRANSFORMER MODEL
###############################################################################
def transformer_block(inputs, num_heads=4, ff_dim=64, dropout_rate=0.5):
   
    x_norm1 = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=16)(x_norm1, x_norm1)
    attention = layers.Dropout(dropout_rate)(attention)
    x_res1 = inputs + attention 

    
    x_norm2 = layers.LayerNormalization(epsilon=1e-6)(x_res1)
    ffn = layers.Dense(ff_dim, activation="gelu")(x_norm2)
    ffn = layers.Dense(inputs.shape[-1])(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    x_out = x_res1 + ffn  

    return x_out





def create_transformer_model(max_windows, feat_dim, num_heads=4, ff_dim=64, num_transformer_blocks=3):

   
    inputs = layers.Input(shape=(max_windows, feat_dim))
    
    x = layers.LayerNormalization()(inputs)

    x = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    
    x = layers.Dropout(0.5)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)

   
    x = layers.GlobalAveragePooling1D()(x)

    
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  

    model = models.Model(inputs, outputs)

    
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
###############################################################################
#               6. MAIN
###############################################################################
if __name__ == "__main__":
    liked_data, liked_labels = load_eeg_data(liked_path, label=1)
    disliked_data, disliked_labels = load_eeg_data(disliked_path, label=0)

    all_data = np.array(liked_data + disliked_data, dtype=object)  
    all_labels = np.array(liked_labels + disliked_labels)

    trials_list = []
    for arr in all_data:
        arr_T = arr.T 
        trials_list.append(arr_T)
    
    # (num_trials, num_samples, 8)
    trials_array = np.stack(trials_list, axis=0).astype(np.float64)
    print(f"Raw EEG data shape: {trials_array.shape}")

   
    trials_array = bandpass_filter(trials_array, sfreq=fs, l_freq=4, h_freq=40)
    trials_array = wavelet_denoise_trials(trials_array, wavelet='db4', level=2)
    print("After filtering & denoising:", trials_array.shape)

    
    all_trials_features_seq = segment_and_extract_features(
        eeg_data=trials_array, fs=fs, window_size=3, step_size=1
    )
   
    
    max_windows = max(seq.shape[0] for seq in all_trials_features_seq)
    feat_dim = all_trials_features_seq[0].shape[1] 
    all_features = pad_sequences_to_fixed(all_trials_features_seq, max_windows, feat_dim)

    X = all_trials_features_seq
    y = all_labels

   
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies_test = []
    fold_accuracies_train = []

    fold_index = 1
    for train_index, test_index in kf.split(trials_array, y):
        X_train_raw = trials_array[train_index]
        X_test_raw  = trials_array[test_index]
        y_train = y[train_index]
        y_test  = y[test_index]

        
        train_features_seq = segment_and_extract_features(
            eeg_data=X_train_raw, fs=fs, window_size=3, step_size=1
        )
        test_features_seq  = segment_and_extract_features(
            eeg_data=X_test_raw, fs=fs, window_size=3, step_size=1
        )


        max_windows = max(seq.shape[0] for seq in train_features_seq)
        feat_dim = train_features_seq[0].shape[1]

        X_train = pad_sequences_to_fixed(train_features_seq, max_windows, feat_dim)
        X_test  = pad_sequences_to_fixed(test_features_seq, max_windows, feat_dim)

      
        mean_val = np.mean(X_train)
        std_val  = np.std(X_train)
        X_train_norm = (X_train - mean_val) / (std_val + 1e-8)
        X_test_norm  = (X_test - mean_val) / (std_val + 1e-8)
        print(max_windows)
        print(feat_dim)
        
        model = create_transformer_model(max_windows, feat_dim) 
        

        #model.summary()
        print(X_train_norm.shape)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',              
            patience=3,                    
            restore_best_weights=True,
            verbose=1                        
        )
        

        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_norm, y_train, test_size=0.5, random_state=42, stratify=y_train)

       
        history = model.fit(
            X_train_sub, y_train_sub,
            epochs=100,
            batch_size=16,
            validation_data=(X_val_sub, y_val_sub),
            callbacks=[early_stopping],
            verbose=1
        )
        
        
        loss_test, accuracy_test = model.evaluate(X_test_norm, y_test, verbose=0)
        loss_train, accuracy_train = model.evaluate(X_train_sub, y_train_sub, verbose=0)

        
        
        print(f"Fold Test Accuracy = {accuracy_test:.4f}")
        print(f"Fold Train Accuracy  = {accuracy_train:.4f}")
        print(f"Fold Test loss  = {loss_test:.4f}")
        fold_accuracies_test.append(accuracy_test)
        fold_accuracies_train.append(accuracy_train)
    
    
    mean_acc_test = np.mean(fold_accuracies_test)
    std_acc_test = np.std(fold_accuracies_test)
    
    mean_acc_train = np.mean(fold_accuracies_train)
    std_acc_train = np.std(fold_accuracies_train)
    
    print("\n- R 5 folds CVs")
    print(f"Accuracies per fold: {fold_accuracies_test}")
    print(f"Accuracies per fold: {fold_accuracies_train}")
    print(f"Mean Accuracy_test: {mean_acc_test:.4f} ± {std_acc_test:.4f}")
    print(f"Mean Accuracy_train: {mean_acc_train:.4f} ± {std_acc_train:.4f}")