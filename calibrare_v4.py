import threading
import time
import csv
import os
import random
import numpy as np
from datetime import datetime
import psycopg2

# ======== BrainFlow Imports ========
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

# ======== Spotify Imports ========
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ======== Scikit-learn Imports ========
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# ==============================
#        BRAINFLOW SETUP
# ==============================
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = 'COM3'  # Update to your port if needed
board = BoardShim(0, params)  # Use the correct integer ID for your board



# Sampling frequency and required samples
sfreq = 250
duration = 10  # seconds
required_samples = 1800
stop_event = threading.Event()

# ==============================
#       SPOTIFY API SETUP
# ==============================
SPOTIPY_CLIENT_ID = '8e3c71335cd64dcba2ca8aa88353bf83'
SPOTIPY_CLIENT_SECRET = 'e77ad5c78796468f84e8d8bc24d18203'
SPOTIPY_REDIRECT_URI = 'http://localhost:8080/callback'
scope = "user-read-playback-state user-modify-playback-state user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=scope,
    cache_path="spotipy_cache"
))

# ==============================
#        PLAYLISTS & TRACKS
# ==============================
playlist_id_1 = "5nlbo8RyvvclIgZEQcqTYH"
playlist_id_2 = "31tJVez6DcLdymPhEpIeQO"

track_uris_1 = [track["track"]["uri"] for track in sp.playlist_tracks(playlist_id_1)["items"]][:8]
track_uris_2 = [track["track"]["uri"] for track in sp.playlist_tracks(playlist_id_2)["items"]][:8]

use_playlist_1 = True

# ==============================
#  CSV LOG FILE (Optional Demo)
# ==============================
log_file = "user_feedback.csv"

if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_uri", "like_dislike"])

# ==============================
#      POSTGRES LOGGING
# ==============================
def log_session_to_db(user_id, model_type, accuracy, calibration_folder):
    """
    Logs session details to a PostgreSQL 'session_logs' table.
    
    Table schema example:
        CREATE TABLE session_logs (
            id SERIAL PRIMARY KEY,
            user_id INT NOT NULL,
            session_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_type VARCHAR(50),
            accuracy FLOAT,
            calibration_folder VARCHAR(255)
        );
    """
    try:
        conn = psycopg2.connect(
            dbname="my_experiment_db",
            user="postgres",
            password="parola",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        
        query = """
            INSERT INTO session_logs (user_id, model_type, accuracy, calibration_folder)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (user_id, model_type, accuracy, calibration_folder))
        conn.commit()
        print("Session logged successfully.")
    except Exception as e:
        print(f"Error logging session: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# ==============================
#   EEG ACQUISITION THREAD
# ==============================
def start_streaming(eeg_data_container):
    """
    Acquire EEG data from the board in a background thread.  
    Data is accumulated into eeg_data_container (a list),
    which is converted to a NumPy array once required_samples are collected.
    """
    stop_event.clear()
    board.prepare_session()
    print("Session prepared.")
    board.start_stream()
    print("Stream started.")

    total_data = []
    total_samples = 0

    while not stop_event.is_set():
        time.sleep(0.01)  # Poll data often
        data = board.get_board_data()

        if data.shape[1] > 0:
            total_data.append(data[:6])  # keep first 6 channels
            total_samples += data.shape[1]
            print(f"Accumulated samples: {total_samples}")

        if total_samples >= required_samples:
            stop_event.set()
            break

    # Combine all accumulated data
    if total_samples >= required_samples:
        combined_data = np.hstack(total_data)[:, :required_samples]
    else:
        combined_data = np.hstack(total_data) if total_samples > 0 else np.zeros((6, 0))

    eeg_data_container.append(combined_data)
    print(f"Final EEG data shape: {combined_data.shape}")

    board.stop_stream()
    board.release_session()
    print("Stream stopped and session released.")

# ==============================
#   SPOTIFY PLAYBACK & EEG
# ==============================
def play_music_and_collect_eeg():
    """
    Plays a random 10-second clip from one of the two playlists
    and collects the corresponding EEG data.
    """
    global use_playlist_1

    # Alternate or randomly choose between playlists
    track_uri = random.choice(track_uris_1 if use_playlist_1 else track_uris_2)
    use_playlist_1 = not use_playlist_1

    track_info = sp.track(track_uri)
    track_duration = track_info['duration_ms']
    track_name = track_info['name']

    # Random start time
    start_pos = random.randint(0, max(0, track_duration - duration * 1000))

    # Start playback
    sp.start_playback(uris=[track_uri])
    sp.seek_track(start_pos)

    eeg_data_container = []

    # Launch EEG streaming in a thread
    streaming_thread = threading.Thread(
        target=start_streaming,
        args=(eeg_data_container,)
    )
    streaming_thread.start()

    print(f"Playing track {track_name} for {duration} seconds at {start_pos} ms.")
    time.sleep(duration)

    stop_event.set()
    streaming_thread.join()

    # Pause after done
    sp.pause_playback()
    print("Playback paused.")

    if len(eeg_data_container) == 0:
        return None
    eeg_array = eeg_data_container[0]
    if eeg_array.shape[1] < required_samples:
        print("Insufficient EEG data collected.")
        return None
    return eeg_array, track_uri

# ==============================
#  REAL EEG FEATURE EXTRACTION
# ==============================
def extract_features_from_eeg(eeg_signal):
    """
    Perform bandpass filtering (0.5-50 Hz) + PSD (Welch) + band power extraction.
    """
    from brainflow.data_filter import DataFilter, FilterTypes
    print(eeg_signal.shape)

    n_channels, n_samples = eeg_signal.shape
    print(eeg_signal.shape)

    bandpass_low = 0.5
    bandpass_high = 50.0

    center_freq = (bandpass_high + bandpass_low) / 2.0  # e.g., 25.25
    band_width  = bandpass_high - bandpass_low          # e.g., 49.5 (NOT half)

    for ch in range(n_channels):
        DataFilter.perform_bandpass(
            eeg_signal[ch],
            sfreq,
            center_freq,                # 25.25
            band_width,                 # 49.5
            4,
            FilterTypes.BUTTERWORTH.value,
            0
        )

    # Frequency bands
    freq_bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
    }
    for ch in range(n_channels):
        channel_data = eeg_signal[ch]
        count_nans = np.isnan(channel_data).sum()
        count_infs = np.isinf(channel_data).sum()
        print(f"Channel {ch} -> NaNs: {count_nans}, Infs: {count_infs}")


    # Compute band powers
    nfft = 256
    overlap = 64 # samples overlap
    band_powers = []
    HAMMING_WINDOW = 1
    for ch in range(n_channels):
        psd_data = DataFilter.get_psd_welch(
    eeg_signal[ch],
    nfft,
    nfft // 2,
    overlap,
    HAMMING_WINDOW
)
        psd, freqs = psd_data[0], psd_data[1]

        # Sum band powers
        channel_features = []
        for (f_low, f_high) in freq_bands.values():
            idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            bp = np.sum(psd[idx])
            channel_features.append(bp)

        band_powers.append(channel_features)

    # Flatten to shape (6*4,) = (24,)
    return np.array(band_powers).flatten()

# ==============================
# SAVE & LOAD CALIBRATION DATA
# ==============================
def save_calibration_data(features, labels, user_id):
    """
    Saves or appends calibration data to user-specific folder:
      e.g., calibration_files/user_1/features.npy & labels.npy
    """
    base_path = f"calibration_files/user_{user_id}"
    os.makedirs(base_path, exist_ok=True)

    features_file = os.path.join(base_path, "features.npy")
    labels_file = os.path.join(base_path, "labels.npy")

    if os.path.exists(features_file) and os.path.exists(labels_file):
        existing_features = np.load(features_file)
        existing_labels = np.load(labels_file)
        features = np.vstack([existing_features, features])
        labels = np.hstack([existing_labels, labels])

    np.save(features_file, features)
    np.save(labels_file, labels)
    print(f"Calibration data saved in {base_path}")

def load_calibration_data(user_id):
    """
    Loads user-specific calibration data.
      e.g., calibration_files/user_1/features.npy & labels.npy
    """
    base_path = f"calibration_files/user_{user_id}"
    features_file = os.path.join(base_path, "features.npy")
    labels_file = os.path.join(base_path, "labels.npy")

    if os.path.exists(features_file) and os.path.exists(labels_file):
        features = np.load(features_file)
        labels = np.load(labels_file)
        return features, labels, base_path
    else:
        return None, None, base_path


#Choose model

def choose_model(model_type: str):
    """
    Return an initialized scikit-learn model based on model_type.
    """
    model_dict = {
        "svm":                 SVC(kernel='rbf', probability=True),
        "gradient_boosting":   GradientBoostingClassifier(),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "neural_net":          MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        "knn":                 KNeighborsClassifier(n_neighbors=5)
    }
    return model_dict.get(model_type, SVC(kernel='rbf', probability=True))  # default = SVM

#
# ==============================
#            MAIN
# ==============================
def main():
    # Ask user for ID and model type
    user_id = input("Enter user ID (e.g., 1): ").strip()
    if not user_id.isdigit():
        print("Invalid user ID, defaulting to 1.")
        user_id = "1"
    user_id = int(user_id)

    model_type = input(
        "Select a model type (svm/gradient_boosting/random_forest/neural_net/knn): "
    ).strip().lower()

    model = choose_model(model_type)
    print(f"Using model: {model}")


    # Data containers
    user_feedback = []
    corrected_labels = []

    # Start collecting EEG data + feedback
    for idx in range(6):  # Or however many trials you want
        eeg_data = play_music_and_collect_eeg()
        if eeg_data is None:
            continue
        eeg_signal, track_uri = eeg_data

        # Extract features
        features = extract_features_from_eeg(eeg_signal)

        # Prompt user
        feedback = input("Did you like this sequence? (yes/no/skip): ").strip().lower()
        if feedback == 'skip':
            print("Skipped this track; no data saved for this trial.")
            continue

        label = 1 if feedback == 'yes' else 0
        user_feedback.append(features)
        corrected_labels.append(label)

        # Also log to CSV (optional)
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                track_uri,
                feedback
            ])

    # After data collection, save calibration data
    if len(user_feedback) > 0:
        user_feedback = np.vstack(user_feedback)
        corrected_labels = np.array(corrected_labels)
        save_calibration_data(user_feedback, corrected_labels, user_id=user_id)
    else:
        print("No valid trials collected. Exiting.")
        return

    # Load all calibration data for this user
    all_features, all_labels, user_calib_folder = load_calibration_data(user_id)
    if all_features is None or all_labels is None:
        print("No calibration data available for training.")
        return

    # Shuffle data
    from sklearn.utils import shuffle
    all_features, all_labels = shuffle(all_features, all_labels, random_state=42)

    # Cross-validation
    scores = cross_val_score(model, all_features, all_labels, cv=3)
    accuracy = np.mean(scores)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Accuracy: {accuracy:.2f}")

    # Final training
    model.fit(all_features, all_labels)

    # Save the model if desired
    base_path = f"calibration_files/user_{user_id}"
    model_path = os.path.join(base_path, f"{model_type}_model.joblib")

    dump(model, model_path)
    print(f"Trained {model_type} model saved at: {model_path}")

    # Log session details to PostgreSQL
    log_session_to_db(
        user_id=user_id,
        model_type=model_type,
        accuracy=accuracy,
        calibration_folder=user_calib_folder
    )

    print("Done. Session completed successfully.")

if __name__ == "__main__":
    main()
