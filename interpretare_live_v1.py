import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# BrainFlow imports
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

# Model loading
from joblib import load


##############################################################################
# HELPER FUNCTION: Feature Extraction
##############################################################################
def extract_features_for_live(eeg_signal, sfreq=250):
    """
    Perform bandpass 0.5-50 Hz, then Welch PSD, summing power in delta/theta/alpha/beta.
    Returns a (24,) feature vector if you have 6 channels and 4 frequency bands.
    """
    n_channels, n_samples = eeg_signal.shape
    if n_samples < 1:
        return None

    # 1) Bandpass 0.5 - 50 Hz
    bandpass_low = 0.5
    bandpass_high = 50.0
    center_freq = (bandpass_high + bandpass_low) / 2.0
    band_width = bandpass_high - bandpass_low

    for ch in range(n_channels):
        DataFilter.perform_bandpass(
            eeg_signal[ch],
            sfreq,
            center_freq,
            band_width,
            4,
            FilterTypes.BUTTERWORTH.value,
            0
        )

    # 2) Welch PSD
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30)
    }

    nfft = 256
    window_size = nfft // 2   # 128
    overlap = window_size // 2  # 64
    HAMMING_WINDOW = 1

    band_powers = []
    for ch in range(n_channels):
        psd_data = DataFilter.get_psd_welch(
            eeg_signal[ch],
            nfft,
            window_size,
            overlap,
            HAMMING_WINDOW
        )
        psd, freqs = psd_data[0], psd_data[1]

        # Sum power in each band
        channel_features = []
        for (f_low, f_high) in freq_bands.values():
            idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            bp = np.sum(psd[idx])
            channel_features.append(bp)
        band_powers.append(channel_features)

    return np.array(band_powers).flatten()  # shape: (6*4,) = (24,)


def load_user_model(user_id, model_type):
    """
    Loads a user-specific model from 'calibration_files/user_{user_id}/{model_type}_model.joblib'.
    Raises FileNotFoundError if not found.
    """
    base_path = f"calibration_files/user_{user_id}"
    model_path = os.path.join(base_path, f"{model_type}_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = load(model_path)
    return model


##############################################################################
# APPROACH 1: ACCUMULATE 1800 SAMPLES (~7.2 SECONDS)
##############################################################################
def run_accumulate_1800(model, sfreq=250, n_channels=6):
    """
    Continuously gather EEG data, accumulate until we have 1800 samples,
    then classify with the loaded 'model'.
    """
    REQUIRED_SAMPLES = 1800
    board = setup_board()

    # Start streaming
    board.prepare_session()
    board.start_stream()
    print("EEG stream started (accumulate_1800 mode)...")

    # Buffer to accumulate data
    accumulated_data = np.zeros((n_channels, 0))  # shape: (6, 0)

    # Setup matplotlib for live visualization
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live Classification (1800-sample blocks)")
    ax.set_ylim([0, 1])
    bar_container = ax.bar(["Dislike (0)", "Like (1)"], [0, 0], color=["r", "g"])
    plt.draw()
    plt.pause(0.001)

    try:
        while True:
            # Grab new data every 0.5 seconds
            time.sleep(0.5)
            new_data = board.get_board_data()
            if new_data.shape[1] == 0:
                continue

            # Take the first n_channels, accumulate horizontally
            new_data_eeg = new_data[:n_channels, :]
            accumulated_data = np.hstack([accumulated_data, new_data_eeg])
            print(f"Buffer size: {accumulated_data.shape[1]}")

            # Once we have 1800 samples or more, extract a chunk
            if accumulated_data.shape[1] >= REQUIRED_SAMPLES:
                chunk = accumulated_data[:, :REQUIRED_SAMPLES]
                # Remove those samples from buffer
                accumulated_data = accumulated_data[:, REQUIRED_SAMPLES:]

                # Extract features
                features = extract_features_for_live(chunk, sfreq=sfreq)
                if features is None:
                    continue

                # Classify
                update_plot_with_prediction(model, features, bar_container)
                plt.draw()
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        board.stop_stream()
        board.release_session()
        plt.ioff()
        plt.show()
        print("Session released (accumulate_1800 mode).")


##############################################################################
# APPROACH 2: SHORT CHUNKS (1 SECOND = 250 SAMPLES)
##############################################################################
def run_short_chunks(model, sfreq=250, n_channels=6):
    """
    Every 1 second, read the latest EEG data. If there's >= 250 samples, classify.
    """
    board = setup_board()

    # Start streaming
    board.prepare_session()
    board.start_stream()
    print("EEG stream started (short_chunk mode)...")

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live Classification (Short 1s Chunks)")
    ax.set_ylim([0, 1])
    bar_container = ax.bar(["Dislike (0)", "Like (1)"], [0, 0], color=["r", "g"])
    plt.draw()
    plt.pause(0.001)

    try:
        while True:
            time.sleep(1.0)  # poll once every second
            data = board.get_board_data()
            if data.shape[1] < 10:
                # Very few samples arrived
                continue

            # Take first 6 channels
            chunk_data = data[:n_channels, :]

            # If we have >= 250 new samples, take last 250
            if chunk_data.shape[1] >= 250:
                chunk = chunk_data[:, -250:]
            else:
                chunk = chunk_data

            features = extract_features_for_live(chunk, sfreq=sfreq)
            if features is None:
                continue

            # Classify
            update_plot_with_prediction(model, features, bar_container)
            plt.draw()
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        board.stop_stream()
        board.release_session()
        plt.ioff()
        plt.show()
        print("Session released (short_chunk mode).")


##############################################################################
# UTILITY FUNCTIONS
##############################################################################
def setup_board():
    """
    Common function to create a BrainFlow BoardShim with typical parameters.
    Change board_id or serial_port as needed.
    """
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = 'COM3'  # adjust for your environment
    board_id = 0  # e.g., 0 for OpenBCI Cyton
    board = BoardShim(board_id, params)
    return board

def update_plot_with_prediction(model, features, bar_container):
    """
    Update the bar chart based on model output (predict_proba or predict).
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([features])[0]  # e.g., array of shape (2,)
        bar_container[0].set_height(probs[0])  # Dislike
        bar_container[1].set_height(probs[1])  # Like
    else:
        pred_label = model.predict([features])[0]
        if pred_label == 0:
            bar_container[0].set_height(1.0)
            bar_container[1].set_height(0.0)
        else:
            bar_container[0].set_height(0.0)
            bar_container[1].set_height(1.0)


##############################################################################
# MAIN: Choose which approach to run
##############################################################################
def main():
    # Ask user for ID and model type
    user_id = input("Enter user ID (e.g. 1): ").strip()
    if not user_id.isdigit():
        print("Invalid user ID, defaulting to 1.")
        user_id = "1"
    user_id = int(user_id)

    model_type = input("Select a model type (svm/gradient_boosting/random_forest/neural_net/knn): ").strip().lower()

    # Try to load the user's trained model
    try:
        model = load_user_model(user_id, model_type)
    except FileNotFoundError as e:
        print(str(e))
        print("Exiting.")
        sys.exit(1)

    # Prompt for approach
    approach = input("Choose approach: 'accumulate_1800' or 'short_chunk': ").strip().lower()
    if approach == "accumulate_1800":
        run_accumulate_1800(model, sfreq=250, n_channels=6)
    elif approach == "short_chunk":
        run_short_chunks(model, sfreq=250, n_channels=6)
    else:
        print("Invalid approach. Defaulting to short_chunk.")
        run_short_chunks(model, sfreq=250, n_channels=6)

if __name__ == "__main__":
    main()
