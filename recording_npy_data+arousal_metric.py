import time
import os
import random
import numpy as np
from datetime import datetime

# ======== BrainFlow Imports ========
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes

# ======== Spotify Imports ========
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ==============================
#       BRAINFLOW SETUP
# ==============================
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = "/dev/cu.usbserial-DQ0081Z3"  
board = BoardShim(0, params)

sfreq = 250
duration = 20      
required_samples = 4500
HP_CUTOFF = 1.0
NOTCH_FREQ = 50.0
NOTCH_BW = 2.0
NOTCH_ORDER = 4

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
#        PLAYLISTS 
# ==============================
playlist_id_1 = "6PYMiBT7xZ99DAUrlvNgZk"
playlist_id_2 = "6ytqBorxj1RBjI5mwGwQo7"
track_uris_1 = [track["track"]["uri"] for track in sp.playlist_tracks(playlist_id_1)["items"]][:14]
track_uris_2 = [track["track"]["uri"] for track in sp.playlist_tracks(playlist_id_2)["items"]][:18]

use_playlist_1 = True

# ==============================
#   SAVE EEG DATA
# ==============================
def save_eeg_data(eeg_data, song_name, user_id, label="", arousal_level=""):
    """
    Saves the EEG data as a .npy file, including:
    [arousal_level]_[song_name]_[optional label]_[timestamp].npy
    """

    import os
    from datetime import datetime
    import numpy as np

    base_path = f"eeg_recordings/user_{user_id}"
    os.makedirs(base_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #filename format
    file_name_parts = []

    #arousal at the start of the filename
    if arousal_level:
        file_name_parts.append(arousal_level)

    #Song name second
    file_name_parts.append(song_name.replace(" ", "_"))

    # liked/dislike label
    if label:
        file_name_parts.append(label)

    # timestamp to avoid overwriting
    file_name_parts.append(timestamp)

    #undescrore
    file_name = "_".join(file_name_parts) + ".npy"
    file_path = os.path.join(base_path, file_name)

    np.save(file_path, eeg_data)
    print(f"[INFO] EEG data saved at: {file_path}")


# ============================
#   EEG ACQUISITION (INLINE)
# ==============================
def start_streaming(record_time):
    """
    Starts the EEG stream, records for `record_time` seconds, then stops.
    Applies filters and returns the combined EEG data array.
    """
    board.prepare_session()
    print("[INFO] Board session prepared.")

    # starting command brainflow for each channel
    for ch in range(1, 9):
        command_str = f"x{ch}060110X"
        board.config_board(command_str)
    time.sleep(0.5)

    # Start streaming
    board.start_stream(45000, "")
    print("[INFO] Started data streaming.")

    # Record for `record_time` seconds:
    time_start = time.time()
    total_data = []

    while (time.time() - time_start) < record_time:
        time.sleep(0.01)  # slight pause to allow data buffer to fill
        data = board.get_board_data()
        if data.shape[1] > 0:
            chunk = data[1:9, :]  # channels [1..8]
            total_data.append(chunk)

    # Stop and release the board
    board.stop_stream()
    board.release_session()
    print("[INFO] Stopped data streaming.")

    # Combine chunks into one array
    if len(total_data) > 0:
        combined_data = np.hstack(total_data)
    else:
        combined_data = np.zeros((8, 0))

   
    sample_count = combined_data.shape[1]
    print(f"[INFO] Total collected samples: {sample_count}")

   
    if sample_count >= required_samples:
        combined_data = combined_data[:, :required_samples]
    else:
        print("[WARNING] Not enough samples were collected to reach the required amount.")

   
    if combined_data.shape[1] > 0:
        for ch_idx in range(combined_data.shape[0]):
            # High-pass 1 Hz
            DataFilter.perform_highpass(
                combined_data[ch_idx, :], sfreq, HP_CUTOFF, 4, FilterTypes.BUTTERWORTH.value, 0
            )
            # Notch 50 Hz => 49-51 Hz
            low_notch = NOTCH_FREQ - (NOTCH_BW / 2)
            high_notch = NOTCH_FREQ + (NOTCH_BW / 2)
            DataFilter.perform_bandstop(
                combined_data[ch_idx, :], sfreq, low_notch, high_notch,
                NOTCH_ORDER, FilterTypes.BUTTERWORTH.value, 0
            )

    print(f"[INFO] Final EEG data shape: {combined_data.shape}")
    return combined_data

# ==============================
#   SPOTIFY PLAYBACK & EEG
# ==============================
def play_music_and_collect_eeg():
    global use_playlist_1
    chosen_uris = track_uris_1 if use_playlist_1 else track_uris_2
    track_uri = random.choice(chosen_uris)
    use_playlist_1 = not use_playlist_1  # alternate next time

    #retrieve track info
    track_info = sp.track(track_uri)
    track_duration = track_info['duration_ms']
    track_name = track_info['name']
    label = "liked" if chosen_uris is track_uris_1 else "disliked"

    #ranndom start position if the track is long enough
    max_start = max(0, track_duration - (duration + 15) * 1000)
    start_pos = random.randint(0, max_start) if max_start > 0 else 0

    #start Spotify playback
    sp.start_playback(uris=[track_uri])
    sp.seek_track(start_pos)
    print(f"[INFO] Playing '{track_name}' ({label.upper()}). Starting at {start_pos} ms.")



    
    record_seconds = duration 
    eeg_array = start_streaming(record_time=record_seconds)

    #pause the music
    sp.pause_playback()
    print("[INFO] Playback paused. Now collecting user arousal feedback...")

    #prompt for arousal level
    try:
        arousal_level = input("Enter your arousal level for this song (1-9): ")
    except:
        arousal_level = "NA"

    #15-second baseline/quiet period before next track
    print("[INFO] 15-second break before next track.")
    time.sleep(15)

    #check if we have enough data (optional)
    if eeg_array.shape[1] < required_samples:
        print("[WARNING] EEG data has fewer than the required samples. Saving anyway.")
    #save the data for this trial
    save_eeg_data(eeg_array, track_name, user_id=3, label=label, arousal_level=arousal_level)

    return eeg_array

# ==============================
#            MAIN
# ==============================
def main():
    num_trials = 8  
    for i in range(num_trials):
        print(f"--- TRIAL {i+1}/{num_trials} ---")
        result = play_music_and_collect_eeg()
        if result is None or result.shape[1] == 0:
            print("[WARNING] No valid EEG data for this trial.")

    print("[INFO] Recording session complete.")

if __name__ == "__main__":
    main()
