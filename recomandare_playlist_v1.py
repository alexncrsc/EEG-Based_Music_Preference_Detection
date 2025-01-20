import os
import sys
import time
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

# ======== Model & Utils Imports ========
from joblib import load

# ======== Visualization Imports ========
import matplotlib.pyplot as plt


##############################################################################
#                              CONFIG & CONSTANTS
##############################################################################
# Replace these with your actual Spotify app credentials
SPOTIPY_CLIENT_ID = "YOUR_SPOTIPY_CLIENT_ID"
SPOTIPY_CLIENT_SECRET = "YOUR_SPOTIPY_CLIENT_SECRET"
SPOTIPY_REDIRECT_URI = "http://localhost:8080/callback"
SPOTIFY_SCOPE = "user-read-playback-state user-modify-playback-state playlist-modify-public"

# BrainFlow settings
BOARD_ID = 0              # 0 for OpenBCI Cyton, adjust based on your board
SERIAL_PORT = "COM3"      # e.g., "COM3" on Windows or "/dev/ttyUSB0" on Linux
SFREQ = 250               # Sampling rate (Hz), ensure this matches your EEG device
DURATION = 10             # Duration to collect EEG data per track (seconds)
N_CHANNELS = 6            # Number of EEG channels to use
REQUIRED_SAMPLES = 1800   # Number of samples to collect (e.g., 7.2 seconds at 250 Hz)

# Recommendations
RECOMMENDATIONS_LIMIT = 5  # Number of recommended tracks per liked track


##############################################################################
#                      EEG FEATURE EXTRACTION FUNCTION
##############################################################################
def extract_features_from_eeg(eeg_signal, sfreq=250):
    """
    Perform bandpass 0.5-50 Hz, then compute Welch PSD, summing power in delta/theta/alpha/beta.
    Expects eeg_signal shape: (n_channels, n_samples).
    Returns a feature vector of shape (n_channels * 4,) = (24,) if 6 channels and 4 bands.
    """
    n_channels, n_samples = eeg_signal.shape
    if n_samples < 1:
        return None

    # 1) Bandpass filter: 0.5 - 50.0 Hz
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
            4,  # Order
            FilterTypes.BUTTERWORTH.value,
            0
        )

    # 2) Define frequency bands
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30)
    }

    # 3) Welch PSD
    nfft = 256
    window_size = nfft // 2   # 128
    overlap = window_size // 2  # 64
    HAMMING_WINDOW = 1         # Hamming window

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

        channel_features = []
        for (f_low, f_high) in freq_bands.values():
            idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            bp = np.sum(psd[idx])
            channel_features.append(bp)
        band_powers.append(channel_features)

    return np.array(band_powers).flatten()  # shape: (6*4,) = (24,)


##############################################################################
#                           MODEL LOADING
##############################################################################
def load_user_model(user_id, model_type):
    """
    Loads the user's trained model from: calibration_files/user_{id}/{model_type}_model.joblib
    """
    base_path = f"calibration_files/user_{user_id}"
    model_path = os.path.join(base_path, f"{model_type}_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[INFO] Loading model from: {model_path}")
    model = load(model_path)
    return model


##############################################################################
#                        SPOTIFY HELPER FUNCTIONS
##############################################################################
def get_playlist_id(playlist_url):
    """
    Extracts the playlist ID from a Spotify playlist URL or URI.
    """
    if "playlist/" in playlist_url:
        # e.g., https://open.spotify.com/playlist/<id>?si=...
        try:
            playlist_id = playlist_url.split("playlist/")[1].split("?")[0]
            return playlist_id
        except IndexError:
            raise ValueError("Invalid playlist URL format.")
    else:
        # Assume it's a plain playlist ID or URI
        return playlist_url


def fetch_all_tracks(sp, playlist_id):
    """
    Fetches all track URIs from a Spotify playlist, handling pagination.
    """
    track_uris = []
    results = sp.playlist_tracks(playlist_id, limit=100, offset=0)
    tracks = results['items']
    track_uris.extend([item['track']['uri'] for item in tracks if item['track'] is not None])

    while results['next']:
        results = sp.next(results)
        tracks = results['items']
        track_uris.extend([item['track']['uri'] for item in tracks if item['track'] is not None])

    return track_uris


def create_new_playlist(sp, user_spotify_id, user_id, model_type):
    """
    Creates a new Spotify playlist for the user to store liked tracks.
    """
    new_playlist_name = f"EEG_Liked_Songs_User_{user_id}_{model_type}"
    new_playlist = sp.user_playlist_create(
        user=user_spotify_id,
        name=new_playlist_name,
        public=True,  # Set to False for private playlists
        description="Playlist of songs the EEG model predicted I like"
    )
    new_playlist_id = new_playlist['id']
    print(f"[INFO] Created new playlist: {new_playlist_name}")
    return new_playlist_id


def add_tracks_to_playlist(sp, playlist_id, track_uris):
    """
    Adds a list of track URIs to a Spotify playlist in batches of 100.
    """
    if not track_uris:
        return
    for i in range(0, len(track_uris), 100):
        batch = track_uris[i:i+100]
        try:
            sp.playlist_add_items(playlist_id, batch)
            print(f"[INFO] Added {len(batch)} tracks to the playlist.")
        except Exception as e:
            print(f"[ERROR] Could not add tracks: {e}")


def fetch_recommended_tracks(sp, seed_tracks=None, seed_artists=None, seed_genres=None, limit=RECOMMENDATIONS_LIMIT, market='US'):
    """
    Fetches recommended tracks based on seed tracks, artists, and genres using Spotify's Recommendations API.
    """
    try:
        # Ensure seed parameters are lists or None
        params = {}
        if seed_tracks:
            params['seed_tracks'] = seed_tracks
        if seed_artists:
            params['seed_artists'] = seed_artists
        if seed_genres:
            params['seed_genres'] = seed_genres

        # Ensure at least one seed is provided
        if not any([seed_tracks, seed_artists, seed_genres]):
            print("[ERROR] At least one seed parameter must be provided.")
            return []

        recs = sp.recommendations(limit=limit, country=market, **params)
        recommended_uris = [r['uri'] for r in recs['tracks']]
        return recommended_uris
    except spotipy.exceptions.SpotifyException as e:
        print(f"[ERROR] Spotify API error: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Recommendations API failed: {e}")
        return []


##############################################################################
#                           TRACK VALIDATION FUNCTION
##############################################################################
def is_track_available(sp, track_id, market='US'):
    """
    Checks if a track is available in the specified market.
    """
    try:
        track = sp.track(track_id)
        available_markets = track['available_markets']
        if market in available_markets:
            return True
        else:
            print(f"[WARNING] Track ID {track_id} is not available in {market}.")
            return False
    except spotipy.exceptions.SpotifyException as e:
        print(f"[ERROR] Spotify API error while fetching track: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error while fetching track: {e}")
        return False


##############################################################################
#                           MAIN SCRIPT
##############################################################################
def main():
    # 1) Prompt for user ID & model type
    user_id = input("Enter user ID (e.g. 1): ").strip()
    if not user_id.isdigit():
        print("Invalid user ID, defaulting to 1.")
        user_id = "1"
    user_id = int(user_id)

    model_type = input("Select a model type (svm/gradient_boosting/random_forest/neural_net/knn): ").strip().lower()
    try:
        model = load_user_model(user_id, model_type)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    # 2) Setup Spotify
    print("[INFO] Authenticating with Spotify...")
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id="8e3c71335cd64dcba2ca8aa88353bf83",
        client_secret="e77ad5c78796468f84e8d8bc24d18203",
        redirect_uri="http://localhost:8080/callback",
        scope=(
                "user-read-playback-state "
                "user-modify-playback-state "
                "playlist-modify-public "
                "playlist-modify-private "
                "user-read-currently-playing"
            ),
        cache_path="spotipy_cache_new"  # Change to a new cache path to force re-authentication if needed
    ))
    current_user = sp.current_user()
    if current_user is None:
        print("Could not authenticate with Spotify. Check your credentials/scopes.")
        sys.exit(1)
    user_spotify_id = current_user["id"]
    print(f"[INFO] Logged into Spotify as user: {user_spotify_id}")

    # 3) Ask for a sample Spotify playlist
    playlist_url = input("Enter a Spotify playlist URL or ID to classify tracks from: ").strip()
    try:
        playlist_id = get_playlist_id(playlist_url)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    # 4) Fetch all tracks from the playlist
    print(f"[INFO] Fetching tracks from playlist: {playlist_id}")
    track_uris = fetch_all_tracks(sp, playlist_id)
    print(f"[INFO] Found {len(track_uris)} tracks in the playlist.")

    # 5) Limit the number of tracks to process
    max_tracks_input = input("Enter the maximum number of tracks to process (or press Enter for all): ").strip()
    if max_tracks_input.isdigit():
        max_tracks = int(max_tracks_input)
        track_uris = track_uris[:max_tracks]
        print(f"[INFO] Processing the first {max_tracks} tracks.")
    else:
        print("[INFO] Processing all tracks in the playlist.")

    # 6) Create a new playlist for "liked" songs
    new_playlist_id = create_new_playlist(sp, user_spotify_id, user_id, model_type)

    # 7) Setup BrainFlow board for EEG
    print("[INFO] Setting up EEG board...")
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    try:
        board.prepare_session()
        board.start_stream()
        print("[INFO] EEG stream started.")
    except Exception as e:
        print(f"[ERROR] Failed to start EEG stream: {e}")
        sys.exit(1)

    # 8) Visual: simple bar chart for like vs. dislike
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live EEG Classification (Like vs Dislike)")
    ax.set_ylim([0, 1])
    bar_container = ax.bar(["Dislike (0)", "Like (1)"], [0, 0], color=["r", "g"])
    plt.draw()
    plt.pause(0.001)

    liked_tracks = []

    try:
        # Iterate over each track in the playlist
        for idx, t_uri in enumerate(track_uris):
            track_info = sp.track(t_uri)
            track_name = track_info.get("name", "Unknown")
            artists = [a["name"] for a in track_info.get("artists", [])]
            track_id = track_info['id']  # Extract the track ID

            print(f"\n[INFO] Classifying track {idx+1}/{len(track_uris)}: '{track_name}' by {', '.join(artists)}")

            # 8a) Start playback
            track_duration_ms = track_info["duration_ms"]
            if track_duration_ms < DURATION * 1000:
                play_duration = track_duration_ms // 1000
            else:
                play_duration = DURATION
            start_pos = random.randint(0, max(0, track_duration_ms - play_duration * 1000))
            try:
                sp.start_playback(uris=[t_uri], position_ms=start_pos)
                print(f"   > Playing snippet for {play_duration} seconds at offset {start_pos} ms...")
            except Exception as e:
                print(f"   > [ERROR] Could not start playback: {e}")
                continue

            # 8b) Acquire EEG for play_duration seconds
            snippet_data_container = []
            snippet_total_samples = 0
            start_time = time.time()
            while (time.time() - start_time) < play_duration:
                time.sleep(0.05)  # poll EEG often
                try:
                    data = board.get_board_data()  # shape: (all_ch, new_samples)
                except Exception as e:
                    print(f"   > [ERROR] Error fetching EEG data: {e}")
                    data = np.array([[]])

                if data.shape[1] > 0:
                    snippet_data_container.append(data[:N_CHANNELS])  # keep first N_CHANNELS
                    snippet_total_samples += data.shape[1]

            # 8c) Pause Spotify after snippet
            try:
                sp.pause_playback()
                print("   > Playback paused.")
            except Exception as e:
                print(f"   > [ERROR] Could not pause playback: {e}")

            # 8d) Combine snippet data
            if snippet_total_samples > 0:
                snippet_eeg = np.hstack(snippet_data_container)
            else:
                snippet_eeg = np.zeros((N_CHANNELS, 0))

            print(f"   > Collected {snippet_eeg.shape[1]} EEG samples for this track snippet.")
            if snippet_eeg.shape[1] < 50:
                print("   > Not enough EEG data to classify. Skipping track.")
                continue

            # 8e) If fixed size (e.g., REQUIRED_SAMPLES), slice or pad as needed
            if snippet_eeg.shape[1] >= REQUIRED_SAMPLES:
                snippet_eeg = snippet_eeg[:, :REQUIRED_SAMPLES]
            else:
                # Optionally pad with zeros or repeat data to reach REQUIRED_SAMPLES
                padding = REQUIRED_SAMPLES - snippet_eeg.shape[1]
                snippet_eeg = np.pad(snippet_eeg, ((0,0), (0, padding)), 'constant')

            # 8f) Extract features
            features = extract_features_from_eeg(snippet_eeg, sfreq=SFREQ)
            if features is None:
                print("   > Could not extract features. Skipping track.")
                continue

            # 8g) Predict with the loaded model
            predicted_like = False
            prob_like = 0.0

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([features])[0]  # e.g., [p0, p1]
                prob_dislike, prob_like = probs[0], probs[1]
                predicted_label = np.argmax(probs)
            else:
                predicted_label = model.predict([features])[0]
                # We'll just do a pseudo-prob with 1.0 for that label
                if predicted_label == 0:
                    prob_like = 0.0
                else:
                    prob_like = 1.0

            # 8h) Update bar chart
            bar_container[0].set_height(1 - prob_like)  # Dislike
            bar_container[1].set_height(prob_like)      # Like
            plt.draw()
            plt.pause(0.001)

            # 8i) Determine if "like" or not
            #     We assume class 1 => "like"
            if predicted_label == 1:
                predicted_like = True
                print(f"   > Predicted LIKE (prob ~ {prob_like:.2f}) => adding to new playlist.")
                liked_tracks.append(t_uri)

                # 8j) Fetch recommended tracks based on this liked track
                try:
                    # Extract the track ID
                    track_id = track_info['id']
                    print(f"   > Fetching recommendations with seed_tracks: [{track_id}]")

                    # Validate track availability
                    if is_track_available(sp, track_id, market='US'):
                        recommended_uris = fetch_recommended_tracks(
                            sp,
                            seed_tracks=[track_id],
                            limit=RECOMMENDATIONS_LIMIT,
                            market='US'
                        )
                        if recommended_uris:
                            add_tracks_to_playlist(sp, new_playlist_id, recommended_uris)
                            print(f"   > Added {len(recommended_uris)} recommended tracks based on this liked track.")
                        else:
                            print("   > No recommended tracks found.")
                    else:
                        print(f"   > Track ID {track_id} is not available in US. Skipping recommendations.")
                except Exception as e:
                    print(f"   > [ERROR] Could not fetch/add recommended tracks: {e}")
            else:
                predicted_like = False
                print(f"   > Predicted DISLIKE (prob ~ {1 - prob_like:.2f}) => not adding.")

            # Optional short wait before next track
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        # 9) Stop EEG
        try:
            board.stop_stream()
            board.release_session()
            print("[INFO] EEG stream ended.")
        except Exception as e:
            print(f"[ERROR] Error stopping EEG stream: {e}")
        plt.ioff()
        plt.show()

        # 10) Add liked tracks to the new playlist (if not already added via recommendations)
        if liked_tracks:
            add_tracks_to_playlist(sp, new_playlist_id, liked_tracks)
            print(f"[INFO] Added {len(liked_tracks)} liked tracks to playlist.")
        else:
            print("[INFO] No liked tracks found, so no tracks were added to the playlist.")

        print("Done. Enjoy your EEG-based recommendations!")


##############################################################################
#                           HELPER FUNCTIONS
##############################################################################
def get_playlist_id(playlist_url):
    """
    Extracts the playlist ID from a Spotify playlist URL or URI.
    """
    if "playlist/" in playlist_url:
        # e.g., https://open.spotify.com/playlist/<id>?si=...
        try:
            playlist_id = playlist_url.split("playlist/")[1].split("?")[0]
            return playlist_id
        except IndexError:
            raise ValueError("Invalid playlist URL format.")
    else:
        # Assume it's a plain playlist ID or URI
        return playlist_url


def fetch_all_tracks(sp, playlist_id):
    """
    Fetches all track URIs from a Spotify playlist, handling pagination.
    """
    track_uris = []
    results = sp.playlist_tracks(playlist_id, limit=100, offset=0)
    tracks = results['items']
    track_uris.extend([item['track']['uri'] for item in tracks if item['track'] is not None])

    while results['next']:
        results = sp.next(results)
        tracks = results['items']
        track_uris.extend([item['track']['uri'] for item in tracks if item['track'] is not None])

    return track_uris


def create_new_playlist(sp, user_spotify_id, user_id, model_type):
    """
    Creates a new Spotify playlist for the user to store liked tracks.
    """
    new_playlist_name = f"EEG_Liked_Songs_User_{user_id}_{model_type}"
    new_playlist = sp.user_playlist_create(
        user=user_spotify_id,
        name=new_playlist_name,
        public=True,  # Set to False for private playlists
        description="Playlist of songs the EEG model predicted I like"
    )
    new_playlist_id = new_playlist['id']
    print(f"[INFO] Created new playlist: {new_playlist_name}")
    return new_playlist_id


def add_tracks_to_playlist(sp, playlist_id, track_uris):
    """
    Adds a list of track URIs to a Spotify playlist in batches of 100.
    """
    if not track_uris:
        return
    for i in range(0, len(track_uris), 100):
        batch = track_uris[i:i+100]
        try:
            sp.playlist_add_items(playlist_id, batch)
            print(f"[INFO] Added {len(batch)} tracks to the playlist.")
        except Exception as e:
            print(f"[ERROR] Could not add tracks: {e}")


def fetch_recommended_tracks(sp, seed_tracks=None, seed_artists=None, seed_genres=None, limit=RECOMMENDATIONS_LIMIT, market='US'):
    """
    Fetches recommended tracks based on seed tracks, artists, and genres using Spotify's Recommendations API.
    """
    try:
        # Ensure seed parameters are lists or None
        params = {}
        if seed_tracks:
            params['seed_tracks'] = seed_tracks
        if seed_artists:
            params['seed_artists'] = seed_artists
        if seed_genres:
            params['seed_genres'] = seed_genres

        # Ensure at least one seed is provided
        if not any([seed_tracks, seed_artists, seed_genres]):
            print("[ERROR] At least one seed parameter must be provided.")
            return []

        recs = sp.recommendations(limit=limit, country=market, **params)
        recommended_uris = [r['uri'] for r in recs['tracks']]
        return recommended_uris
    except spotipy.exceptions.SpotifyException as e:
        print(f"[ERROR] Spotify API error: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Recommendations API failed: {e}")
        return []


def is_track_available(sp, track_id, market='US'):
    """
    Checks if a track is available in the specified market.
    """
    try:
        track = sp.track(track_id)
        available_markets = track['available_markets']
        if market in available_markets:
            return True
        else:
            print(f"[WARNING] Track ID {track_id} is not available in {market}.")
            return False
    except spotipy.exceptions.SpotifyException as e:
        print(f"[ERROR] Spotify API error while fetching track: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error while fetching track: {e}")
        return False


##############################################################################
#                           RUN MAIN
##############################################################################
if __name__ == "__main__":
    main()
