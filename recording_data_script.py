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

# ======== Spotify Imports========
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ======== Plotting Imports========
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch, butter, filtfilt
from scipy.stats import entropy

# ==============================
#       BRAINFLOW 
# ==============================
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
params.serial_port = "/dev/cu.usbserial-DQ0081Z3"  #Portul meu Mac pt dongle

#Donge-ul are id-ul 0
board = BoardShim(0, params)


sfreq = 250 # frecventa de sample-uire Cyton
duration = 20  # secunde per stimul
required_samples = 4500 # eu am ales valoarea aceasta 
stop_event = threading.Event()

# Constante
HP_CUTOFF = 1.0           # Highpass cutoff la 1 Hz
NOTCH_FREQ = 50.0         # Notch center freq  la 50 Hz
NOTCH_BW = 2.0            # Notch bandwidth => 49–51 Hz
NOTCH_ORDER = 2           #Filtrare mai putina agresiva poate sa fie maxim 8

# ==============================
#       SPOTIFY API SETUP
# ==============================
SPOTIPY_CLIENT_ID = '8e3c71335cd64dcba2ca8aa88353bf83' #cont spotify
SPOTIPY_CLIENT_SECRET = 'e77ad5c78796468f84e8d8bc24d18203' 
SPOTIPY_REDIRECT_URI = 'http://localhost:8080/callback' 
scope = "user-read-playback-state user-modify-playback-state user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth( #functie de conectare spotify
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=scope,
    cache_path="spotipy_cache"
))

# ==============================
#        PLAYLISTS 
# ==============================
playlist_id_1 = "52oRg7up9kuXfdLXdO49zU" #liked playlist
playlist_id_2 = "2ptLyv2A6vjgXi2Wo3dRlg" #disliked playlist

track_uris_1 = [track["track"]["uri"] for track in sp.playlist_tracks(playlist_id_1)["items"]][:30] #primele 30 de piese 
track_uris_2 = [track["track"]["uri"] for track in sp.playlist_tracks(playlist_id_2)["items"]][:30]

use_playlist_1 = True


# ==============================
#   SAVE EEG DATA
# ==============================
def save_eeg_data(eeg_data, song_name, user_id):
    """
    Functie care salveaza .npy file cu tot cu timestamp pt ca sa nu se overwriteuiasca
    """
    base_path = f"eeg_recordings/user_{user_id}"
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(base_path, f"{song_name}_{timestamp}.npy")
    np.save(file_path, eeg_data)
    print(f"Se salveaza la: {file_path}")

# ============================
#   EEG ACQUISITION THREAD
# ==============================
def start_streaming(eeg_data_container):
    """
    Se aduna datele EEG in background, datele ating limita setata de 4500 sampleuri sunt schimbare in µV, apoi se aplica 
    filtrele highpass+ notch salvandu-se intr-un fisier .npy de forma (8,4500, 8 canale, 4500 sample-uri
    """
    stop_event.clear()
    board.prepare_session()
    print("Sesiunea este pregatita.")

    #documentatie openBCI:
    # gain=24, bias on, SRB2 on, SRB1 off:
    # "x{ch}060110X" => channel=ch, powerDown=0, gain=6(=24), input=0, bias=1, srb2=1, srb1=0
    for ch in range(1, 9):
        command_str = f"x{ch}060110X"
        board.config_board(command_str)
    time.sleep(0.5)  #  delay intre comenzi for safety

    board.start_stream(45000, "")
    print("A inceput stream-ul de date.")

    total_data = []
    total_samples = 0

    while not stop_event.is_set():
        time.sleep(0.01)
        data = board.get_board_data()
        if data.shape[1] > 0:
            chunk = data[1:9, :]
            total_data.append(chunk)
            total_samples += chunk.shape[1]
            print(f"Accumulated samples: {total_samples}")

        if total_samples >= required_samples:
            stop_event.set()
            break

    if total_samples >= required_samples:
        combined_data = np.hstack(total_data)[:, :required_samples]
    else:
        combined_data = np.hstack(total_data) if total_samples > 0 else np.zeros((6, 0))

    board.stop_stream()
    board.release_session()
    print("S-a oprit stream-ul.")

    #Scalare la microvolti
    if combined_data.shape[1] > 0:
        for ch_idx in range(combined_data.shape[0]):
            # Highpass la 1 Hz
            #eliminare frecventa de <1Hz si eliminare drift
            DataFilter.perform_highpass(combined_data[ch_idx, :], sfreq, HP_CUTOFF, 4, FilterTypes.BUTTERWORTH.value, 0)
            # Notch (49..51 Hz)
            #eliminare zgomot electric la 50Hz (de regula in Europa) cauzat de reteaua electrica. Intre 49 si 51 Hz
            low_notch = NOTCH_FREQ - (NOTCH_BW / 2)  # 49
            high_notch = NOTCH_FREQ + (NOTCH_BW / 2) # 51
            DataFilter.perform_bandstop(combined_data[ch_idx, :], sfreq, low_notch, high_notch,
                                        NOTCH_ORDER, FilterTypes.BUTTERWORTH.value, 0)

    eeg_data_container.append(combined_data)
    print(f"Data eeg finala si filtrata este forma: {combined_data.shape}")

# ==============================
#   SPOTIFY PLAYBACK & EEG
# ==============================
def play_music_and_collect_eeg():
    """
    Se da play la o melodie random din cele 2 playlist-uri pe rand si se inregistreaza date de la Cyton pentru 
    duratia specificata, se returneaza (6, samples) 
    """
    global use_playlist_1
    track_uri = random.choice(track_uris_1 if use_playlist_1 else track_uris_2)
    use_playlist_1 = not use_playlist_1

    track_info = sp.track(track_uri)
    track_duration = track_info['duration_ms']
    track_name = track_info['name']

    # Pozitie random de start am scazut inentionat 10 pe lanaga duration sa nu am probleme cand se da play, mai apar
    #cand este fix pe fix
    start_pos = random.randint(0, max(0, track_duration - (duration+10) * 1000))

    #start playback
    sp.start_playback(uris=[track_uri])
    sp.seek_track(start_pos)

    #Se porneste thread-ul cu stream-ul la eeg.
    eeg_data_container = []
    streaming_thread = threading.Thread(target=start_streaming, args=(eeg_data_container,))
    streaming_thread.start()

    print(f"Se reda {track_name} pentru {duration} la pozitia de start {start_pos} ms.")
    time.sleep(duration + 10)  # este o problema de delay de la dongle si ca sa se citeasca cum trebuie  am mai pus eu cateva secunde

    stop_event.set()
    streaming_thread.join()

    sp.pause_playback()
    print("Playback pe pauza.")

    time.sleep(15)  # PAUZA DE 15 SECUNDE ÎNTRE PIESE
    print("Pauza de 15 secunde.")

    if len(eeg_data_container) == 0:
        return None
    eeg_array = eeg_data_container[0]
    if eeg_array.shape[1] < required_samples:
        print("Nu sunt sample-uri destule.")
        return None

    # se salveaza data
    save_eeg_data(eeg_array, track_name, user_id=3)
    return eeg_array

# ==============================
#            MAIN
# ==============================
def main():
    """
    Se mentioneaza cate trials sunt per sesiunea de inregistrare, de mentionat ca nu exista pauze, tocmai pentru
    ca inregistrarea incepe mai tarziu din cauza trimiterii comentariilor catre Cyton subiectul are timp sa se obisnuiasca
    cu stimulul"""
    num_trials = 10  # how many songs to record
    for _ in range(num_trials):
        result = play_music_and_collect_eeg()
        if result is None:
            print("Nu sunt date suficiente")
    print("Inregistrarea a luat sfarsit")

if __name__ == "__main__":
    main()
