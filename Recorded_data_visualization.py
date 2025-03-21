import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Se aplica un filtru butterworth bandpass pe semnalul deja procesat.

    Parametrii: 

    semnalul este semnalul pe canal,
    lowcut, highcut- frecventele de interval
    fs-sampling rate
    order- filter order
    se returneaza semnalul filtart
    Applies a Butterworth bandpass filter to the signal.
    
   
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered

def main():
    # ----------------------------------------------------
    # 1) Parametrii initiali
    # ----------------------------------------------------
    #file_path = "/Users/alexnicorescu/eeg_recordings/user_2/baseline_23_februarie.npy" #cale .npy
    file_path = "/Users/alexnicorescu/eeg_recordings/user_3/NA_Hymns_in_Dissonance_disliked_20250306_152208.npy"
    fs = 250  # Sampling rate in Hz
    skip_secs = 2.5  # se ignora primele 2.5 sec zgomot de la conexiunea cu Cyton
    
    # Parametrii filtru bandpass
    lowcut = 4.0   
    highcut = 40.0 
    filter_order = 4

    # ----------------------------------------------------
    # 2) Incaracare date
    # ----------------------------------------------------
    eeg_data = np.load(file_path)  
    num_channels, num_samples = eeg_data.shape
    print(f"Datele EEG au shape-ul: {eeg_data.shape}")

    # ----------------------------------------------------
    # 3) Se inlatureaza primele 2.5 secunde ft zgomotoase
    # ----------------------------------------------------
    skip_samples = int(skip_secs * fs) # *250
    if skip_samples >= num_samples:
        raise ValueError("skip_secs > sample-uri") #verificare extra nenecesara
    eeg_data = eeg_data[:, skip_samples:]
    print(f"Shape-ul datelor EEG dupa taierea de {skip_secs} s: {eeg_data.shape}")

    # Update nr de sample-uri
    _, num_samples = eeg_data.shape

    # ----------------------------------------------------
    # 4) Aplicare bandpass filter (4-40 Hz)
    # ----------------------------------------------------
    for ch in range(num_channels):
        eeg_data[ch] = bandpass_filter(eeg_data[ch], lowcut, highcut, fs, order=filter_order)

    # ----------------------------------------------------
    # 5) Create axa de timp
    # ----------------------------------------------------
    # Reset time axis to start at 0 seconds for convenience
    time_axis = np.arange(num_samples) / fs

    # ----------------------------------------------------
    # 6)Plotare fiecare canal in domneiu
    # ----------------------------------------------------
    fig, axes = plt.subplots(num_channels, 1, sharex=True, figsize=(12, 10))
    if num_channels == 1:
        axes = [axes]  # verificare ca este iterabil

    for i in range(num_channels):
        axes[i].plot(time_axis, eeg_data[i], label=f"Channel {i+1}")
        axes[i].set_ylabel("Amplitude (µV)")
        axes[i].grid(True)
        axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    plt.suptitle("EEG Data (4–40 Hz Bandpass, Ignoring First 2.5 s)")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # 7) Plotare PSD metoda Welch
    # ----------------------------------------------------
    fig, axes = plt.subplots(num_channels, 1, sharex=True, figsize=(12, 10))
    if num_channels == 1:
        axes = [axes]

    
    psd_min = float("inf")
    psd_max = float("-inf")

    for i in range(num_channels):
        freqs, psd = welch(eeg_data[i], fs=fs, nperseg=256)
        psd_min = min(psd_min, psd.min()) 
        psd_max = max(psd_max, psd.max())  

    for i in range(num_channels):
        freqs, psd = welch(eeg_data[i], fs=fs, nperseg=256)
        axes[i].semilogy(freqs, psd, label=f"Channel {i+1} PSD")
        axes[i].set_ylim([psd_min * 0.9, psd_max * 1.1]) 
        axes[i].grid(True)
        axes[i].legend(loc="upper right")
        axes[i].set_xlim([0, 60]) 

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.suptitle("EEG Power Spectral Density (PSD) (4–40 Hz Bandpass)")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
