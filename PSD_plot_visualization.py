import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch
import pywt

# ==============================
# PARAMETRI EEG
# ==============================
fs = 250  # Frecventa de esantionare
notch_freq = 50  # Zgomot de la retea 50 Hz
freq_bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

#cut 2.5 sec
skip_secs = 2.5
skip_samples = int(fs * skip_secs)

# ==============================
# FOLDER PATHS 
# ==============================
liked_path = "eeg_recordings/user_3/liked/"
disliked_path = "eeg_recordings/user_3/disliked/"

# ==============================
# Functii filtrare si denoising
# ==============================

def bandpass_filter(data, lowcut=4, highcut=49, fs=256, order=4):
    """Filtru 4-49Hz"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return np.array([filtfilt(b, a, channel) for channel in data])



def wavelet_denoise(data, wavelet='db4', level=4):
    """
    wavelet_denoise
    """
    denoised = np.zeros_like(data)
    n_channels, n_samples = data.shape
    for ch in range(n_channels):
        coeffs = pywt.wavedec(data[ch], wavelet, level=level)
        # Estimeaza zgomotul pe baza coeficientilor de detaliu de la nivelul cel mai inalt
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(n_samples))
        coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, wavelet)
        denoised[ch] = np.real(rec[:n_samples])
    return denoised

# ==============================
# Incarcarea date
# ==============================
def load_and_preprocess_eeg_data(path):
    """
   Incaracare si preprocesare
    """
    eeg_data = []
    for file in os.listdir(path):
        if file.endswith(".npy"):
            filepath = os.path.join(path, file)
            data = np.load(filepath) 
            # Taierea primelor 2,5 secunde
            if data.shape[1] > skip_samples:
                data = data[:, skip_samples:]
            else:
                continue
            # Aplica filtrarea bandpass 4-40 Hz
            data = bandpass_filter(data, lowcut=4, highcut=40, fs=fs)
            # Aplica denoising cu wavelet
            data = wavelet_denoise(data, wavelet='db4', level=4)
            eeg_data.append(data)
    return np.array(eeg_data)

# Incarca datele preprocesate
liked_data = load_and_preprocess_eeg_data(liked_path)
disliked_data = load_and_preprocess_eeg_data(disliked_path)

if liked_data.size == 0 or disliked_data.size == 0:
    print("Nu s-au gasit fisier in ambele foldere")
    exit()

print("\n===== Rezumat Date EEG =====")
print(f"Total files liekd {len(liked_data)}")
print(f"Total files disliked: {len(disliked_data)}")
print(f"Forma unui file EEG: {liked_data[0].shape} (Canale x Samples)")

# ==============================
# FEATURE EXTRACTION: Calculul puterii în benzi
# ==============================
def compute_bandpower(eeg_signal):
    """
    Psd metoda welch, calculeaza puterea in diferite benzi pe semnalul unui singur canal eeg.
    """
    band_powers = {}
    freqs, psd = welch(eeg_signal, fs, nperseg=min(256, eeg_signal.shape[0]))
    for band, (low, high) in freq_bands.items():
        band_power = np.sum(psd[(freqs >= low) & (freqs <= high)])
        band_powers[band] = band_power
    return band_powers

#stocheaza caracteristicele de band power
liked_band_power = {band: [] for band in freq_bands}
disliked_band_power = {band: [] for band in freq_bands}

for eeg_file in liked_data:
    for ch in range(eeg_file.shape[0]):  # Pentru fiecare canal
        power = compute_bandpower(eeg_file[ch])
        for band in freq_bands:
            liked_band_power[band].append(power[band])

for eeg_file in disliked_data:
    for ch in range(eeg_file.shape[0]):
        power = compute_bandpower(eeg_file[ch])
        for band in freq_bands:
            disliked_band_power[band].append(power[band])

# converite numpy array
liked_band_power = {band: np.array(values) for band, values in liked_band_power.items()}
disliked_band_power = {band: np.array(values) for band, values in disliked_band_power.items()}

# ==============================
# PLOTAREA REZULTATELOR
# ==============================

# Plot distributie puteri fiecare banda
plt.figure(figsize=(12, 8))
for i, band in enumerate(freq_bands):
    plt.subplot(3, 2, i + 1)
    plt.hist(liked_band_power[band], bins=30, alpha=0.6, color='blue', label='Liked', density=True)
    plt.hist(disliked_band_power[band], bins=30, alpha=0.6, color='red', label='Disliked', density=True)
    plt.title(f'Distributia puterii în banda {band.capitalize()}')
    plt.xlabel("Putere")
    plt.ylabel("Densitate")
    plt.legend()

plt.tight_layout()
plt.show()

# Plot comparatia mediei puterii in benzi pentru liked vs. disliked
plt.figure(figsize=(10, 5))
bands = list(freq_bands.keys())
liked_means = [np.mean(liked_band_power[band]) for band in bands]
disliked_means = [np.mean(disliked_band_power[band]) for band in bands]

x = np.arange(len(bands))
plt.bar(x - 0.2, liked_means, width=0.4, label="Liked", color='blue', alpha=0.7)
plt.bar(x + 0.2, disliked_means, width=0.4, label="Disliked", color='red', alpha=0.7)
plt.xticks(x, bands)
plt.ylabel("Putere medie")
plt.title("Comparatie medie putere pe benzi")
plt.legend()
plt.show()
