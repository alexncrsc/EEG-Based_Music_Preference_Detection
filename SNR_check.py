import numpy as np
import mne
import pywt

# --- param
baseline_file = "/Users/alexnicorescu/eeg_recordings/user_3/baseline" 
eeg_file = "/Users/alexnicorescu/eeg_recordings/user_3/liked/Cireasa De Pe Tort Interlude_20250223_230110.npy" 
sampling_rate = 250  # sfreq
cut_start_sec = 2.5  
low_freq, high_freq = 4, 40  # bandpass filter range
frequencies = np.linspace(4, 40, 50)  # freq range
wavelet = 'db4'  

# load data
baseline_data = np.load(baseline_file)  
eeg_data = np.load(eeg_file)  

#cut primele 2.5 sec noisy
cut_samples = int(cut_start_sec * sampling_rate)
eeg_data = eeg_data[:, cut_samples:]
baseline_data = baseline_data[:, cut_samples:]

#MNE object 
channel_names = ["0", "1", "2", "3", "4", "5", "6", "7"]
info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types="eeg")
raw_eeg = mne.io.RawArray(eeg_data, info)
raw_baseline = mne.io.RawArray(baseline_data, info)

# 4-40hz filter 
raw_eeg.filter(l_freq=low_freq, h_freq=high_freq, fir_design="firwin")
raw_baseline.filter(l_freq=low_freq, h_freq=high_freq, fir_design="firwin")

# extract
eeg_data = raw_eeg.get_data()
baseline_data = raw_baseline.get_data()

# --- WAVELET denoise
wavelet_coeffs_signal = []
wavelet_coeffs_noise = []

for i, ch in enumerate(channel_names):
    signal = eeg_data[i, :]
    noise = baseline_data[i, :]
    
    
    coef_signal, _ = pywt.cwt(signal, scales=frequencies, wavelet=wavelet, sampling_period=1/sampling_rate)
    coef_noise, _ = pywt.cwt(noise, scales=frequencies, wavelet=wavelet, sampling_period=1/sampling_rate)
    
    wavelet_coeffs_signal.append(coef_signal)
    wavelet_coeffs_noise.append(coef_noise)

wavelet_coeffs_signal = np.array(wavelet_coeffs_signal)  
wavelet_coeffs_noise = np.array(wavelet_coeffs_noise)  

# --- SNR ---
signal_power = np.mean(np.abs(eeg_data) ** 2, axis=(1, 2))  # puterea signal
noise_power = np.mean(np.abs(wavelet_coeffs_noise) ** 2, axis=(1, 2))  #puterea noise
snr_db = 10 * np.log10(signal_power / noise_power)  # snr in dB
print(signal_power / noise_power)
print(snr_db)


#[ 1.1564655   3.07143703  2.7870392   0.18962716 -2.51810512  1.27760687]

#[1.11934121 0.36587099 2.51877218 3.1293008  2.52298988 3.23679289 7.26508879 6.19988793]
