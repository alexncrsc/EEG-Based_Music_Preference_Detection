# 🧠 EEG-Based Music Preference Detection

This project uses real-time EEG data to classify whether a person likes or dislikes a song. It integrates a live Spotify playback system with an OpenBCI Cyton board, performs deep signal processing, extracts time-frequency features, and trains multiple machine learning and deep learning models — including a Transformer architecture.

---

## 🎧 Real-Time System Overview

- ✅ **Spotify API integration**: Plays random songs from “liked” and “disliked” playlists.
- ✅ **Live EEG acquisition**: Records brain activity using 8 OpenBCI Cyton channels.
- ✅ **Automated labeling**: Saves each EEG trial with track metadata and user feedback.
- ✅ **Impedance control**: Swim cap + gel + alcohol prep for impedance < 45 kΩ.

---

## 📈 Data Collection

- Total recordings: **600+ EEG trials**
- Trial length: **20 seconds**
- First 2.5 seconds cut (dongle noise)
- Rest periods: **15 seconds** between songs
- 8 electrodes: F1, F2, FT7, FT8, T7, T8, P3, P4 (varies with configuration)
- One subject, multiple recording sessions

---

## 🧼 Preprocessing Pipeline

- **High-pass filter** at 1 Hz (removes drift)
- **Notch filter** at 50 Hz (removes electrical noise)
- **Butterworth bandpass filter**: 4–40 Hz
- **Wavelet denoising**: `db4`, level 2
- **Windowing**: 3s segments, 1s stride

---

## 🧬 Feature Extraction

Extracted per window & per channel:
- Welch PSD (theta, alpha, beta, gamma)
- Hjorth parameters (mobility, complexity)
- Shannon Entropy
- Zero Crossing Rate (ZCR)
- Spectral Entropy
- Wavelet coefficients
- Fractal Dimension
- PLV (phase synchronization)

---

## 🧠 Classical ML Results (RFE + PCA + Stratified K-Fold)

| Model           | Train Acc | Test Acc | Kappa | Overfit Gap |
|----------------|-----------|----------|--------|--------------|
| **KNN**         | 72%       | **65%**  | 0.30   | 0.07         |
| SVM             | 63%       | 60%      | 0.20   | 0.03         |
| Random Forest   | 76%       | 64%      | 0.27   | 0.13         |
| XGBoost         | 68%       | 64%      | 0.28   | 0.04         |
| MLP             | 70%       | 62%      | 0.24   | 0.08         |

All models trained using:
- 🔁 5-Fold Stratified Cross-Validation  
- 🔍 RFE (Recursive Feature Elimination)  
- 📉 PCA (95% explained variance retained)

---

## 🤖 Deep Learning (Transformer Model)

A Transformer model was trained on time-windowed EEG feature sequences.

| Metric                | Result            |
|-----------------------|-------------------|
| Mean Test Accuracy    | **68.6% ± 9.2%**   |
| Mean Train Accuracy   | 72.7% ± 10.2%      |
| Input Shape           | (windows × features) |
| Optimizer             | Adam (lr=5e-4)     |
| Regularization        | Dropout + L2       |

---

## 🔬 Signal Quality

- SNR analysis done using baseline recordings
- Wavelet-based power comparison with no-stimulus baseline
- Final configuration showed better separation between “liked” and “disliked” responses (especially in alpha/theta bands)

---

## 📚 Tools & Tech Stack

- Python, NumPy, SciPy, MNE, PyWavelets  
- TensorFlow / Keras, Scikit-learn, XGBoost  
- Spotify API (Spotipy)  
- BrainFlow SDK (OpenBCI)  
- Matplotlib, Seaborn

---

## 💡 Conclusions

- Classical models perform reasonably well, with **KNN reaching 65% accuracy**
- Transformer-based models show promising results with more data
- Signal quality, filtering, and electrode placement are crucial
- Project will benefit from continued training and expansion to multiple users

---

## 📁 Project Structure

