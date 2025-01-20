#Spotify_recommendation_EEG_data_model


## Description
This project involves using EEG data to control Spotify playback based on the user's brain activity interpreted through machine learning models. The application integrates with Spotify's API to manage music playlists based on EEG signals that suggest user preferences.

## Table of Contents
- [Functionality 1: EEG Music Preference Analysis](#functionality-1-eeg-music-preference-analysis)
- [Functionality 2: Live EEG Data Processing and Prediction](#functionality-2-live-eeg-data-processing-and-prediction)
- [Functionality 3: Spotify Playlist Management Based on EEG](#functionality-3-spotify-playlist-management-based-on-eeg)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Functionality 1: EEG Music Preference Analysis
This code handles EEG data acquisition, processes the EEG signals to extract features, and uses pre-trained models to predict the user's music preference (like/dislike).

### Requirements
- BrainFlow
- NumPy
- Matplotlib
- Joblib

### How to Run
To run this script, ensure your EEG device is properly set up and connected. Modify the serial port settings as per your configuration. Use the following command:
```bash
python eeg_music_preference.py
