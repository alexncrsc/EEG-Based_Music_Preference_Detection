# EEG-Based Music Preference Detection

This project explores how brain signals can be used to detect whether someone enjoys a song. Using the OpenBCI Cyton EEG headset and a real-time Spotify playback system, it records brain activity while users listen to music. The system performs deep signal processing, extracts features, and uses them to classify user preferences through machine learning and deep learning models.

## Minimum Viable Product (MVP) – Currently working on code implementation for EEG-based music preference detection

The minimum viable product includes a working pipeline that:
- Connects to Spotify to play tracks from “liked” and “disliked” playlists
- Records synchronized EEG data in real time using 8 electrodes
- Automatically saves each trial with metadata and user feedback
- Applies standard EEG preprocessing and feature extraction techniques
- Outputs labeled EEG segments ready for classification

This project serves as a foundation for brain-based music recommendation systems, with potential extensions toward multi-user generalization and live adaptive playlist generation.
