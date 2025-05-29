
# Real-Time Audio Analyzer
A Python app for capturing and visualizing real-time audio input using Fast Fourier Transform (FFT), waveform plots, and spectrograms. 

## Features

- 🎤 **Input Device Selector** – Choose any microphone connected to your system.
- 📈 **Live Visualization Modes**:
  - FFT: View real-time frequency spectrum
  - Waveform: Plot of the last 10 seconds of audio
  - Spectrogram: Rolling time-frequency analysis
- 🎚️ **Frequency Range Sliders** – Set min and max Hz to zoom into specific bands in the FFT view
- ⏯️ **Start/Stop Button** – Toggle audio streaming on the fly

## Installation

### 1. Create a Conda environment:
```bash
conda create -n audio-env python=3.11
conda activate audio-env
```

### 2. Install dependencies:
```bash
conda install -c conda-forge numpy matplotlib python-sounddevice PySimpleGUI
```

## Running the App

```bash
python realtime_audio_analyzer.py
```

## UI Overview
<img width="865" alt="Image" src="https://github.com/user-attachments/assets/1994a13f-2b59-473e-8a0d-8a35e4d04770" /> 
<img width="865" alt="Image" src="https://github.com/user-attachments/assets/6113abee-0a30-48c0-841c-e7591420de54" /> 
<img width="865" alt="Image" src="https://github.com/user-attachments/assets/1ecff3ea-0d7e-470a-822f-92fa637d1502" />
