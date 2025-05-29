
# Real-Time Audio Analyzer

A desktop Python app for capturing and visualizing real-time audio input using Fast Fourier Transform (FFT), waveform plots, and spectrograms. 

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


