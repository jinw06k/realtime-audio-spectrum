
# Real-Time Audio Analyzer

A cross-platform desktop Python app for capturing and visualizing real-time audio input using Fast Fourier Transform (FFT), waveform plots, and spectrograms. Built with `PySimpleGUI`, `matplotlib`, and `sounddevice`.

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

> 📸 _Add screenshots here to showcase the interface, e.g., FFT view, waveform, spectrogram_

## Usage Notes

- Sampling rate is set to `48 kHz`.
- Block size (`BLOCKSIZE`) is `2048`, which balances resolution and responsiveness.
- Up to 10 seconds of audio is stored and displayed in the waveform/spectrogram views.
- The min/max frequency sliders are locked to prevent invalid ranges (e.g., min > max).

## Known Limitations

- Spectrogram view does not resize dynamically to window size.
- Requires a functioning microphone input device.

## License

MIT License
