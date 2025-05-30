# realtime_audio_analyzer.py
# by Jin Wook Shin

"""
How to run:

1. conda create -n audio-env python=3.11
2. conda activate audio-env
3. conda install -c conda-forge numpy matplotlib python-sounddevice PySimpleGUI scipy
4. python realtime_audio_analyzer.py
"""

from collections import deque
import queue, os, sys

import numpy as np
import sounddevice as sd
import PySimpleGUI as sg
from scipy.io import wavfile

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use("bmh")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FS          = 48000     # mic sample rate in Hz
BLOCKSIZE   = 2048
HISTORY_SEC = 10       # adjust to change history length

class AudioStream:                                     # microphone
    def __init__(self, device_idx: int, q: queue.Queue, fs:int):        
        self.q = q
        self.fs = fs
        self.device_idx = device_idx
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        if status: 
            print(status, file=sys.stderr)
        try: 
            self.q.put_nowait(np.mean(indata, axis=1).copy()) 
        except queue.Full: 
            pass
            
    def start(self):
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=self.fs,
                blocksize=BLOCKSIZE,
                device=self.device_idx,
                channels=1,
                callback=self.audio_callback,
            )
            self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class LoopingWavStream:                           # WAV file
    def __init__(self, filepath:str, q:queue.Queue):
        self.filepath, self.q   = filepath, q
        self.running, self.ptr  = False, 0
        self.data, self.fs      = None, FS

    def start(self):
        try:
            self.fs, data = wavfile.read(self.filepath)
            if data.ndim > 1: data = data[:,0] 
            self.data    = data.astype(np.float32)
            mx           = np.max(np.abs(self.data)) or 1.0
            self.data   /= mx      
            self.running = True
        except Exception as e:
            sg.popup_error(f"Could not load WAV: {e}")

    def stop(self): self.running = False

    def pump(self):
        if not self.running or self.data is None: return
        end = self.ptr + BLOCKSIZE
        if end >= len(self.data):
            chunk = np.concatenate((self.data[self.ptr:], self.data[:end % len(self.data)]))
            self.ptr = end % len(self.data)
        else:
            chunk = self.data[self.ptr:end]; self.ptr = end
        try: self.q.put_nowait(chunk.copy())
        except queue.Full: pass


class RealTimeAudioApp:
    MODES = ("FFT", "Waveform", "Spectrogram")

    def __init__(self):
        self.q          = queue.Queue(maxsize=30)
        self.audio      = None   
        self.fs         = FS  
        self.history    = deque(maxlen=HISTORY_SEC * self.fs)
        self.running    = False
        self.wav_files  = self._scan_wavs()
        self._build_gui()

    # ────────────── GUI ──────────────
    def _scan_wavs(self):
        os.makedirs("wav_files", exist_ok=True)
        return [f for f in os.listdir("wav_files") if f.lower().endswith(".wav")]

    def _device_list(self):
        wav_entries = [f"WAV: {f}" for f in self.wav_files]
        mic_entries = [f"{i}: {d['name']}"
                       for i,d in enumerate(sd.query_devices()) if d['max_input_channels']>0]
        return wav_entries + mic_entries or ["No Devices Found"]

    def _build_gui(self):
        sg.theme("TanBlue")
        controls = [
            [sg.Text("Audio Source", size=(12,1)),
             sg.Combo(self._device_list(), key="SRC", default_value=self._device_list()[0],
                      readonly=True, expand_x=True)],
            [sg.Text("Mode", size=(12,1)),
             sg.Combo(self.MODES,   key="MODE", default_value="FFT",
                      readonly=True, expand_x=True)],
            [sg.Text("Min Hz", size=(12,1)),
             sg.Slider((0, FS//2), 10, orientation="h",
                       key="FMIN", enable_events=True, size=(40,15))],
            [sg.Text("Max Hz", size=(12,1)),
             sg.Slider((0, FS//2), FS//2, orientation="h",
                       key="FMAX", enable_events=True, size=(40,15))],
            [sg.Button("Start", key="TOGGLE", size=(8,1)), sg.Push(),
             sg.Button("Exit",  button_color=("white","#C33"), size=(8,1))],
        ]
        layout = [[sg.Frame("Controls", controls, expand_x=True)],
                  [sg.Canvas(size=(800,520), key="CANVAS", expand_x=True, expand_y=True)]]

        self.win  = sg.Window("Real-Time Audio Analyzer", layout,
                              finalize=True, resizable=True, element_justification="center")
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, self.win["CANVAS"].TKCanvas)
        self.canvas.get_tk_widget().pack(fill="both", expand=1)

    # ────────────── Main ──────────────
    def run(self):
        while True:
            ev, val = self.win.read(timeout=40)
            if ev in (sg.WIN_CLOSED, "Exit"): self._stop(); break

            if ev == "FMIN" and val["FMIN"] > val["FMAX"]:
                self.win["FMIN"].update(value=val["FMAX"])
            if ev == "FMAX" and val["FMAX"] < val["FMIN"]:
                self.win["FMAX"].update(value=val["FMIN"])

            if ev == "TOGGLE":
                self._start(val) if not self.running else self._stop()

            if self.running:
                if isinstance(self.audio, LoopingWavStream): self.audio.pump()
                self._collect_audio()
                self._draw(val)

        self.win.close()

    def _start(self, val):
        self._stop()        
        src = val["SRC"]
        if src.startswith("WAV:"):
            path = os.path.join("wav_files", src.replace("WAV: ",""))
            self.audio = LoopingWavStream(path, self.q); self.audio.start()
            self.fs    = self.audio.fs
        else:
            dev_idx    = int(src.split(":")[0])
            self.fs    = FS
            self.audio = AudioStream(dev_idx, self.q, self.fs); self.audio.start()

        self.history = deque(maxlen=HISTORY_SEC * self.fs)
        self.running = True
        self.win["TOGGLE"].update("Stop")

    def _stop(self):
        if self.audio: self.audio.stop()
        self.running, self.audio = False, None
        self.win["TOGGLE"].update("Start")

    def _collect_audio(self):
        try:
            while True: self.history.extend(self.q.get_nowait())
        except queue.Empty:
            pass

    def _draw(self, val):
        self.fig.clf(); self.ax = self.fig.add_subplot(111)
        mode = val["MODE"]
        if mode == "FFT":        self._plot_fft(val)
        elif mode == "Waveform": self._plot_wave(val)
        else:                    self._plot_spec(val)
        self.canvas.draw()

    # ────────────── Plot ──────────────
    def _plot_fft(self, v):
        if len(self.history) < BLOCKSIZE: return
        data = np.array(self.history)[-BLOCKSIZE:]
        yf   = np.fft.rfft(data * np.hanning(len(data)))
        xf   = np.fft.rfftfreq(len(data), 1/self.fs)
        mag  = 20*np.log10(np.abs(yf)+1e-6)
        fmin, fmax = v["FMIN"], max(v["FMAX"], v["FMIN"]+1)
        mask = (xf>=fmin)&(xf<=fmax)
        self.ax.plot(xf[mask], mag[mask])
        self.ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)",
                    title="FFT", xlim=(fmin,fmax), ylim=(-120,20))
        self.ax.grid(True)

    def _plot_wave(self, _):
        if not self.history: return
        data = np.array(self.history)
        t    = np.linspace(-len(data)/self.fs, 0, len(data))
        self.ax.plot(t,data)
        self.ax.set(xlabel="Time (s)", ylabel="Amplitude",
                    title="Waveform", xlim=(t[0],t[-1]), ylim=(-1,1))
        self.ax.grid(True)

    def _plot_spec(self, v):
        if len(self.history) < self.fs: return
        data = np.array(self.history)
        fmin, fmax = max(0,v["FMIN"]), min(self.fs//2, v["FMAX"])
        Pxx,_,_,im = self.ax.specgram(data, NFFT=1024, Fs=self.fs,
                                      noverlap=512, cmap="magma")
        self.ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)",
                    title=f"Spectrogram ({int(fmin)}–{int(fmax)} Hz)",
                    ylim=(fmin,fmax))
        self.fig.colorbar(im, ax=self.ax, label="Power/Freq (dB/Hz)")


if __name__ == "__main__":
    RealTimeAudioApp().run()
