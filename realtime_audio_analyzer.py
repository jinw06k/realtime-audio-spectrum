# realtime_audio_analyzer.py
# by Jin Wook Shin

"""
How to run:

1. Create conda environment with:
    conda create -n audio-env python=3.11
2. Activate the environment:
    conda activate audio-env
3. Install required packages:
    conda install -c conda-forge numpy matplotlib python-sounddevice PySimpleGUI
4. Run the script:
    python realtime_audio_analyzer.py
"""

from collections import deque
import queue

import numpy as np
import sounddevice as sd
import PySimpleGUI as sg

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use("bmh")   
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FS          = 48000     # mic sample rate in Hz
BLOCKSIZE   = 2048
HISTORY_SEC = 10       # adjust to change history length

class AudioStream:
    def __init__(self, device_idx: int, q: queue.Queue):
        self.q = q
        self.device_idx = device_idx
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        mono = np.mean(indata, axis=1) 
        try:
            self.q.put_nowait(mono.copy())
        except queue.Full:
            pass                   

    def start(self):
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=FS,
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


class RealTimeAudioApp:
    MODES = ("FFT", "Waveform", "Spectrogram")

    def __init__(self):
        self.q          = queue.Queue(maxsize=30)
        self.history    = deque(maxlen=HISTORY_SEC * FS)
        self.audio      = None
        self.running    = False
        self._build_gui()

    # ────────────── GUI ──────────────
    def _build_gui(self):
        sg.theme("TanBlue")
        device_list = self._get_input_devices()

        controls = [
            [sg.Text("Audio Device", size=(12,1)),
             sg.Combo(device_list, default_value=device_list[0], key="DEVICE", readonly=True, expand_x=True)],
            [sg.Text("Mode", size=(12,1)),
             sg.Combo(self.MODES, default_value="FFT", key="MODE", readonly=True, expand_x=True)],
            [sg.Text("Min Hz", size=(12,1)),
             sg.Slider(range=(0, FS//2), resolution=10, orientation="h", size=(40,15), key="FMIN", enable_events=True)],
            [sg.Text("Max Hz", size=(12,1)),
             sg.Slider(range=(0, FS//2), resolution=10, orientation="h", size=(40,15), default_value=FS//2, key="FMAX", enable_events=True)],
            [sg.Button("Start", key="START_STOP", size=(8,1)),
             sg.Push(), 
             sg.Button("Exit", button_color=('white', '#CC3333'), size=(8,1))],
        ]

        layout = [
            [sg.Frame("Controls", controls, expand_x=True, relief=sg.RELIEF_SUNKEN)],
            [sg.Canvas(size=(800,520), key="CANVAS", expand_x=True, expand_y=True)],
        ]

        self.window = sg.Window(
            "Real-Time Audio Analyzer",
            layout, finalize=True, resizable=True, element_justification="center"
        )

        self.fig, self.ax = plt.subplots()
        self.canvas_agg = FigureCanvasTkAgg(self.fig, self.window["CANVAS"].TKCanvas)
        self.canvas_agg.get_tk_widget().pack(fill='both', expand=1)

    def _get_input_devices(self):
        devs = [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices())
                if d['max_input_channels'] > 0]
        return devs or ["No Devices Found"]

    # ────────────── Main ──────────────
    def run(self):
        while True:
            event, values = self.window.read(timeout=40)

            if event in (sg.WIN_CLOSED, "Exit"):
                self._stop_stream()
                break

            if event == "FMIN":
                if values["FMIN"] > values["FMAX"]:
                    self.window["FMIN"].update(value=values["FMAX"])
            elif event == "FMAX":
                if values["FMAX"] < (values["FMIN"]-1):
                    self.window["FMAX"].update(value=values["FMIN"])

            if event == "START_STOP":
                if not self.running:
                    self._start_stream(values)
                else:
                    self._stop_stream()


            if self.running:
                self._collect_audio()
                self._update_plot(values)

        self.window.close()

    # ────────────── Microphone ──────────────
    def _start_stream(self, values):
        try:
            device_idx = int(values["DEVICE"].split(":")[0])
        except Exception:
            sg.popup_error("No device selected!")
            return
        self.audio = AudioStream(device_idx, self.q)
        self.audio.start()
        self.history.clear()
        self.running = True
        self.window["START_STOP"].update("Stop")

    def _stop_stream(self):
        if self.audio:
            self.audio.stop()
        self.running = False
        self.window["START_STOP"].update("Start")

    def _collect_audio(self):
        while True:
            try:
                self.history.extend(self.q.get_nowait())
            except queue.Empty:
                break

    # ────────────── Plot ──────────────
    def _update_plot(self, values):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

        mode = values["MODE"]
        if mode == "FFT":
            self._plot_fft(values)
        elif mode == "Waveform":
            self._plot_waveform()
        else:
            self._plot_spectrogram(values)

        self.canvas_agg.draw()

    def _plot_fft(self, values):
        if len(self.history) < BLOCKSIZE:
            return
        data  = np.array(self.history)[-BLOCKSIZE:]
        yf    = np.fft.rfft(data * np.hanning(len(data)))
        xf    = np.fft.rfftfreq(len(data), 1 / FS)
        mag   = 20 * np.log10(np.abs(yf) + 1e-6)
        fmin  = values["FMIN"]
        fmax  = max(values["FMAX"], fmin + 1)

        mask = (xf >= fmin) & (xf <= fmax)
        self.ax.plot(xf[mask], mag[mask])

        self.ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        self.ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
        self.ax.minorticks_on() 
        
        x_range = fmax - fmin
        self.ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
        
        self.ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)",
                    title="FFT", xlim=(fmin, fmax), ylim=(-120, 20))

    def _plot_waveform(self):
        if not self.history:
            return
        data = np.array(self.history)
        t    = np.linspace(-len(data)/FS, 0, len(data))
        self.ax.plot(t, data)
        self.ax.set(xlabel="Time (s)", ylabel="Amplitude",
                    title="Waveform",
                    xlim=(t[0], t[-1]), ylim=(-1, 1))
        self.ax.grid(True)

    def _plot_spectrogram(self, values):
        if len(self.history) < FS:
            return
        data = np.array(self.history)

        fmin = 0
        fmax = FS//2
        if values:
            fmin = max(0, values["FMIN"])
            fmax = min(FS//2, values["FMAX"])
        
        Pxx, freqs, bins, im = self.ax.specgram(
            data, NFFT=1024, Fs=FS, noverlap=512, cmap="magma"
        )
        
        title = "Spectrogram"
        if values and (values["FMIN"] > 0 or values["FMAX"] < FS//2):
            title += f" ({int(fmin)}-{int(fmax)} Hz)"
        
        self.ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)",
                    title=title, ylim=(fmin, fmax))
        self.fig.colorbar(im, ax=self.ax, label="Power/Frequency (dB/Hz)")

if __name__ == "__main__":
    RealTimeAudioApp().run()