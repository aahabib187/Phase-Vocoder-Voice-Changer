
# SpectraCraft

High-fidelity voice pitch and time-scale modification via STFT phase analysis.
CSE 220 project — Team Fourier & Furious (Subsection A2).

## Folder structure

```
phase-vocoder-voice-changer/
├── backend/
│   ├── app.py              # Flask API — receives audio, returns processed audio + spectrograms
│   ├── phase_vocoder.py    # Core DSP: STFT, phase unwrapping/propagation, WOLA synthesis
│   └── requirements.txt
├── frontend/
│   ├── index.html          # UI layout
│   ├── style.css           # Styling
│   └── script.js           # Uploads audio, calls the API, renders results
└── README.md
```

## How it works (mapping to the DSP)

| Step | Where | Course concept |
|---|---|---|
| Windowing + framing | `stft()` in `phase_vocoder.py` | Windowing, DFT |
| STFT | `stft()` | Short-Time Fourier Transform |
| Phase unwrapping & instantaneous frequency | `phase_vocoder_stretch()` | Phase spectra, discrete-time signal analysis |
| Phase-coherent synthesis at a new hop size | `phase_vocoder_stretch()` | Core phase vocoder trick — this is what avoids "phasiness" |
| Inverse STFT + overlap-add | `istft_overlap_add()` | Convolution / linear overlap-add reconstruction |
| Pitch shift = stretch + resample | `pitch_shift_phase_vocoder()` | Sampling rate conversion |
| Naive baseline | `naive_pitch_shift()` | Plain resampling — the "chipmunk effect" you're comparing against |

## Running it

### Running the app

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate          # Windows PowerShell / CMD
pip install -r requirements.txt
python app.py
```

This starts the server and automatically opens **SpectraCraft** in your default browser (Microsoft Edge or Google Chrome) at `http://127.0.0.1:5000`.

## Using the app

1. Upload a voice recording (WAV/MP3).
2. Set the pitch shift (semitones) and/or time-scale factor.
3. Click **Process audio**.
4. Compare the phase vocoder output against the naive resampling baseline —
   both as audio (A/B players) and as spectrograms (listen for/see the
   "phasiness" artifacts in the naive version that the phase vocoder avoids).
5. Download either result as a WAV file.

## Notes for extending

- `frame_size` and `hop_analysis` in `phase_vocoder.py` control time/frequency
  resolution trade-off — worth experimenting with and discussing in your report.
- The naive baseline currently only demonstrates pitch shifting (naive
  time-stretch-without-pitch-change isn't really a well-defined baseline),
  which is enough for the required side-by-side comparison.
- If you want a live microphone recorder instead of just file upload, that's a
  small addition to `script.js` using `navigator.mediaDevices.getUserMedia` +
  `MediaRecorder` — ask if you want that added.
