
# SpectraCraft

High-fidelity voice pitch and time-scale modification via STFT phase analysis.
CSE 220 project — Team Fourier & Furious (Subsection A2).

## Key Features

- **Independent Pitch & Time Scaling:** Decouple pitch and tempo using phase-coherent STFT analysis and Weighted Overlap-Add (WOLA) synthesis.
- **Formant Shifting via Cepstral Liftering:** Estimate vocal tract spectral envelopes and modify formant resonances independently of pitch to eliminate the "chipmunk effect".
- **Voice Character Presets:** Instant one-click presets for **Male**, **Female**, **Kid**, and **Robot** vocal profiles.
- **Side-by-Side Comparison:** Interactive 3-way spectral analysis (Original, Naive Resampling Baseline, and SpectraCraft Phase-Corrected) with synchronized waveforms and playback.
- **Interactive UI & Export:** Clean control surface with real-time waveform displays, baseline comparison toggles, and direct WAV export.

## Folder Structure

```
phase-vocoder-voice-changer/
├── backend/
│   ├── app.py              # Flask API — receives audio, returns processed audio + spectrograms
│   ├── phase_vocoder.py    # Core DSP: STFT/iSTFT, phase propagation, formant liftering, presets
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Web UI layout & audio transport
│   ├── style.css           # Styling
│   └── script.js           # Client logic, API integration, waveform visualization
├── Implementation Details (presentation slides)/
│   └── SpectraCraft_presentation.pdf   # Project presentation slides & architecture details
└── README.md
```

## How It Works (Mapping to DSP Concepts)

| Step | Function / Location | Course Concept |
|---|---|---|
| Windowing + framing | `stft()` in `phase_vocoder.py` | Hann windowing, Discrete Fourier Transform (DFT) |
| STFT analysis | `stft()` | Short-Time Fourier Transform (time-frequency representation) |
| Phase unwrapping & instantaneous frequency | `phase_vocoder_stretch()` | Phase spectra, discrete-time frequency estimation, phase unwrapping |
| Phase-coherent synthesis at a new hop size | `phase_vocoder_stretch()` | Phase propagation — prevents phase cancellation and "phasiness" |
| Inverse STFT + overlap-add | `istft_overlap_add()` | Weighted Overlap-Add (WOLA) synthesis, signal reconstruction |
| Time stretching (tempo change) | `time_stretch_phase_vocoder()` | Synthesis hop-size scaling without altering pitch |
| Pitch shift = stretch + resample | `pitch_shift_phase_vocoder()` | Time-scale modification followed by linear resampling |
| Formant shifting & envelope estimation | `formant_shift()`, `compute_spectral_envelope()` | Homomorphic signal processing, Cepstral analysis & liftering |
| Naive baseline | `naive_pitch_shift()` | Direct resampling (tape-speed effect) — demonstrates pitch/tempo coupling |

## Running It

### Prerequisites

- Python 3.10+ (tested with Python 3.12 / 3.13)
- Google Chrome, Microsoft Edge, or any modern web browser

### Running the App

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate          # Windows PowerShell / CMD
# On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

This starts the server and automatically opens **SpectraCraft** in your default browser at `http://127.0.0.1:5000`.

## Using the App

1. **Upload an audio file:** Drop a voice recording (WAV or MP3) into the file slot or click to browse.
2. **Adjust parameters:**
   - **Pitch shift:** Shift pitch up or down (-12 to +12 semitones).
   - **Time scale:** Speed up or slow down speech (0.50× to 2.00×) without altering pitch.
   - **Formant shift:** Scale vocal tract resonances (0.50× to 2.00×) to reshape vocal timbre.
   - **Voice presets:** Select **Male**, **Female**, **Kid**, or **Robot** for quick predefined settings.
3. **Compare with baseline:** Use the **Show naive resampling** toggle to show or hide the naive pitch-shift baseline.
4. **Process audio:** Click **Process audio** to run the DSP pipeline.
5. **Inspect & compare results:**
   - **Playback rack:** Listen to the Original, Naive Resampled, and SpectraCraft outputs with animated waveform meters.
   - **Spectral analysis:** Examine the side-by-side spectrograms (Original, Naive Resampling, and SpectraCraft Phase-Corrected) in magma colormap to observe harmonic structure and artifact suppression.
6. **Download:** Click **Download WAV** under either output track to save the processed audio file.

## Notes & Technical Details

- **Time/Frequency Resolution:** `frame_size=2048` and `hop_analysis=512` in `phase_vocoder.py` provide high frequency resolution suitable for voice harmonics while maintaining temporal clarity.
- **Cepstral Liftering:** Formants are isolated by taking the real cepstrum (`irfft` of log magnitude) and applying a low-quefrency rectangular lifter (`cutoff=30`), separating the vocal tract filter from glottal pitch impulses.
- **Spectrogram Rendering:** High-resolution spectrograms are generated server-side using Librosa's STFT (2048 FFT bins, 512 hop) clamped to an 80 dB dynamic range below peak to ensure clean, publication-quality spectral visualizations.
