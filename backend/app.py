"""
Flask API for SpectraCraft (Phase Vocoder Voice Changer).

POST /api/process
  form-data:
    audio           - the input audio file (wav/mp3/etc)
    semitones       - float, pitch shift in semitones (e.g. -12 to 12)
    time_factor     - float, time-scale factor (e.g. 0.5 to 2.0)
  returns JSON:
    {
      "sample_rate": int,
      "processed_audio_base64": "<wav base64>",   # phase vocoder result
      "naive_audio_base64": "<wav base64>",        # naive baseline result
      "original_spectrogram_base64": "<png base64>",
      "processed_spectrogram_base64": "<png base64>",
      "naive_spectrogram_base64": "<png base64>"
    }
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os
import sys
import threading
import webbrowser
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from phase_vocoder import process_audio, VOICE_PRESETS

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

# NOTE: max frequency shown in any spectrogram = TARGET_SR / 2 (Nyquist limit).
# 44010 -> spectrograms cap out at ~22,005 Hz. Change to 44100 for the full
# 0-20kHz look like the reference image, at the cost of slower processing.
TARGET_SR = 44010  # keep processing fast; plenty for voice


def audio_to_wav_base64(x: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    # normalize to avoid clipping after processing
    peak = np.max(np.abs(x)) if len(x) else 1.0
    if peak > 1e-6:
        x = x / peak * 0.98
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def spectrogram_png_base64(x: np.ndarray, sr: int, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    # n_fft=2048 (up from 1024) gives finer frequency resolution -> sharper,
    # thinner harmonic lines instead of blurred bands.
    # hop_length=512 (up from 256) keeps the time axis proportionate to the
    # bigger n_fft without generating an excessive number of frames.
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(x, n_fft=2048, hop_length=512)),
        ref=np.max
    )

    # vmin/vmax clip the color range to a fixed 80dB window below peak.
    # Without this, quiet background noise gets spread across the color
    # scale too and the whole plot looks grainy/muddy instead of clean.
    img = librosa.display.specshow(
        D, sr=sr, hop_length=512, x_axis="time", y_axis="hz",
        ax=ax, cmap="magma", vmin=-80, vmax=0
    )
    ax.set_title(title, fontsize=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/api/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file provided"}), 400

    file = request.files["audio"]
    semitones = float(request.form.get("semitones", 0))
    time_factor = float(request.form.get("time_factor", 1.0))
    formant_factor = float(request.form.get("formant_factor", 1.0))
    try:
        x, sr = librosa.load(file, sr=TARGET_SR, mono=True)
    except Exception as e:
        return jsonify({"error": f"could not read audio: {e}"}), 400

    if len(x) == 0:
        return jsonify({"error": "empty audio"}), 400

    pv_result, naive_result = process_audio(x, semitones, time_factor, formant_factor)

    response = {
        "sample_rate": sr,
        "processed_audio_base64": audio_to_wav_base64(pv_result, sr),
        "naive_audio_base64": audio_to_wav_base64(naive_result, sr),
        "original_spectrogram_base64": spectrogram_png_base64(x, sr, "Original"),
        "processed_spectrogram_base64": spectrogram_png_base64(pv_result, sr, "SpectraCraft (corrected)"),
        "naive_spectrogram_base64": spectrogram_png_base64(naive_result, sr, "Naive Resampling (baseline)"),
    }
    return jsonify(response)


@app.route("/api/presets", methods=["GET"])
def presets():
    return jsonify(VOICE_PRESETS)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def open_browser():
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")


if __name__ == "__main__":
    # Automatically open default browser (Edge / Chrome) on startup
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        threading.Timer(1.2, open_browser).start()
    app.run(debug=True, port=5000)