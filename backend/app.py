"""
Flask API for the Phase Vocoder Voice Changer.

POST /api/process
  form-data:
    audio           - the input audio file (wav/mp3/etc)
    semitones       - float, pitch shift in semitones (e.g. -12 to 12)
    time_factor     - float, time-scale factor (e.g. 0.5 to 2.0)
    formant_factor  - float, formant shift factor (e.g. 0.6 to 1.6)
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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from phase_vocoder import process_audio, VOICE_PRESETS

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

TARGET_SR = 22050  # keep processing fast; plenty for voice


def audio_to_wav_base64(x: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    # normalize to avoid clipping after processing
    peak = np.max(np.abs(x)) if len(x) else 1.0
    if peak > 1e-6:
        x = x / peak * 0.98
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def spectrogram_png_base64(x: np.ndarray, sr: int, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=110)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1024, hop_length=256)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
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
        "processed_spectrogram_base64": spectrogram_png_base64(pv_result, sr, "Phase Vocoder (corrected)"),
        "naive_spectrogram_base64": spectrogram_png_base64(naive_result, sr, "Naive Resampling (baseline)"),
    }
    return jsonify(response)


@app.route("/api/presets", methods=["GET"])
def presets():
    return jsonify(VOICE_PRESETS)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)