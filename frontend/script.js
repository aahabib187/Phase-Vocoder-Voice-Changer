const API_BASE = window.location.origin && window.location.origin.startsWith("http")
  ? window.location.origin
  : "http://localhost:5000";

const dropzone = document.getElementById("dropzone");
const dropzoneLabel = document.getElementById("dropzoneLabel");
const fileInput = document.getElementById("fileInput");

const pitchSlider = document.getElementById("pitchSlider");
const pitchValue = document.getElementById("pitchValue");
const timeSlider = document.getElementById("timeSlider");
const timeValue = document.getElementById("timeValue");
const presetButtons = document.querySelectorAll(".preset-btn");
const processBtn = document.getElementById("processBtn");
const eqBars = document.getElementById("eqBars");
const processBtnLabel = document.getElementById("processBtnLabel");
const errorNote = document.getElementById("errorNote");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const specOriginal = document.getElementById("specOriginal");
const specProcessed = document.getElementById("specProcessed");
const specNaive = document.getElementById("specNaive");
const frameOriginal = document.getElementById("frameOriginal");
const frameProcessed = document.getElementById("frameProcessed");
const frameNaive = document.getElementById("frameNaive");

const playerProcessed = document.getElementById("playerProcessed");
const playerNaive = document.getElementById("playerNaive");
const downloadProcessed = document.getElementById("downloadProcessed");
const downloadNaive = document.getElementById("downloadNaive");

let selectedFile = null;

function setStatus(state, label) {
  statusDot.className = `dot ${state}`;
  statusText.textContent = label;
}

function updatePitchLabel() {
  const v = parseInt(pitchSlider.value, 10);
  pitchValue.textContent = `${v > 0 ? "+" : ""}${v} st`;
}

function updateTimeLabel() {
  const v = parseFloat(timeSlider.value);
  timeValue.textContent = `${v.toFixed(2)}×`;
}

pitchSlider.addEventListener("input", () => { updatePitchLabel(); clearActivePreset(); });
timeSlider.addEventListener("input", () => { updateTimeLabel(); clearActivePreset(); });
updatePitchLabel();
updateTimeLabel();

// ---- Voice presets ----
const VOICE_PRESETS = {
  male:   { semitones: -6, time_factor: 1.0 },
  female: { semitones: 8,  time_factor: 1.0 },
  kid:    { semitones: 10, time_factor: 1.08 },
  robot:  { semitones: -12, time_factor: 1.0 },
};

function clearActivePreset() {
  presetButtons.forEach((b) => b.classList.remove("active"));
}

presetButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = VOICE_PRESETS[btn.dataset.preset];
    if (!preset) return;
    pitchSlider.value = preset.semitones;
    timeSlider.value = preset.time_factor;
    updatePitchLabel();
    updateTimeLabel();
    clearActivePreset();
    btn.classList.add("active");
  });
});

// ---- File selection ----
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  if (!file.type.startsWith("audio/")) {
    showError("Please choose an audio file (WAV or MP3).");
    return;
  }
  selectedFile = file;
  dropzoneLabel.textContent = file.name;
  processBtn.disabled = false;
  errorNote.textContent = "";
}

function showError(msg) {
  errorNote.textContent = msg;
  setStatus("error", "error");
}

// ---- Processing ----
processBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  errorNote.textContent = "";
  setStatus("busy", "processing…");
  processBtn.disabled = true;
  processBtnLabel.textContent = "Processing";
  eqBars.hidden = false;

  const formData = new FormData();
  formData.append("audio", selectedFile);
  formData.append("semitones", pitchSlider.value);
  formData.append("time_factor", timeSlider.value);

  try {
    const res = await fetch(`${API_BASE}/api/process`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server returned ${res.status}`);
    }

    const data = await res.json();
    renderResult(data);
    setStatus("ready", "done");
  } catch (err) {
    console.error(err);
    showError(err.message || "Something went wrong while processing.");
  } finally {
    processBtn.disabled = false;
    processBtnLabel.textContent = "Process audio";
    eqBars.hidden = true;
  }
});

function renderResult(data) {
  setSpectrogram(frameOriginal, specOriginal, data.original_spectrogram_base64);
  setSpectrogram(frameProcessed, specProcessed, data.processed_spectrogram_base64);
  setSpectrogram(frameNaive, specNaive, data.naive_spectrogram_base64);

  const processedUrl = `data:audio/wav;base64,${data.processed_audio_base64}`;
  const naiveUrl = `data:audio/wav;base64,${data.naive_audio_base64}`;

  playerProcessed.src = processedUrl;
  playerNaive.src = naiveUrl;
  downloadProcessed.href = processedUrl;
  downloadNaive.href = naiveUrl;
}

function setSpectrogram(frameEl, imgEl, base64Png) {
  if (!base64Png) return;
  imgEl.src = `data:image/png;base64,${base64Png}`;
  frameEl.classList.add("has-image");
}