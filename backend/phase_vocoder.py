"""
Phase Vocoder core DSP.

Implements:
  - STFT / inverse STFT (overlap-add)
  - Phase-vocoder time-stretching (with phase unwrapping + propagation,
    so stretched audio stays phase-coherent instead of sounding "robotic")
  - Pitch shifting = time-stretch by ratio, then resample back to original
    duration (this decouples pitch from tempo)
  - Formant shifting via cepstral liftering (moves vocal-tract resonances
    independently of pitch)
  - A naive baseline (plain resampling) for comparison, which is the
    "chipmunk effect" method that couples pitch and speed together
"""

import numpy as np


def hann_window(size: int) -> np.ndarray:
    return np.hanning(size)


def stft(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Short-Time Fourier Transform. Returns array of shape (n_frames, frame_size//2+1)."""
    window = hann_window(frame_size)
    n_frames = 1 + (len(x) - frame_size) // hop_size
    if n_frames < 1:
        n_frames = 1
        # pad short signals
        x = np.pad(x, (0, frame_size - len(x)))

    frames = np.empty((n_frames, frame_size // 2 + 1), dtype=complex)
    for i in range(n_frames):
        start = i * hop_size
        segment = x[start:start + frame_size]
        if len(segment) < frame_size:
            segment = np.pad(segment, (0, frame_size - len(segment)))
        windowed = segment * window
        frames[i] = np.fft.rfft(windowed)
    return frames


def istft_overlap_add(frames: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Inverse STFT via weighted overlap-add (WOLA)."""
    window = hann_window(frame_size)
    n_frames = frames.shape[0]
    output_len = frame_size + hop_size * (n_frames - 1)
    output = np.zeros(output_len)
    window_sum = np.zeros(output_len)

    for i in range(n_frames):
        start = i * hop_size
        time_segment = np.fft.irfft(frames[i], n=frame_size)
        windowed = time_segment * window
        output[start:start + frame_size] += windowed
        window_sum[start:start + frame_size] += window ** 2

    # normalize by the summed window energy to correct for overlap-add gain.
    # Use a relative threshold to avoid boundary division-by-near-zero spikes
    # that distort peak amplitude and make normalized audio too quiet.
    threshold = 1e-2 * np.max(window_sum) if len(window_sum) > 0 else 1e-8
    nonzero = window_sum > threshold
    output[nonzero] /= window_sum[nonzero]
    output[~nonzero] = 0.0
    return output


def phase_vocoder_stretch(x: np.ndarray, stretch_factor: float,
                           frame_size: int = 2048, hop_analysis: int = 512) -> np.ndarray:
    """
    Time-stretch audio by `stretch_factor` (>1 = slower/longer, <1 = faster/shorter)
    while preserving phase coherence across frames.
    """
    hop_synthesis = int(round(hop_analysis * stretch_factor))
    analysis_frames = stft(x, frame_size, hop_analysis)
    n_frames, n_bins = analysis_frames.shape

    magnitude = np.abs(analysis_frames)
    phase = np.angle(analysis_frames)

    bin_freqs = 2 * np.pi * np.arange(n_bins) / frame_size
    expected_advance = bin_freqs * hop_analysis

    synthesis_phase = np.zeros_like(phase)
    synthesis_phase[0] = phase[0]

    for i in range(1, n_frames):
        delta_phase = phase[i] - phase[i - 1]
        deviation = delta_phase - expected_advance
        deviation_wrapped = np.mod(deviation + np.pi, 2 * np.pi) - np.pi
        true_advance = expected_advance + deviation_wrapped
        synthesis_phase[i] = synthesis_phase[i - 1] + true_advance * (hop_synthesis / hop_analysis)

    new_frames = magnitude * np.exp(1j * synthesis_phase)
    return istft_overlap_add(new_frames, frame_size, hop_synthesis)


def resample_linear(x: np.ndarray, factor: float) -> np.ndarray:
    """Simple linear-interpolation resampler used for pitch shifting and the naive baseline."""
    n_out = int(round(len(x) / factor))
    if n_out < 1:
        return np.zeros(1)
    src_indices = np.arange(n_out) * factor
    src_indices = np.clip(src_indices, 0, len(x) - 1)
    idx_floor = np.floor(src_indices).astype(int)
    idx_ceil = np.clip(idx_floor + 1, 0, len(x) - 1)
    frac = src_indices - idx_floor
    return x[idx_floor] * (1 - frac) + x[idx_ceil] * frac


def pitch_shift_phase_vocoder(x: np.ndarray, semitones: float,
                               frame_size: int = 2048, hop_analysis: int = 512) -> np.ndarray:
    """
    Shift pitch by `semitones` while keeping duration constant.
    """
    if semitones == 0:
        return x.copy()
    pitch_ratio = 2 ** (semitones / 12.0)
    stretched = phase_vocoder_stretch(x, pitch_ratio, frame_size, hop_analysis)
    result = resample_linear(stretched, pitch_ratio)
    if len(result) > len(x):
        result = result[:len(x)]
    else:
        result = np.pad(result, (0, len(x) - len(result)))
    return result


def time_stretch_phase_vocoder(x: np.ndarray, time_factor: float,
                                frame_size: int = 2048, hop_analysis: int = 512) -> np.ndarray:
    """
    Change playback tempo/speed by time_factor (>1 faster, <1 slower) without changing pitch.
    time_factor is a playback speed multiplier (e.g. 1.45 -> 1.45x faster).
    In the phase vocoder, stretch_factor = 1.0 / time_factor.
    """
    if time_factor == 1.0 or time_factor <= 0:
        return x.copy()
    stretch_factor = 1.0 / time_factor
    return phase_vocoder_stretch(x, stretch_factor, frame_size, hop_analysis)


def naive_pitch_shift(x: np.ndarray, semitones: float, time_factor: float = 1.0) -> np.ndarray:
    """
    Baseline: plain resampling (tape-speed method). This is the 'chipmunk effect'.
    """
    pitch_ratio = 2 ** (semitones / 12.0)
    total_rate = pitch_ratio * time_factor
    if np.isclose(total_rate, 1.0):
        return x.copy()
    return resample_linear(x, total_rate)


def compute_spectral_envelope(magnitude: np.ndarray, frame_size: int, lifter_cutoff: int = 30) -> np.ndarray:
    """
    Estimate the smooth spectral envelope (formant structure) of a single magnitude
    spectrum via cepstral liftering: low quefrency cepstral coefficients correspond
    to the slow-varying envelope, while high quefrency ones correspond to fine
    harmonic structure (pitch). Keeping only the low ones and transforming back
    gives a smooth envelope with the pitch harmonics averaged out.
    """
    log_mag = np.log(magnitude + 1e-8)
    cepstrum = np.fft.irfft(log_mag, n=frame_size)
    lifter = np.zeros_like(cepstrum)
    lifter[:lifter_cutoff] = 1.0
    lifter[-(lifter_cutoff - 1):] = 1.0
    liftered = cepstrum * lifter
    envelope_log = np.fft.rfft(liftered, n=frame_size).real
    return np.exp(envelope_log)


def formant_shift(x: np.ndarray, factor: float, frame_size: int = 2048,
                   hop_size: int = 512, lifter_cutoff: int = 30) -> np.ndarray:
    """
    Shift formants (vocal tract resonances) by `factor` WITHOUT changing pitch.

    factor > 1 -> resonances move up (smaller-vocal-tract feel)
    factor < 1 -> resonances move down (larger-vocal-tract feel)
    """
    if factor == 1.0:
        return x.copy()

    frames = stft(x, frame_size, hop_size)
    magnitude = np.abs(frames)
    phase = np.angle(frames)
    n_frames, n_bins = magnitude.shape
    bin_idx = np.arange(n_bins)

    new_magnitude = np.empty_like(magnitude)
    for i in range(n_frames):
        mag = magnitude[i]
        envelope = compute_spectral_envelope(mag, frame_size, lifter_cutoff)
        residual = mag / (envelope + 1e-8)
        warped_idx = bin_idx / factor
        warped_envelope = np.interp(warped_idx, bin_idx, envelope, left=envelope[0], right=envelope[-1])
        new_magnitude[i] = residual * warped_envelope

    new_frames = new_magnitude * np.exp(1j * phase)
    return istft_overlap_add(new_frames, frame_size, hop_size)


# Named voice-character presets: (semitones, time_factor, formant_factor)
VOICE_PRESETS = {
    "male":   {"semitones": -6, "time_factor": 1.0,  "formant_factor": 0.80},
    "female": {"semitones": 8,  "time_factor": 1.0,  "formant_factor": 1.30},
    "kid":    {"semitones": 10, "time_factor": 1.08, "formant_factor": 1.45},
    "robot":  {"semitones": -12, "time_factor": 1.0, "formant_factor": 0.65},
}


def process_audio(x: np.ndarray, semitones: float, time_factor: float,
                   formant_factor: float = 1.0,
                   frame_size: int = 2048, hop_analysis: int = 512):
    """
    Full pipeline used by the API: returns (phase_vocoder_result, naive_result).
    """
    pv = x.copy()
    if time_factor != 1.0:
        pv = time_stretch_phase_vocoder(pv, time_factor, frame_size, hop_analysis)
    if semitones != 0:
        pv = pitch_shift_phase_vocoder(pv, semitones, frame_size, hop_analysis)
    if formant_factor != 1.0:
        pv = formant_shift(pv, formant_factor, frame_size, hop_analysis)

    naive = naive_pitch_shift(x, semitones, time_factor)
    return pv, naive