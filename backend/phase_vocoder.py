"""
Phase Vocoder core DSP.

Implements:
  - STFT / inverse STFT (overlap-add)
  - Phase-vocoder time-stretching (with phase unwrapping + propagation,
    so stretched audio stays phase-coherent instead of sounding "robotic")
  - Pitch shifting = time-stretch by ratio, then resample back to original
    duration (this decouples pitch from tempo)
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

    # normalize by the summed window energy to correct for overlap-add gain
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    return output


def phase_vocoder_stretch(x: np.ndarray, stretch_factor: float,
                           frame_size: int = 2048, hop_analysis: int = 512) -> np.ndarray:
    """
    Time-stretch audio by `stretch_factor` (>1 = slower/longer, <1 = faster/shorter)
    while preserving phase coherence across frames.

    This is the heart of the project: naive time-stretching just changes the
    hop size on both analysis and synthesis equally, which scrambles inter-frame
    phase relationships and causes "phasiness" / robotic artifacts. Instead we:
      1. Analyze at a fixed hop (hop_analysis)
      2. Compute the true instantaneous frequency per bin via phase unwrapping
      3. Re-synthesize at a different hop (hop_synthesis), advancing phase
         according to the tracked instantaneous frequency rather than the
         raw (wrapped) phase difference
    """
    hop_synthesis = int(round(hop_analysis * stretch_factor))
    analysis_frames = stft(x, frame_size, hop_analysis)
    n_frames, n_bins = analysis_frames.shape

    magnitude = np.abs(analysis_frames)
    phase = np.angle(analysis_frames)

    # expected phase advance per hop for each bin (bin center frequency * hop, in radians)
    bin_freqs = 2 * np.pi * np.arange(n_bins) / frame_size
    expected_advance = bin_freqs * hop_analysis

    synthesis_phase = np.zeros_like(phase)
    synthesis_phase[0] = phase[0]

    for i in range(1, n_frames):
        # measured phase difference between consecutive analysis frames
        delta_phase = phase[i] - phase[i - 1]
        # remove expected linear phase advance, leaving only the deviation
        deviation = delta_phase - expected_advance
        # wrap deviation into [-pi, pi] -- this is the phase unwrapping step
        deviation_wrapped = np.mod(deviation + np.pi, 2 * np.pi) - np.pi
        # true instantaneous frequency for this frame/bin
        true_advance = expected_advance + deviation_wrapped
        # accumulate phase using the synthesis hop instead of the analysis hop,
        # scaled proportionally -- this is what keeps harmonics phase-locked
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

    Trick: stretch time by the pitch ratio (changes pitch AND duration together,
    but phase-coherently), then resample back to the original duration (which
    changes pitch again, in the opposite direction, cancelling the duration change
    but compounding the pitch change). Net effect: pitch changes, duration doesn't.
    """
    if semitones == 0:
        return x.copy()
    pitch_ratio = 2 ** (semitones / 12.0)
    stretched = phase_vocoder_stretch(x, pitch_ratio, frame_size, hop_analysis)
    result = resample_linear(stretched, 1.0 / pitch_ratio)
    # trim/pad to match original length so playback duration is unchanged
    if len(result) > len(x):
        result = result[:len(x)]
    else:
        result = np.pad(result, (0, len(x) - len(result)))
    return result


def time_stretch_phase_vocoder(x: np.ndarray, time_factor: float,
                                frame_size: int = 2048, hop_analysis: int = 512) -> np.ndarray:
    """Change tempo by time_factor (>1 slower, <1 faster) without changing pitch."""
    if time_factor == 1.0:
        return x.copy()
    return phase_vocoder_stretch(x, time_factor, frame_size, hop_analysis)


def naive_pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """
    Baseline: plain resampling. This is the 'chipmunk effect' -- pitch and
    duration change together because there's no phase correction, just a
    change in playback rate. Included for the required A/B comparison.
    """
    if semitones == 0:
        return x.copy()
    pitch_ratio = 2 ** (semitones / 12.0)
    return resample_linear(x, pitch_ratio)


def process_audio(x: np.ndarray, semitones: float, time_factor: float,
                   frame_size: int = 2048, hop_analysis: int = 512):
    """
    Full pipeline used by the API: returns (phase_vocoder_result, naive_result).
    Combined pitch + time-scale modification via the phase vocoder, plus the
    naive resampling baseline for comparison (naive only supports pitch,
    since naive time-stretch-without-pitch-change isn't meaningfully definable).
    """
    pv = x.copy()
    if time_factor != 1.0:
        pv = time_stretch_phase_vocoder(pv, time_factor, frame_size, hop_analysis)
    if semitones != 0:
        pv = pitch_shift_phase_vocoder(pv, semitones, frame_size, hop_analysis)

    naive = naive_pitch_shift(x, semitones)
    return pv, naive
