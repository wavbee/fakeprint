"""Fakeprint extraction: turn audio into the 3585-d vector the detector reads.

The method is Afchar et al., ISMIR 2025 (see NOTICE). Generative audio models
upsample latents back to sample rate with transposed convolutions, and those
layers leave regularly-spaced peaks in the spectrum. The artifact is a property
of the architecture rather than of the training data, which is why a 3585-
parameter logistic regression can read it.

⚠️ The polarity is the opposite of what it looks like. The residue is divided
by its own MAX, so one towering artifact peak crushes every other bin toward
zero: a SPARSE fakeprint means AI, a dense one means real. Adding spectral
spikes to a track makes it look MORE human, not less — verified by injecting
combs at 250/500/1000 Hz into a real master and watching the score stay at 0.

This is a numpy/scipy port of the upstream torchaudio implementation. It exists
so the service does not have to ship torch to run a 15 KB model. It reproduces
upstream's numbers on real generator output (see README).
"""

import numpy as np
from scipy.ndimage import minimum_filter1d

# Every one of these must match the values the bundled model was trained with;
# they are duplicated in models/detector_config.json and asserted on boot.
SAMPLE_RATE = 16_000
N_FFT = 8192
FREQ_MIN = 1000
FREQ_MAX = 8000
HULL_AREA = 10
MAX_DB = 5.0
MIN_DB = -45.0
MAX_DURATION_S = 300

# 16000/8192 Hz per bin => bins 512..4096 inclusive spans 1000-8000 Hz.
FEATURE_DIM = 3585

# Below this the spectrum is too short to average meaningfully and upstream
# says results degrade. The caller records not_run rather than guessing.
MIN_DURATION_S = 10.0


def _stft_power(samples: np.ndarray) -> np.ndarray:
    """Match torchaudio.transforms.Spectrogram(n_fft, power=2) exactly.

    Its defaults are load-bearing and easy to get wrong: win_length = n_fft,
    hop_length = win_length // 2, a PERIODIC Hann window (np.hanning is
    symmetric and would be subtly different), center=True with reflect padding.
    """
    hop = N_FFT // 2
    window = np.hanning(N_FFT + 1)[:-1]
    padded = np.pad(samples, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // hop
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, N_FFT),
        strides=(padded.strides[0] * hop, padded.strides[0]),
        writeable=False,
    ) * window
    return np.abs(np.fft.rfft(frames, n=N_FFT, axis=1).T) ** 2


def band_indices() -> np.ndarray:
    freqs = np.linspace(0, SAMPLE_RATE / 2, num=(N_FFT // 2) + 1)
    return np.where((freqs >= FREQ_MIN) & (freqs <= FREQ_MAX))[0]


def fakeprint(samples: np.ndarray) -> np.ndarray:
    """Mono float samples at SAMPLE_RATE -> the FEATURE_DIM feature vector."""
    samples = np.asarray(samples, dtype=np.float64)[: MAX_DURATION_S * SAMPLE_RATE]
    if samples.size < N_FFT:
        raise ValueError(f"need at least {N_FFT} samples, got {samples.size}")

    spectrum_db = 10.0 * np.log10(np.clip(_stft_power(samples), 1e-10, 1e6))
    band = spectrum_db.mean(axis=1)[band_indices()]

    # The lower envelope, i.e. the local noise floor the peaks stand on.
    hull = np.clip(minimum_filter1d(band, size=HULL_AREA, mode="nearest"), MIN_DB, None)
    residue = np.clip(np.clip(band - hull, 0, None), 0, MAX_DB)
    return (residue / (residue.max() + 1e-6)).astype(np.float32)
