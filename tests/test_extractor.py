"""No audio is committed to this repo.

Every fixture here is generated from a seeded RNG, so the tests are
deterministic and the repo ships no copyrighted material. That does mean CI
cannot prove "detects Suno" — real generator output is licensed and cannot be
vendored. What CI CAN prove, and what these tests are for, is that the feature
extractor has not drifted: the golden vector below pins the numpy STFT port
against the torchaudio behaviour it was written to reproduce. See
scripts/validate.py for the measurement you run yourself, with your own audio.
"""

import numpy as np
import pytest

import extractor


def tone_plus_noise(seconds=12.0, seed=1234):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * extractor.SAMPLE_RATE)) / extractor.SAMPLE_RATE
    signal = sum(0.2 / n * np.sin(2 * np.pi * 220 * n * t) for n in range(1, 8))
    return (signal + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


def test_feature_vector_has_the_dimension_the_model_expects():
    assert extractor.fakeprint(tone_plus_noise()).shape == (extractor.FEATURE_DIM,)


def test_band_indices_span_1k_to_8k():
    idx = extractor.band_indices()
    assert idx.size == extractor.FEATURE_DIM
    freqs = np.linspace(0, extractor.SAMPLE_RATE / 2, num=(extractor.N_FFT // 2) + 1)
    assert freqs[idx[0]] >= extractor.FREQ_MIN
    assert freqs[idx[-1]] <= extractor.FREQ_MAX


def test_features_are_normalised_into_0_1():
    features = extractor.fakeprint(tone_plus_noise())
    assert features.min() >= 0.0
    assert features.max() == pytest.approx(1.0, abs=1e-3)


def test_is_deterministic():
    audio = tone_plus_noise()
    assert np.array_equal(extractor.fakeprint(audio), extractor.fakeprint(audio))


# The regression guard. These numbers came from the port that was verified
# against real generator output; if a refactor changes them, the extractor no
# longer matches the weights and every score it produces is meaningless.
GOLDEN = {"mean": 0.26389, "sum": 946.04, "nonzero": 3250, "argmax": 49}


def test_golden_feature_vector_pins_the_stft_port():
    features = extractor.fakeprint(tone_plus_noise())
    assert float(features.mean()) == pytest.approx(GOLDEN["mean"], abs=1e-4)
    assert float(features.sum()) == pytest.approx(GOLDEN["sum"], abs=0.05)
    assert int(np.count_nonzero(features)) == GOLDEN["nonzero"]
    assert int(features.argmax()) == GOLDEN["argmax"]


def test_rejects_audio_shorter_than_one_fft_window():
    with pytest.raises(ValueError, match="at least"):
        extractor.fakeprint(np.zeros(1000, dtype=np.float32))


def test_truncates_to_the_duration_cap():
    long_audio = np.tile(tone_plus_noise(1.0), extractor.MAX_DURATION_S + 30)
    capped = extractor.fakeprint(long_audio[: extractor.MAX_DURATION_S * extractor.SAMPLE_RATE])
    assert np.array_equal(extractor.fakeprint(long_audio), capped)
