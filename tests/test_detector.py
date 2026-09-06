import os

import numpy as np
import pytest

import detector
import extractor
from test_extractor import tone_plus_noise


def write_wav(path, samples, sample_rate=extractor.SAMPLE_RATE):
    import struct
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        pcm = np.clip(samples, -1, 1) * 32767
        handle.writeframes(struct.pack(f"<{pcm.size}h", *pcm.astype("<i2")))


def test_model_and_extractor_agree_on_boot():
    """A silent mismatch produces confident nonsense, so it must raise."""
    detector._session = None
    original = extractor.N_FFT
    extractor.N_FFT = 4096
    try:
        with pytest.raises(RuntimeError, match="mismatch on n_fft"):
            detector._load()
    finally:
        extractor.N_FFT = original
        detector._session = None


def test_scores_a_real_recording_as_not_ai(tmp_path):
    """Harmonic tone plus noise is dense in the residue band, i.e. human-like.

    This is a sanity floor, not an accuracy claim — see the module docstring in
    test_extractor.py.
    """
    path = tmp_path / "tone.wav"
    write_wav(path, tone_plus_noise(12.0))
    result = detector.analyze(str(path))

    assert result["measured"] is True
    assert result["ai_probability"] < 0.5


def test_reports_too_short_rather_than_guessing(tmp_path):
    path = tmp_path / "short.wav"
    write_wav(path, tone_plus_noise(3.0))
    result = detector.analyze(str(path))

    assert result["measured"] is False
    assert result["reason"] == "too_short"
    assert "ai_probability" not in result


def test_every_response_names_the_build_and_its_limits(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, tone_plus_noise(12.0))

    for result in (detector.analyze(str(path)), detector.analyze(str(tmp_path / "tone.wav"))):
        assert result["model_version"] == detector.MODEL_VERSION
        assert result["trained_on"] == ["suno<=5", "udio<=1.5"]


def test_decode_failure_raises_rather_than_returning_a_score(tmp_path):
    path = tmp_path / "not-audio.wav"
    path.write_bytes(b"this is not audio")

    with pytest.raises(RuntimeError, match="decode failed"):
        detector.analyze(str(path))


def test_decode_resamples_to_the_models_rate(tmp_path):
    """12 s written at 44.1 kHz must come back as 12 s of 16 kHz samples."""
    path = tmp_path / "hi.wav"
    at_44k = np.tile(tone_plus_noise(1.0), 34)[: 12 * 44100]
    write_wav(path, at_44k, sample_rate=44100)

    samples = detector.decode(str(path))

    assert samples.dtype == np.float32
    assert abs(len(samples) / extractor.SAMPLE_RATE - 12.0) < 0.1
