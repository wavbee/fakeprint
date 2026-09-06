"""ONNX inference plus the decode in front of it.

The model is sigmoid(w.x + b) with b = +4.98, so an all-zero fakeprint scores
0.993 and an all-ones fakeprint scores 0.0. See extractor.py on why that is the
right way round.
"""

import json
import os
import signal
import subprocess
from typing import Any

import numpy as np
import onnxruntime as ort

import extractor

MODEL_DIR = os.environ.get("FAKEPRINT_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "ai_music_detector.onnx")
CONFIG_PATH = os.path.join(MODEL_DIR, "detector_config.json")

# Bumped whenever the weights or the extractor change. The caller stores this
# against every finding: a score from one build is not comparable to a score
# from another, and a threshold tuned against one does not transfer.
MODEL_VERSION = "fakeprint-1.0.0+lofcz-suno5"

# What the bundled weights were actually trained to recognise. Stated in every
# response because it is the honest limit of the answer: silence here is not
# evidence of a human, only evidence that these generators were not detected.
TRAINED_ON = ["suno<=5", "udio<=1.5"]

DECODE_TIMEOUT_S = 120

_session: ort.InferenceSession | None = None


def _load() -> ort.InferenceSession:
    global _session
    if _session is None:
        with open(CONFIG_PATH) as handle:
            config = json.load(handle)
        # A mismatch here means the extractor and the weights disagree about
        # what a feature vector is, which produces confident nonsense rather
        # than an error. Fail on boot instead.
        expected = config["preprocessing"]
        for key, ours in [
            ("sample_rate", extractor.SAMPLE_RATE), ("n_fft", extractor.N_FFT),
            ("freq_min", extractor.FREQ_MIN), ("freq_max", extractor.FREQ_MAX),
            ("hull_area", extractor.HULL_AREA), ("max_db", extractor.MAX_DB),
            ("min_db", extractor.MIN_DB),
        ]:
            if expected[key] != ours:
                raise RuntimeError(f"extractor/model mismatch on {key}: {ours} != {expected[key]}")
        if config["input_features"] != extractor.FEATURE_DIM:
            raise RuntimeError("extractor/model mismatch on feature dimension")
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def decode(path: str) -> np.ndarray:
    """Mono, 16 kHz, float32 — via ffmpeg, which resamples better than we would.

    Killed by process GROUP on timeout: ffmpeg spawns children, and killing only
    the parent leaves them holding the pipe open forever.
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1",
         "-ar", str(extractor.SAMPLE_RATE), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=DECODE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        raise RuntimeError("decode timed out")
    if proc.returncode != 0:
        raise RuntimeError(f"decode failed: {err.decode('utf-8', 'replace')[:200]}")
    return np.frombuffer(out, dtype=np.float32)


def analyze(path: str) -> dict[str, Any]:
    samples = decode(path)
    duration = len(samples) / extractor.SAMPLE_RATE

    # Too short to measure is reported as such, never as a clean result.
    if duration < extractor.MIN_DURATION_S:
        return {
            "measured": False,
            "reason": "too_short",
            "duration_s": round(duration, 3),
            "min_duration_s": extractor.MIN_DURATION_S,
            "model_version": MODEL_VERSION,
            "trained_on": TRAINED_ON,
        }

    features = extractor.fakeprint(samples)
    probability = float(np.ravel(_load().run(None, {"fakeprint": features.reshape(1, -1)})[0])[0])
    return {
        "measured": True,
        "ai_probability": round(probability, 6),
        "duration_s": round(duration, 3),
        # Reported because it is the one manipulation that reliably defeats
        # this method: the features live in 1-8 kHz, so audio band-limited
        # below ~8 kHz has had half the evidence removed before we ever see it.
        "analysis_band_hz": [extractor.FREQ_MIN, extractor.FREQ_MAX],
        "feature_sparsity": round(float(1.0 - features.mean()), 6),
        "model_version": MODEL_VERSION,
        "trained_on": TRAINED_ON,
    }
