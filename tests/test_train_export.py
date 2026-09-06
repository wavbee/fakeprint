"""The export contract, which is where retraining silently breaks.

A retrained model that loads but cannot be FED is the failure mode here, and
it is invisible until something round-trips one through the real detector.
skl2onnx names its input `X` and emits a two-output (label, probabilities)
graph; `detector.py` supplies `fakeprint` and reads a single `ai_probability`.
These tests pin the contract so a future export cannot drift off it.
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import extractor  # noqa: E402


def write_model(tmp_path, weights=None, bias=None):
    from train import _write_onnx

    rng = np.random.default_rng(7)
    weights = rng.normal(0, 0.1, (1, extractor.FEATURE_DIM)).astype(np.float32) if weights is None else weights
    bias = np.array([0.25], dtype=np.float32) if bias is None else bias
    path = str(tmp_path / "m.onnx")
    _write_onnx(path, weights, bias)
    return path, weights, bias


def test_input_is_named_fakeprint(tmp_path):
    """The name the detector feeds. skl2onnx would call it `X` and break."""
    path, _, _ = write_model(tmp_path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    assert [i.name for i in session.get_inputs()] == ["fakeprint"]
    assert session.get_inputs()[0].shape[1] == extractor.FEATURE_DIM


def test_there_is_exactly_one_output_and_it_is_the_probability(tmp_path):
    path, _, _ = write_model(tmp_path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    assert [o.name for o in session.get_outputs()] == ["ai_probability"]


def test_the_graph_computes_the_logistic_regression(tmp_path):
    """sigmoid(x @ w.T + b), checked against numpy rather than assumed."""
    path, weights, bias = write_model(tmp_path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(3).random((1, extractor.FEATURE_DIM)).astype(np.float32)

    got = np.ravel(session.run(None, {"fakeprint": x})[0])[0]
    expected = 1.0 / (1.0 + np.exp(-(x @ weights.T + bias)))

    assert got == pytest.approx(float(np.ravel(expected)[0]), abs=1e-5)


def test_a_zero_fakeprint_scores_the_bias(tmp_path):
    """The polarity check: an all-zero feature vector is the bias alone."""
    path, _, bias = write_model(tmp_path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    got = np.ravel(session.run(None, {"fakeprint": np.zeros((1, extractor.FEATURE_DIM), np.float32)})[0])[0]

    assert got == pytest.approx(float(1 / (1 + np.exp(-bias[0]))), abs=1e-6)


def test_it_batches(tmp_path):
    path, _, _ = write_model(tmp_path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    out = session.run(None, {"fakeprint": np.zeros((4, extractor.FEATURE_DIM), np.float32)})[0]

    assert out.shape == (4, 1)
