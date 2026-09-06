import numpy as np
from fastapi.testclient import TestClient

import extractor
from server import app
from test_detector import write_wav
from test_extractor import tone_plus_noise

client = TestClient(app)


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_about_states_what_the_build_can_answer():
    body = client.get("/about").json()
    assert body["trained_on"] == ["suno<=5", "udio<=1.5"]
    assert body["analysis_band_hz"] == [1000, 8000]
    assert body["min_duration_s"] == extractor.MIN_DURATION_S


def test_detect_upload_scores_audio(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, tone_plus_noise(12.0))

    with open(path, "rb") as handle:
        response = client.post("/detect/upload", files={"file": ("tone.wav", handle, "audio/wav")})

    assert response.status_code == 200
    assert response.json()["measured"] is True


def test_undecodable_upload_is_422_not_500(tmp_path):
    path = tmp_path / "junk.wav"
    path.write_bytes(b"nope")

    with open(path, "rb") as handle:
        response = client.post("/detect/upload", files={"file": ("junk.wav", handle, "audio/wav")})

    assert response.status_code == 422


def test_there_is_no_url_fetch_endpoint():
    """An SSRF proxy inside the perimeter is not a feature we are missing."""
    paths = {route.path for route in app.routes}
    assert "/detect" not in paths
    assert paths >= {"/healthz", "/about", "/detect/upload"}
