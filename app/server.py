"""Stateless HTTP surface. Receives the audio, scores it, forgets it.

Nothing is persisted between requests.

**There is deliberately no URL-fetch endpoint.** A service inside the network
perimeter that fetches whatever URL it is handed is an SSRF proxy, and the only
URLs available to the caller come from tenant-submitted payloads. Callers send
bytes. (The sibling Essentia sidecar shipped one of these and it was removed;
this one never had it.)
"""

import os
import tempfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

import detector
import extractor

MAX_BYTES = 500 * 1024 * 1024

app = FastAPI(title="wavbee fakeprint sidecar", version="1.0.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/about")
def about() -> dict[str, Any]:
    """What this build is and what it can honestly answer.

    Exposed so a caller can record the limits alongside the score instead of
    hardcoding assumptions about them.
    """
    return {
        "model_version": detector.MODEL_VERSION,
        "trained_on": detector.TRAINED_ON,
        "analysis_band_hz": [extractor.FREQ_MIN, extractor.FREQ_MAX],
        "min_duration_s": extractor.MIN_DURATION_S,
        "method": "Afchar et al., A Fourier Explanation of AI-music Artifacts, ISMIR 2025",
    }


@app.post("/detect/upload")
async def detect_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "input")
        written = 0
        with open(path, "wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="audio exceeds size cap")
                handle.write(chunk)
        try:
            return detector.analyze(path)
        except RuntimeError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
