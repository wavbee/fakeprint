# Unlike the Essentia sidecar, this image is architecture-neutral: numpy, scipy
# and onnxruntime all publish arm64 wheels, so it builds and runs natively on
# an Apple-silicon laptop as well as on amd64 in production. No emulation.
FROM python:3.12-slim

# ffmpeg does the decode and the resample to 16 kHz. It is the only system
# dependency; there is no libsndfile, no Essentia, no torch.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/ /app/
COPY models/ /models/
ENV FAKEPRINT_MODEL_DIR=/models
WORKDIR /app
EXPOSE 8000

# The extractor/model compatibility assertion runs on first inference, not at
# import, so a mismatch surfaces as a 500 on the first request rather than a
# container that will not start. Warm it here instead.
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
