# fakeprint

A small, honest AI-music detector you can run yourself.

It reads the spectral fingerprint that generative audio models leave behind, and
returns a probability. It is a **signal, not a verdict** — see [What it cannot
tell you](#what-it-cannot-tell-you), which is the most important section here.

Built by [WAVBEE](https://wavbee.com) for our release-QC engine. MIT licensed,
including the bundled model. See [`NOTICE`](NOTICE) — the method and the weights
are other people's work, and it says whose.

## Why it can be this small

The model is a **logistic regression with 3,585 parameters**. It fits in 15 KB.

That works because of a result from Deezer's research team ([Afchar et al.,
ISMIR 2025 best paper](https://arxiv.org/abs/2506.19108)): generative audio
models upsample latents back to sample rate using transposed convolutions, and
those layers leave regularly-spaced peaks in the frequency domain. The artifact
comes from the **architecture**, not from the training data or the weights. You
do not need a big model to see a structural artifact — you need to look in the
right place, which here is the 1–8 kHz band of the time-averaged spectrum.

The consequence worth internalising: the durable asset is the feature extractor
and a labelled corpus. The classifier on top is almost free.

## Use it

```bash
docker build -t fakeprint .
docker run -p 8000:8000 fakeprint

curl -F file=@track.wav http://localhost:8000/detect/upload
```

```json
{
  "measured": true,
  "ai_probability": 0.996,
  "duration_s": 128.4,
  "analysis_band_hz": [1000, 8000],
  "feature_sparsity": 0.887,
  "model_version": "fakeprint-1.0.0+lofcz-suno5",
  "trained_on": ["suno<=5", "udio<=1.5"]
}
```

`GET /about` returns the same limits without sending audio. There is
deliberately **no URL-fetch endpoint**: a service inside your perimeter that
fetches arbitrary URLs is an SSRF proxy. Callers send bytes.

A file too short to measure comes back `{"measured": false, "reason":
"too_short"}` rather than a confident score. Nothing here reports a clean result
it did not measure.

## What we measured

Four genuine Suno tracks and one real commercial master, each put through the
treatments a release actually survives between the DAW and the store.
Reproduce with your own audio:

```bash
python scripts/validate.py --real 'masters/*.wav' --ai 'generated/*.mp3'
```

| treatment | AI flagged | real flagged (false positives) |
|---|---|---|
| none | **4/4** | 0/1 |
| resample to 22.05 kHz | **4/4** | 0/1 |
| lowpass 15 kHz | **4/4** | 0/1 |
| loudness mastering (`I=-9 TP=-1 LRA=7`) | **4/4** | 0/1 |
| MP3 96 kbps | **4/4** | 0/1 |
| band-limit to 8 kHz | 3/4 | 0/1 |

Mastering, lossy encoding and sample-rate conversion do not defeat it. The one
manipulation that does is **band-limiting below ~8 kHz**, which removes half the
evidence before the detector sees it. That limit is structural, not a bug, and
`analysis_band_hz` is in every response so a caller can reason about it.

Separately: a full **Encodec round-trip of a real master still scores 0.0%**.
The artifact is generator-specific. "It passed through a neural codec" is not
the signal, so the detector does not fire on one.

## Retraining it

The bundled weights cover Suno ≤ 5 and Udio ≤ 1.5. Newer generators change
their upsamplers, so the classifier goes stale while the **method does not** —
the artifact is architectural. That asymmetry is the whole design: the feature
extractor is the durable asset, and the classifier is 3,585 parameters.

```bash
pip install scikit-learn onnx          # not runtime dependencies
python scripts/train.py --real 'masters/*.wav' --ai 'generated/*.mp3' \
                        --out models/mine
```

It prints a **held-out** evaluation and **refuses to write a model that fails
the floors** — 90% accuracy, and a false-positive rate at or under 2%, which is
stricter because a false positive tells a real musician they faked their own
record. `--force` overrides. The written config records the corpus size and the
held-out numbers, so a model always carries its own provenance.

⚠️ It cannot detect a mislabelled corpus. Two arbitrary halves of the same
population score ~0.4 accuracy and are correctly refused, but a corpus where
"real" quietly contains generated tracks will train happily and evaluate well.
The labels are yours to get right.

Bump `MODEL_VERSION` in `app/detector.py` before serving a retrained model. A
score from one build is not comparable to a score from another, and any
threshold tuned against one does not transfer.

## What it cannot tell you

**These numbers are not a generalisation estimate.** The four Suno tracks come
from the SONICS dataset, which the bundled model was trained on. That makes the
measurement above a valid test of two things — that the feature extractor is
correct, and that the feature survives post-processing (a fragile feature would
degrade even on a memorised track, and it did not) — and an invalid estimate of
accuracy on unseen audio.

Concretely, all of the following are **unmeasured**: Suno 6 and later, Udio 2
and later, every other generator, and the false-positive rate across a broad
catalogue of real music. One commercial master is not a false-positive rate.

**A low score is not proof a human made it.** It means these generators were not
detected. Absence of evidence is the whole caveat.

**It will need retraining.** The bundled weights cover Suno ≤ 5 and Udio ≤ 1.5.
Newer generators change their upsamplers. Because the extractor is here and the
classifier is 3,585 parameters, retraining on a fresh corpus is a laptop-scale
job — that is the point of the split.

**It has no idea about provenance.** Watermarks (SynthID, AudioSeal, Suno's own
fingerprinting) answer a different and stronger question, and none of them
expose a public detector we can call. Absence of a watermark proves nothing
either.

### Please don't use this to accuse anyone

A false positive here says a real musician faked their own record. Treat the
output as one input to a question — *"was this AI-generated?"* — and not as a
finding. That is how we use it: it opens a disclosure question, it never rejects
a release on its own.

## Layout

```
app/extractor.py   audio -> 3585-d fakeprint (numpy/scipy port of the
                   torchaudio original, so no torch in the image)
app/detector.py    ffmpeg decode + ONNX inference
app/server.py      FastAPI, multipart only
models/            the bundled MIT model (15 KB)
scripts/validate.py  the measurement harness above
```

No audio is committed. Tests generate their own from a seeded RNG, so CI proves
the extractor has not drifted — there is a golden feature vector pinning the
STFT port — but cannot prove detection. That measurement is yours to run.

```bash
pip install -r requirements.txt pytest httpx && python -m pytest tests -q
```

## Licence

MIT, for the code and the bundled model both. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
