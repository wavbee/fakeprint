#!/usr/bin/env python3
"""Measure this detector against YOUR audio, including under post-processing.

CI cannot do this: real generator output is licensed and is not vendored here.
So the honest thing is to make the measurement one command, and to publish the
numbers we got rather than ask anyone to take them on trust.

    python scripts/validate.py --real path/to/real/*.wav --ai path/to/ai/*.mp3

Every treatment is applied with ffmpeg to both classes, because a detector that
only degrades on one of them is telling you about the processing, not about the
audio (this is the "production pipeline artifact" trap that Bhattacharjee et
al., TISMIR 2025, warn about).
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import detector  # noqa: E402

# Each is something a real release actually goes through between the DAW and
# the store. `band8k` is the known kill shot, kept in so the limit is visible.
TREATMENTS = {
    "(none)": None,
    "sr22k": ["-ar", "22050"],
    "lp15k": ["-af", "lowpass=f=15000"],
    "mastered": ["-af", "loudnorm=I=-9:TP=-1:LRA=7"],
    "mp3_96k": ["-b:a", "96k"],
    "band8k": ["-af", "aresample=8000,aresample=44100"],
}


def treated(path, args, scratch):
    if args is None:
        return path
    out = os.path.join(scratch, f"t_{os.path.basename(path)}.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, *args, out], check=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", nargs="*", default=[], help="human-made audio (globs ok)")
    parser.add_argument("--ai", nargs="*", default=[], help="generator output (globs ok)")
    args = parser.parse_args()

    classes = {
        "real": [p for g in args.real for p in glob.glob(g)],
        "ai": [p for g in args.ai for p in glob.glob(g)],
    }
    if not any(classes.values()):
        parser.error("give at least one of --real / --ai")

    print(f"model: {detector.MODEL_VERSION}  trained on: {', '.join(detector.TRAINED_ON)}\n")
    header = f"{'class':<6}{'n':>4}" + "".join(f"{name:>11}" for name in TREATMENTS)
    print(header)
    print("-" * len(header))

    with tempfile.TemporaryDirectory() as scratch:
        for label, paths in classes.items():
            if not paths:
                continue
            cells = []
            for name, ffargs in TREATMENTS.items():
                scores = []
                for path in paths:
                    result = detector.analyze(treated(path, ffargs, scratch))
                    if result["measured"]:
                        scores.append(result["ai_probability"])
                # The number that matters is the DECISION rate, not the mean:
                # a mean is dragged around by one saturated score.
                flagged = sum(s >= 0.5 for s in scores)
                cells.append(f"{flagged}/{len(scores)}" if scores else "-")
            print(f"{label:<6}{len(paths):>4}" + "".join(f"{c:>11}" for c in cells))

    print("\nCells are `flagged as AI / measured`. For --real, every non-zero cell")
    print("is a false positive. For --ai, every shortfall is a miss.")


if __name__ == "__main__":
    main()
