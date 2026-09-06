#!/usr/bin/env python3
"""Retrain the detector on your own corpus.

The bundled model covers Suno <= 5 and Udio <= 1.5. Newer generators change
their upsamplers, so the classifier goes stale while the METHOD does not — the
artifact is architectural. That asymmetry is the point of this script: the
feature extractor is the durable asset, the classifier is 3,585 parameters and
retrains on a laptop.

    python scripts/train.py --real 'masters/*.wav' --ai 'generated/*.mp3' \
                            --out models/mine

Writes <out>.onnx, <out>.npz and <out>_config.json, and prints a HELD-OUT
evaluation. It refuses to write a model that does not beat the floor, because
a detector nobody evaluated is worse than none: it would be trusted.

Requires: pip install scikit-learn onnx  (not runtime dependencies)
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import detector  # noqa: E402
import extractor  # noqa: E402

# Below this a "model" is a coin flip with extra steps. Deliberately generous:
# the point is to refuse the obviously broken, not to certify the good.
MIN_HELD_OUT_ACCURACY = 0.90
# A false positive tells a real musician they faked their own record, so this
# floor is stricter than the accuracy one.
MAX_FALSE_POSITIVE_RATE = 0.02
# Fewer than this per class and a held-out split measures nothing.
MIN_PER_CLASS = 20


def features_for(paths, label, quiet=False):
    """Extract fakeprints, skipping anything unmeasurable rather than zero-filling."""
    rows, skipped = [], []
    for path in paths:
        try:
            samples = detector.decode(path)
            if len(samples) / extractor.SAMPLE_RATE < extractor.MIN_DURATION_S:
                skipped.append((path, "too short"))
                continue
            rows.append(extractor.fakeprint(samples))
        except Exception as error:  # noqa: BLE001 - a bad file must not stop a corpus
            skipped.append((path, str(error)[:60]))
        if not quiet and len(rows) % 25 == 0 and rows:
            print(f"  ...{len(rows)} extracted", flush=True)
    for path, why in skipped:
        print(f"  SKIPPED {os.path.basename(path)}: {why}", file=sys.stderr)
    return np.asarray(rows, dtype=np.float32), np.full(len(rows), label, dtype=np.int64)


def _write_onnx(path, weights, bias):
    """Build the graph by hand: sigmoid(fakeprint @ W + b).

    NOT via skl2onnx. Its converter names the input `X` and emits a two-output
    (label, probabilities) graph, so a model exported that way loads but cannot
    be fed — `detector.py` supplies `fakeprint` and reads a single
    `ai_probability`. Caught by round-tripping an exported model through the
    real detector, which is the only test that would have found it.

    Logistic regression is three nodes. Writing them directly guarantees the
    retrained model is contract-identical to the bundled one.
    """
    from onnx import TensorProto, helper, numpy_helper, save

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["fakeprint", "W"], ["z"]),
            helper.make_node("Add", ["z", "B"], ["logit"]),
            helper.make_node("Sigmoid", ["logit"], ["ai_probability"]),
        ],
        name="fakeprint_logreg",
        inputs=[helper.make_tensor_value_info(
            "fakeprint", TensorProto.FLOAT, ["batch_size", extractor.FEATURE_DIM])],
        outputs=[helper.make_tensor_value_info(
            "ai_probability", TensorProto.FLOAT, ["batch_size", 1])],
        initializer=[
            numpy_helper.from_array(weights.T.copy().astype(np.float32), "W"),
            numpy_helper.from_array(bias.astype(np.float32), "B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    save(model, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", nargs="+", required=True, help="human-made audio (globs ok)")
    parser.add_argument("--ai", nargs="+", required=True, help="generator output (globs ok)")
    parser.add_argument("--out", default="models/retrained")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="write the model even if it fails the quality floors")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    real_paths = sorted({p for g in args.real for p in glob.glob(g)})
    ai_paths = sorted({p for g in args.ai for p in glob.glob(g)})
    print(f"corpus: {len(real_paths)} real, {len(ai_paths)} ai")
    if min(len(real_paths), len(ai_paths)) < MIN_PER_CLASS:
        print(f"\nWARNING: fewer than {MIN_PER_CLASS} files in a class. A held-out\n"
              "split on this measures almost nothing — treat any number below as\n"
              "an indication that the pipeline ran, not as an evaluation.\n", file=sys.stderr)

    print("extracting real...")
    x_real, y_real = features_for(real_paths, 0)
    print("extracting ai...")
    x_ai, y_ai = features_for(ai_paths, 1)
    if len(x_real) == 0 or len(x_ai) == 0:
        sys.exit("both classes need at least one measurable file")

    features = np.vstack([x_real, x_ai])
    labels = np.concatenate([y_real, y_ai])
    print(f"usable: {Counter(labels.tolist())}  (0=real, 1=ai)")

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    true_pos = int(((predicted == 1) & (y_test == 1)).sum())
    false_pos = int(((predicted == 1) & (y_test == 0)).sum())
    true_neg = int(((predicted == 0) & (y_test == 0)).sum())
    false_neg = int(((predicted == 0) & (y_test == 1)).sum())
    accuracy = (true_pos + true_neg) / max(1, len(y_test))
    fpr = false_pos / max(1, true_neg + false_pos)
    recall = true_pos / max(1, true_pos + false_neg)

    print(f"\nheld-out ({len(y_test)} files)")
    print(f"  accuracy            {accuracy:.4f}")
    print(f"  recall (AI found)   {recall:.4f}   [{true_pos}/{true_pos + false_neg}]")
    print(f"  FALSE POSITIVE rate {fpr:.4f}   [{false_pos}/{true_neg + false_pos}]  <- the one that matters")

    failures = []
    if accuracy < MIN_HELD_OUT_ACCURACY:
        failures.append(f"accuracy {accuracy:.4f} < {MIN_HELD_OUT_ACCURACY}")
    if fpr > MAX_FALSE_POSITIVE_RATE:
        failures.append(f"false-positive rate {fpr:.4f} > {MAX_FALSE_POSITIVE_RATE}")
    if failures and not args.force:
        sys.exit("\nNOT WRITTEN — " + "; ".join(failures) +
                 "\nA detector nobody evaluated is worse than none, because it gets trusted."
                 "\nPass --force to write it anyway.")

    _export(model, args.out, len(real_paths), len(ai_paths), accuracy, fpr, recall)


def _export(model, out, n_real, n_ai, accuracy, fpr, recall):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    weights = model.coef_.astype(np.float32)
    bias = model.intercept_.astype(np.float32)
    np.savez(f"{out}.npz", weights=weights, bias=bias, classes=model.classes_)

    _write_onnx(f"{out}.onnx", weights, bias)

    # The preprocessing block MUST match extractor.py — detector.py asserts it
    # on boot, because a mismatch produces confident nonsense rather than an
    # error.
    config = {
        "model_type": "logistic_regression",
        "task": "audio-classification",
        "labels": ["real", "ai_generated"],
        "preprocessing": {
            "sample_rate": extractor.SAMPLE_RATE, "n_fft": extractor.N_FFT,
            "freq_min": extractor.FREQ_MIN, "freq_max": extractor.FREQ_MAX,
            "hull_area": extractor.HULL_AREA, "max_db": extractor.MAX_DB,
            "min_db": extractor.MIN_DB,
        },
        "input_features": extractor.FEATURE_DIM,
        "threshold": 0.5,
        "provenance": {
            "corpus": {"real_files": n_real, "ai_files": n_ai},
            "held_out": {"accuracy": round(accuracy, 4),
                         "false_positive_rate": round(fpr, 4),
                         "recall": round(recall, 4)},
        },
    }
    with open(f"{out}_config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    print(f"\nwrote {out}.onnx, {out}.npz, {out}_config.json")
    print("The config records the corpus size and the held-out numbers — a score")
    print("from one build is not comparable to a score from another, so bump")
    print("MODEL_VERSION in app/detector.py before serving this.")


if __name__ == "__main__":
    main()
