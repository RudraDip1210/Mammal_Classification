"""
main.py

The single entry point for the whole pipeline:

    build data splits -> train encoder (Stage 1) -> train classifier
    (Stage 2) -> evaluate -> save results

Run with:
    python main.py

No logic of its own beyond orchestration -- every step is implemented
in its own module and configured through config.py.
"""

import json

import config
from data.splits import main as build_splits
from train_encoder import train_encoder
from train_classifier import train_classifier
from evaluate import evaluate_model


def main():
    print("=== Step 1/4: Building data splits ===")
    build_splits()  # no-op if splits already exist; pass force=True to rebuild

    print("\n=== Step 2/4: Training encoder (Stage 1) ===")
    encoder = train_encoder(config)

    print("\n=== Step 3/4: Training classifier (Stage 2) ===")
    classifier = train_classifier(encoder, config)

    print("\n=== Step 4/4: Evaluating on test set ===")
    metrics = evaluate_model(encoder, classifier, config)

    print("\n=== Done ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
