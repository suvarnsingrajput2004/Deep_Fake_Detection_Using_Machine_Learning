from __future__ import annotations

import argparse
import subprocess
import sys

from utils.pipeline import evaluate_model, train_model
from utils.preprocessing import build_processed_dataset


def cmd_prepare(args):
    build_processed_dataset(raw_root=args.raw_data, out_root=args.processed_data)
    print("Dataset prepared at:", args.processed_data)


def cmd_train(args):
    metrics = train_model(
        processed_root=args.processed_data,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("Training complete:", metrics)


def cmd_evaluate(args):
    results = evaluate_model(
        processed_root=args.processed_data,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    print("Evaluation complete. Confusion matrix:", results["confusion_matrix"])


def cmd_run(args):
    subprocess.run([sys.executable, "app.py"], check=True)


def cmd_all(args):
    build_processed_dataset(raw_root=args.raw_data, out_root=args.processed_data)
    train_model(
        processed_root=args.processed_data,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    evaluate_model(
        processed_root=args.processed_data,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    subprocess.run([sys.executable, "app.py"], check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Deepfake detection end-to-end CLI")
    parser.add_argument("--raw-data", default="data/raw")
    parser.add_argument("--processed-data", default="data/processed")
    parser.add_argument("--model-path", default="models/deepfake_cnn_model.h5")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)

    sub = parser.add_subparsers(dest="command", required=False)
    sub.add_parser("prepare")
    sub.add_parser("train")
    sub.add_parser("evaluate")
    sub.add_parser("run")
    sub.add_parser("all")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "all"
    {
        "prepare": cmd_prepare,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "run": cmd_run,
        "all": cmd_all,
    }[command](args)
