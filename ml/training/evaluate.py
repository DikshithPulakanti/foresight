"""
Evaluate a trained SpendingCategoryBERT model against the test set, track
experiments in MLflow, and generate a formatted evaluation report.

Loads the fine-tuned model from disk, runs batch inference on every test
example, computes per-class precision / recall / F1 plus a confusion matrix,
logs everything to MLflow, and prints a recruiter-ready summary.

Usage:
    python -m ml.training.evaluate                          # full eval + MLflow
    python -m ml.training.evaluate --no-mlflow              # skip MLflow logging
    python -m ml.training.evaluate --inference-test          # quick 20-example check
    python -m ml.training.evaluate --model-path ml/models/spending_bert_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix as sk_confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
except ImportError:
    accuracy_score = None  # type: ignore[assignment]

try:
    import mlflow
    import mlflow.pytorch  # noqa: F401 – registers the pytorch flavour
except ImportError:
    mlflow = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Label maps (must match train_bert.py / generate_dataset.py) ──────────────

LABEL2ID: dict[str, int] = {
    "grocery": 0,
    "restaurant": 1,
    "transport": 2,
    "shopping": 3,
    "entertainment": 4,
    "utilities": 5,
    "healthcare": 6,
    "general": 7,
}

ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    """Paths and parameters for an evaluation run."""

    model_path: str = "ml/models/spending_bert"
    test_path: str = "ml/datasets/transactions_test.jsonl"
    dataset_path: str = "ml/datasets/transactions_50k.jsonl"
    results_path: str = "ml/models/eval_results.json"
    mlflow_experiment: str = "SpendingCategoryBERT"
    mlflow_run_name: str = "evaluation"
    max_length: int = 64
    batch_size: int = 64


# ── Dependency check ─────────────────────────────────────────────────────────


def _check_deps(*, need_mlflow: bool = False) -> None:
    missing: list[str] = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if load_dataset is None:
        missing.append("datasets")
    if AutoTokenizer is None:
        missing.append("transformers")
    if accuracy_score is None:
        missing.append("scikit-learn")
    if need_mlflow and mlflow is None:
        missing.append("mlflow")
    if missing:
        raise RuntimeError(
            f"Missing required packages: {', '.join(missing)}. "
            "Install with: pip install -r ml/training/requirements_ml.txt"
        )


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Batch inference ──────────────────────────────────────────────────────────


def predict_batch(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    config: EvalConfig,
    device: str,
) -> tuple[list[int], list[float]]:
    """
    Run inference on a batch of transaction strings.

    Returns:
        predicted_labels: integer class labels
        confidence_scores: softmax probability of the predicted class
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        logits = model(**encodings).logits

    probs = F.softmax(logits, dim=-1)
    confidence, predicted = probs.max(dim=-1)

    return predicted.cpu().tolist(), confidence.cpu().tolist()


# ── Full evaluation ──────────────────────────────────────────────────────────


def evaluate_model(config: EvalConfig) -> dict[str, Any]:
    """
    Load a trained model and run it against the test set.

    Returns a dict with: accuracy, f1_weighted, f1_macro, precision, recall,
    per_class_metrics, confusion_matrix, confidence_stats,
    low_confidence_examples, and test_set_size.
    """
    _check_deps()

    device = _detect_device()
    log.info("Evaluation device: %s", device)

    # ── Load model ───────────────────────────────────────────────────────
    if not os.path.isdir(config.model_path):
        raise FileNotFoundError(
            f"Model not found at {config.model_path}. "
            "Train one with: python -m ml.training.train_bert"
        )

    log.info("Loading model from %s", config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_path)
    model.to(device)
    model.eval()

    # ── Load test data ───────────────────────────────────────────────────
    if not os.path.isfile(config.test_path):
        raise FileNotFoundError(
            f"Test set not found at {config.test_path}. "
            "Generate it with: python -m ml.training.generate_dataset"
        )

    log.info("Loading test data from %s", config.test_path)
    ds = load_dataset("json", data_files=config.test_path, split="train")
    texts: list[str] = ds["text"]
    true_labels: list[int] = ds["label"]
    categories: list[str] = ds["category"]

    log.info("Test set: %d examples", len(texts))

    # ── Batch inference ──────────────────────────────────────────────────
    all_preds: list[int] = []
    all_confs: list[float] = []

    num_batches = math.ceil(len(texts) / config.batch_size)
    for i in range(num_batches):
        start = i * config.batch_size
        end = min(start + config.batch_size, len(texts))
        batch_texts = texts[start:end]

        preds, confs = predict_batch(batch_texts, model, tokenizer, config, device)
        all_preds.extend(preds)
        all_confs.extend(confs)

    y_true = np.array(true_labels)
    y_pred = np.array(all_preds)
    confs_arr = np.array(all_confs)

    # ── Overall metrics ──────────────────────────────────────────────────
    acc = float(accuracy_score(y_true, y_pred))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    # ── Per-class metrics ────────────────────────────────────────────────
    per_class: dict[str, dict[str, float]] = {}
    for label_id in range(len(LABEL2ID)):
        cat = ID2LABEL[label_id]
        mask_true = y_true == label_id
        mask_pred = y_pred == label_id

        tp = int(np.sum(mask_true & mask_pred))
        support = int(np.sum(mask_true))
        predicted_count = int(np.sum(mask_pred))

        cat_prec = tp / predicted_count if predicted_count > 0 else 0.0
        cat_rec = tp / support if support > 0 else 0.0
        cat_f1 = (
            2 * cat_prec * cat_rec / (cat_prec + cat_rec)
            if (cat_prec + cat_rec) > 0
            else 0.0
        )

        per_class[cat] = {
            "precision": round(cat_prec, 4),
            "recall": round(cat_rec, 4),
            "f1": round(cat_f1, 4),
            "support": support,
        }

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL2ID))))
    cm_list = cm.tolist()

    # ── Confidence statistics ────────────────────────────────────────────
    conf_stats = {
        "mean": round(float(confs_arr.mean()), 4),
        "min": round(float(confs_arr.min()), 4),
        "max": round(float(confs_arr.max()), 4),
        "std": round(float(confs_arr.std()), 4),
    }

    # ── Low-confidence examples (< 0.6) ─────────────────────────────────
    low_conf_examples: list[dict[str, Any]] = []
    for idx in range(len(texts)):
        if all_confs[idx] < 0.6:
            low_conf_examples.append({
                "text": texts[idx],
                "true_category": categories[idx],
                "predicted_category": ID2LABEL.get(all_preds[idx], "unknown"),
                "confidence": round(all_confs[idx], 4),
                "correct": true_labels[idx] == all_preds[idx],
            })

    low_conf_examples.sort(key=lambda x: x["confidence"])

    metrics: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "per_class_metrics": per_class,
        "confusion_matrix": cm_list,
        "confusion_matrix_labels": [ID2LABEL[i] for i in range(len(LABEL2ID))],
        "confidence_stats": conf_stats,
        "low_confidence_examples": low_conf_examples,
        "test_set_size": len(texts),
        "model_path": config.model_path,
    }

    log.info("Evaluation complete — accuracy=%.4f  f1=%.4f", acc, f1_w)
    return metrics


# ── MLflow tracking ──────────────────────────────────────────────────────────


def log_to_mlflow(
    metrics: dict[str, Any],
    config: EvalConfig,
    model_path: str,
) -> str:
    """
    Log evaluation results to an MLflow experiment.

    Logs scalar metrics, per-class F1, parameters, and result artifacts.
    Returns the MLflow run ID.
    """
    if mlflow is None:
        raise RuntimeError(
            "mlflow is not installed. Install with: pip install mlflow>=2.9.0"
        )

    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run(run_name=config.mlflow_run_name):
        # Scalar metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
        mlflow.log_metric("f1_macro", metrics["f1_macro"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])

        # Per-class F1
        for category, class_metrics in metrics["per_class_metrics"].items():
            mlflow.log_metric(f"f1_{category}", class_metrics["f1"])
            mlflow.log_metric(f"precision_{category}", class_metrics["precision"])
            mlflow.log_metric(f"recall_{category}", class_metrics["recall"])

        # Confidence stats
        for stat_name, stat_val in metrics["confidence_stats"].items():
            mlflow.log_metric(f"confidence_{stat_name}", stat_val)

        # Parameters
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("test_set_size", metrics["test_set_size"])
        mlflow.log_param("num_categories", len(LABEL2ID))
        mlflow.log_param("low_confidence_count", len(metrics["low_confidence_examples"]))

        # Artifacts
        os.makedirs(os.path.dirname(config.results_path) or ".", exist_ok=True)

        with open(config.results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(config.results_path)

        cm_path = str(Path(config.results_path).parent / "confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump({
                "labels": metrics["confusion_matrix_labels"],
                "matrix": metrics["confusion_matrix"],
            }, f, indent=2)
        mlflow.log_artifact(cm_path)

        run_id = mlflow.active_run().info.run_id
        log.info("MLflow run logged: %s", run_id)
        return run_id


# ── Pretty-print report ─────────────────────────────────────────────────────


def print_evaluation_report(metrics: dict[str, Any]) -> None:
    """Print a formatted evaluation report to stdout."""

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║     SpendingCategoryBERT — Evaluation        ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    print("  Overall Metrics:")
    print(f"    Accuracy:   {metrics['accuracy']:.1%}")
    print(f"    F1 Score:   {metrics['f1_weighted']:.4f} (weighted)")
    print(f"    F1 Macro:   {metrics['f1_macro']:.4f}")
    print(f"    Precision:  {metrics['precision']:.4f}")
    print(f"    Recall:     {metrics['recall']:.4f}")
    print(f"    Test size:  {metrics['test_set_size']} examples")
    print()

    print("  Per-Category Performance:")
    print(f"    {'Category':<16} {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>6}")
    print(f"    {'─' * 46}")
    for cat in LABEL2ID:
        m = metrics["per_class_metrics"].get(cat, {})
        p = m.get("precision", 0)
        r = m.get("recall", 0)
        f = m.get("f1", 0)
        n = m.get("support", 0)
        print(f"    {cat:<16} {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {n:>5d}")
    print()

    cs = metrics.get("confidence_stats", {})
    print("  Confidence Stats:")
    print(f"    Mean: {cs.get('mean', 0):.3f}  "
          f"Min: {cs.get('min', 0):.3f}  "
          f"Max: {cs.get('max', 0):.3f}  "
          f"Std: {cs.get('std', 0):.3f}")
    print()

    low = metrics.get("low_confidence_examples", [])
    if low:
        shown = low[:10]
        print(f"  Low Confidence Examples (< 0.6): {len(low)} total")
        for ex in shown:
            mark = "✓" if ex["correct"] else "✗"
            print(f"    {mark} \"{ex['text']}\"")
            print(f"      → predicted: {ex['predicted_category']} "
                  f"(conf: {ex['confidence']:.3f}), "
                  f"true: {ex['true_category']}")
    else:
        print("  Low Confidence Examples: none — all predictions ≥ 0.6 confidence")

    print()
    print("  ─────────────────────────────────────────────")
    print()


# ── Quick inference test ─────────────────────────────────────────────────────

TEST_CASES: list[tuple[str, str]] = [
    ("WHOLEFDS MKT #1523", "grocery"),
    ("STARBUCKS #12345", "restaurant"),
    ("UBER *TRIP", "transport"),
    ("AMZN MKTP US*2K4J9", "shopping"),
    ("NETFLIX.COM", "entertainment"),
    ("ATT*BILL PAYMENT", "utilities"),
    ("CVS PHARMACY #1234", "healthcare"),
    ("VENMO PAYMENT", "general"),
    ("SQ *LOCAL COFFEE SHOP", "restaurant"),
    ("TRADER JOES #456", "grocery"),
    ("LYFT *RIDE 04/10", "transport"),
    ("WALMART #1234", "shopping"),
    ("SPOTIFY USA", "entertainment"),
    ("VERIZON*WIRELESS", "utilities"),
    ("WALGREENS #5678", "healthcare"),
    ("ATM WITHDRAWAL", "general"),
    ("CHIPOTLE ONLINE", "restaurant"),
    ("SHELL OIL 12345", "transport"),
    ("HULU", "entertainment"),
    ("CIGNA HEALTH INS", "healthcare"),
]


def test_inference(model_path: str) -> None:
    """
    Run the model on 20 hand-crafted examples and print pass / fail for each.
    Useful as a quick sanity check that the model loads and predicts correctly.
    """
    _check_deps()

    device = _detect_device()
    log.info("Loading model from %s (device: %s)", model_path, device)

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Train one with: python -m ml.training.train_bert"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    config = EvalConfig(model_path=model_path)
    correct = 0

    print()
    print("  SpendingCategoryBERT — Inference Test (20 examples)")
    print("  ───────────────────────────────────────────────────")

    for text, expected in TEST_CASES:
        preds, confs = predict_batch([text], model, tokenizer, config, device)
        predicted_cat = ID2LABEL.get(preds[0], "unknown")
        is_correct = predicted_cat == expected
        correct += int(is_correct)

        mark = "✓" if is_correct else "✗"
        conf_str = f"{confs[0]:.3f}"
        print(f"  {mark}  \"{text}\"")
        print(f"      predicted: {predicted_cat} ({conf_str})  expected: {expected}")

    print()
    print(f"  Result: {correct}/{len(TEST_CASES)} correct")

    if correct == len(TEST_CASES):
        print("  All tests passed!")
    elif correct >= len(TEST_CASES) * 0.8:
        print("  Good — minor misclassifications on edge cases.")
    else:
        print("  WARNING: Model accuracy below 80%. Consider retraining.")

    print()


# ── CLI entry-point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SpendingCategoryBERT and log results to MLflow",
    )
    parser.add_argument(
        "--model-path",
        default="ml/models/spending_bert",
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--test-path",
        default="ml/datasets/transactions_test.jsonl",
        help="Path to the test JSONL file",
    )
    parser.add_argument(
        "--results-path",
        default="ml/models/eval_results.json",
        help="Path to write the results JSON",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--inference-test",
        action="store_true",
        help="Run quick 20-example inference test instead of full evaluation",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="SpendingCategoryBERT",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default="evaluation",
        help="MLflow run name",
    )
    args = parser.parse_args()

    if args.inference_test:
        test_inference(args.model_path)
    else:
        config = EvalConfig(
            model_path=args.model_path,
            test_path=args.test_path,
            results_path=args.results_path,
            mlflow_experiment=args.mlflow_experiment,
            mlflow_run_name=args.mlflow_run_name,
        )
        metrics = evaluate_model(config)
        print_evaluation_report(metrics)

        if not args.no_mlflow:
            _check_deps(need_mlflow=True)
            log_to_mlflow(metrics, config, args.model_path)


if __name__ == "__main__":
    main()
