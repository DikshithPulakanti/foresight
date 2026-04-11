"""
Fine-tune a BERT model on the 50k bank-transaction dataset to create
SpendingCategoryBERT — a classifier that maps raw bank transaction strings
(e.g. "SQ *BLUE BOTTLE 04/07") to one of 8 spending categories.

The fine-tuned model replaces keyword-based category mapping in the Receipt
Scanner agent, giving significantly better accuracy on messy real-world
merchant strings.

Usage:
    python -m ml.training.train_bert                      # full training run
    python -m ml.training.train_bert --fast-test          # ~2 min CPU smoke-test
    python -m ml.training.train_bert --model bert-base-uncased --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError:
    Dataset = None  # type: ignore[assignment,misc]
    DatasetDict = None  # type: ignore[assignment,misc]
    load_dataset = None  # type: ignore[assignment]

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    AutoModelForSequenceClassification = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    EarlyStoppingCallback = None  # type: ignore[assignment,misc]
    Trainer = None  # type: ignore[assignment,misc]
    TrainingArguments = None  # type: ignore[assignment,misc]

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
except ImportError:
    accuracy_score = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Label maps (must match generate_dataset.py) ─────────────────────────────

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

# ── Training configuration ───────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """All hyperparameters and paths for a training run."""

    model_name: str = "bert-base-uncased"
    dataset_path: str = "ml/datasets/transactions_50k.jsonl"
    test_path: str = "ml/datasets/transactions_test.jsonl"
    output_dir: str = "ml/models/spending_bert"
    hub_model_id: str = "SpendingCategoryBERT"

    num_labels: int = 8
    max_length: int = 64
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 500
    save_steps: int = 500
    early_stopping_patience: int = 3

    seed: int = 42
    fp16: bool = False

    # populated at runtime
    device: str = field(default="cpu", init=False)


# ── Dependency checks ────────────────────────────────────────────────────────


def _check_deps() -> None:
    missing: list[str] = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if Dataset is None:
        missing.append("datasets")
    if AutoTokenizer is None:
        missing.append("transformers")
    if accuracy_score is None:
        missing.append("scikit-learn")
    if missing:
        raise RuntimeError(
            f"Missing required packages: {', '.join(missing)}. "
            "Install with: pip install -r ml/training/requirements_ml.txt"
        )


def _detect_device() -> str:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_transaction_dataset(
    config: TrainingConfig,
    *,
    max_examples: int | None = None,
) -> DatasetDict:
    """
    Load JSONL files and create train / val / test splits.

    Returns a HuggingFace ``DatasetDict`` with keys ``train``, ``validation``,
    ``test``.  Prints class distribution for each split.
    """
    if not os.path.isfile(config.dataset_path):
        raise FileNotFoundError(
            f"Training dataset not found at {config.dataset_path}. "
            "Run `python -m ml.training.generate_dataset` first."
        )

    ds = load_dataset("json", data_files=config.dataset_path, split="train")

    if max_examples is not None and len(ds) > max_examples:
        ds = ds.shuffle(seed=config.seed).select(range(max_examples))

    # 80 / 10 / 10 split
    split1 = ds.train_test_split(test_size=0.2, seed=config.seed)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=config.seed)

    dataset_dict = DatasetDict(
        {
            "train": split1["train"],
            "validation": split2["train"],
            "test": split2["test"],
        }
    )

    # Override test split with curated file when available
    if os.path.isfile(config.test_path):
        curated_test = load_dataset("json", data_files=config.test_path, split="train")
        dataset_dict["test"] = curated_test
        log.info("Using curated test set from %s (%d examples)", config.test_path, len(curated_test))

    for split_name, split_ds in dataset_dict.items():
        counts = Counter(split_ds["category"])
        log.info(
            "  %-12s %5d examples  |  %s",
            split_name,
            len(split_ds),
            "  ".join(f"{cat}={n}" for cat, n in sorted(counts.items())),
        )

    return dataset_dict


# ── Tokenization ─────────────────────────────────────────────────────────────


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: Any,
    config: TrainingConfig,
) -> DatasetDict:
    """Tokenize every split, keeping only input_ids / attention_mask / label."""

    def tokenize_fn(examples: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
        )

    cols_to_remove = [c for c in dataset["train"].column_names if c not in ("label",)]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    tokenized.set_format("torch")
    return tokenized


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute accuracy, weighted F1, precision, and recall."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


# ── Training ─────────────────────────────────────────────────────────────────


def train(
    config: TrainingConfig,
    *,
    max_examples: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Full training pipeline: load data → tokenize → fine-tune → evaluate → save.

    Returns ``(trainer, metrics_dict)``.
    """
    _check_deps()
    _set_seed(config.seed)

    config.device = _detect_device()
    config.fp16 = config.device == "cuda"

    log.info("═══════════════════════════════════════════════════")
    log.info("  SpendingCategoryBERT — Training Run")
    log.info("═══════════════════════════════════════════════════")
    log.info("  Model        : %s", config.model_name)
    log.info("  Device       : %s", config.device)
    log.info("  Dataset      : %s", config.dataset_path)
    log.info("  Output       : %s", config.output_dir)
    log.info("  Epochs       : %d", config.num_epochs)
    log.info("  Batch size   : %d", config.batch_size)
    log.info("  Learning rate: %s", config.learning_rate)
    log.info("  Max length   : %d", config.max_length)
    log.info("  FP16         : %s", config.fp16)
    log.info("═══════════════════════════════════════════════════")

    if config.device == "cpu":
        log.warning("Training on CPU will be slow. Consider Google Colab for GPU training.")
        if max_examples is None:
            log.warning("Estimated time on CPU: 2-4 hours for full dataset, 5-10 min for --fast-test")

    # ── Load tokenizer and model ─────────────────────────────────────────
    log.info("Loading tokenizer and model: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Load and tokenize data ───────────────────────────────────────────
    log.info("Loading dataset …")
    dataset = load_transaction_dataset(config, max_examples=max_examples)

    log.info("Tokenizing …")
    tokenized = tokenize_dataset(dataset, tokenizer, config)

    # ── Training arguments ───────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        seed=config.seed,
        fp16=config.fp16,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # ── Train ────────────────────────────────────────────────────────────
    log.info("Starting training …")
    trainer.train()

    # ── Evaluate on test set ─────────────────────────────────────────────
    log.info("Evaluating on test set …")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    log.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    # Per-category classification report
    test_preds = trainer.predict(tokenized["test"])
    pred_labels = np.argmax(test_preds.predictions, axis=-1)
    true_labels = test_preds.label_ids

    target_names = [ID2LABEL[i] for i in range(config.num_labels)]
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        zero_division=0,
    )
    log.info("Classification Report:\n%s", report)

    # ── Save ─────────────────────────────────────────────────────────────
    log.info("Saving model and tokenizer to %s", config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    results = {
        "model_name": config.model_name,
        "device": config.device,
        "num_train_examples": len(tokenized["train"]),
        "num_val_examples": len(tokenized["validation"]),
        "num_test_examples": len(tokenized["test"]),
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "test_metrics": test_metrics,
        "classification_report": report,
    }

    results_path = os.path.join(config.output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Training results saved to %s", results_path)

    return trainer, results


# ── Fast test mode ───────────────────────────────────────────────────────────


def train_fast_test(config: TrainingConfig) -> tuple[Any, dict[str, Any]]:
    """
    Quick validation run with a tiny BERT model, 500 examples, and 1 epoch.
    Completes in ~2 minutes on CPU — useful for verifying the pipeline works
    end-to-end before committing to a full training run.
    """
    log.info("╔═══════════════════════════════════════════╗")
    log.info("║   FAST TEST MODE — pipeline validation    ║")
    log.info("╚═══════════════════════════════════════════╝")

    config.model_name = "prajjwal1/bert-tiny"
    config.num_epochs = 1
    config.batch_size = 16
    config.eval_steps = 50
    config.save_steps = 50
    config.early_stopping_patience = 2
    config.output_dir = config.output_dir + "_fast_test"

    return train(config, max_examples=500)


# ── CLI entry-point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for bank-transaction category classification",
    )
    parser.add_argument(
        "--model",
        default="bert-base-uncased",
        help="HuggingFace model name (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-device batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--dataset",
        default="ml/datasets/transactions_50k.jsonl",
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--test-dataset",
        default="ml/datasets/transactions_test.jsonl",
        help="Path to test JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/models/spending_bert",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Quick validation run with tiny BERT, 500 examples, 1 epoch",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        test_path=args.test_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )

    if args.fast_test:
        train_fast_test(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
