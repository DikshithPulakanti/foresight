"""
Publish SpendingCategoryBERT and its training dataset to HuggingFace Hub.

A published model on HuggingFace is a permanent, citable artifact — anyone
can download and fine-tune it with a single ``pipeline()`` call.  This script
generates professional model and dataset cards, runs pre-flight checks, and
pushes everything in one shot.

Usage:
    python -m ml.training.push_to_hub --username YOUR_HF_NAME
    python -m ml.training.push_to_hub --username YOUR_HF_NAME --model-only
    python -m ml.training.push_to_hub --username YOUR_HF_NAME --dataset-only --private
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from datasets import DatasetDict, load_dataset
except ImportError:
    DatasetDict = None  # type: ignore[assignment,misc]
    load_dataset = None  # type: ignore[assignment]

try:
    from huggingface_hub import HfApi, login
except ImportError:
    HfApi = None  # type: ignore[assignment,misc]
    login = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class HubConfig:
    """Paths and naming for the HuggingFace Hub push."""

    model_path: str = "ml/models/spending_bert"
    dataset_path: str = "ml/datasets/transactions_50k.jsonl"
    test_path: str = "ml/datasets/transactions_test.jsonl"
    results_path: str = "ml/models/eval_results.json"

    hf_username: str = ""
    model_repo: str = "SpendingCategoryBERT"
    dataset_repo: str = "bank-transaction-categories-50k"

    private: bool = False


# ── Dependency check ─────────────────────────────────────────────────────────


def _check_deps() -> None:
    missing: list[str] = []
    if load_dataset is None:
        missing.append("datasets")
    if HfApi is None:
        missing.append("huggingface_hub")
    if AutoTokenizer is None:
        missing.append("transformers")
    if missing:
        raise RuntimeError(
            f"Missing required packages: {', '.join(missing)}. "
            "Install with: pip install -r ml/training/requirements_ml.txt"
        )


# ── Pre-flight checks ───────────────────────────────────────────────────────


def preflight_check(config: HubConfig, *, check_model: bool = True, check_dataset: bool = True) -> bool:
    """
    Verify all prerequisites before pushing.

    Checks:
    - Model directory exists and contains ``config.json``
    - Dataset file exists and has > 1,000 lines
    - HuggingFace token is valid (``whoami()`` succeeds)

    Returns ``True`` if all checks pass, prints diagnostics and returns
    ``False`` otherwise.
    """
    ok = True

    if check_model:
        model_dir = Path(config.model_path)
        if not model_dir.is_dir():
            log.error("Model directory not found: %s", config.model_path)
            log.error("  → Train first: python -m ml.training.train_bert")
            ok = False
        elif not (model_dir / "config.json").is_file():
            log.error("Model directory missing config.json: %s", config.model_path)
            log.error("  → The model may not have been saved correctly")
            ok = False
        else:
            log.info("✓ Model directory found: %s", config.model_path)

    if check_dataset:
        ds_path = Path(config.dataset_path)
        if not ds_path.is_file():
            log.error("Dataset file not found: %s", config.dataset_path)
            log.error("  → Generate first: python -m ml.training.generate_dataset")
            ok = False
        else:
            line_count = sum(1 for _ in open(ds_path))
            if line_count < 1000:
                log.error(
                    "Dataset has only %d lines (expected > 1,000): %s",
                    line_count, config.dataset_path,
                )
                log.error("  → Re-run generation with a larger --target-per-category")
                ok = False
            else:
                log.info("✓ Dataset file found: %s (%d examples)", config.dataset_path, line_count)

    try:
        api = HfApi()
        user_info = api.whoami()
        log.info("✓ Authenticated as: %s", user_info.get("name", user_info.get("fullname", "unknown")))
    except Exception as exc:
        log.error("HuggingFace authentication failed: %s", exc)
        log.error("  → Set HF_TOKEN env var or pass --token")
        log.error("  → Get your token at: https://huggingface.co/settings/tokens")
        ok = False

    return ok


# ── Model card ───────────────────────────────────────────────────────────────


def create_model_card(config: HubConfig, metrics: dict[str, Any]) -> str:
    """Generate a professional HuggingFace model card with real metrics."""

    accuracy = metrics.get("accuracy", 0)
    f1_weighted = metrics.get("f1_weighted", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    test_size = metrics.get("test_set_size", "N/A")
    username = config.hf_username

    return f"""---
language: en
license: mit
tags:
- text-classification
- finance
- banking
- bert
- transaction-categorization
datasets:
- {username}/{config.dataset_repo}
metrics:
- accuracy
- f1
model-index:
- name: SpendingCategoryBERT
  results:
  - task:
      type: text-classification
    metrics:
    - type: accuracy
      value: {accuracy}
    - type: f1
      value: {f1_weighted}
---

# SpendingCategoryBERT

A BERT model fine-tuned to classify raw bank transaction descriptions into
8 spending categories. Built as part of [Foresight](https://github.com/{username}/foresight),
a proactive AI financial operating system.

## What it does

Classifies raw bank statement strings like `"WHOLEFDS MKT #1523"` or
`"AMZN MKTP US*2K4J9"` into clean spending categories — no regex rules needed.

## Categories

| Label | Category | Example transaction |
|-------|----------|---------------------|
| 0 | grocery | WHOLEFDS MKT #1523 |
| 1 | restaurant | STARBUCKS #12345 |
| 2 | transport | UBER *TRIP |
| 3 | shopping | AMZN MKTP US*2K4J9 |
| 4 | entertainment | NETFLIX.COM |
| 5 | utilities | ATT*BILL PAYMENT |
| 6 | healthcare | CVS PHARMACY #1234 |
| 7 | general | VENMO PAYMENT |

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | {accuracy:.1%} |
| F1 (weighted) | {f1_weighted:.3f} |
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |

Evaluated on held-out test set of {test_size} examples.

## Quick start

```python
from transformers import pipeline

classifier = pipeline("text-classification",
                      model="{username}/{config.model_repo}")

result = classifier("WHOLEFDS MKT #1523")
print(result)  # [{{'label': 'grocery', 'score': 0.98}}]
```

## Training

- **Base model:** bert-base-uncased
- **Dataset:** [{username}/{config.dataset_repo}](https://huggingface.co/datasets/{username}/{config.dataset_repo}) (50,000 examples)
- **Training:** 5 epochs, lr=2e-5, batch_size=32, early stopping on F1
- **Hardware:** See `training_results.json` for details

## About Foresight

SpendingCategoryBERT powers the transaction categorization in
[Foresight](https://github.com/{username}/foresight) — a proactive AI system
that monitors finances across bank accounts, email, and calendar using
12 specialized LangGraph agents and 6 custom MCP servers.

## License

MIT
"""


# ── Dataset card ─────────────────────────────────────────────────────────────


def create_dataset_card(config: HubConfig) -> str:
    """Generate a HuggingFace dataset card."""

    username = config.hf_username

    return f"""---
language: en
license: mit
tags:
- finance
- banking
- text-classification
- transaction-categorization
task_categories:
- text-classification
size_categories:
- 10K<n<100K
---

# Bank Transaction Categories (50k)

A dataset of 50,000 labeled bank transaction descriptions for training
spending category classifiers. Generated as part of the
[Foresight](https://github.com/{username}/foresight) project.

## Dataset description

50,000 realistic bank transaction strings in 8 categories.
Transaction strings match real bank statement format (ALL CAPS, truncated,
with store numbers and transaction IDs).

Each example has:
- `text` — the raw bank transaction string
- `label` — integer category label (0–7)
- `category` — human-readable category name

## Categories and distribution

| Category | Count | Examples |
|----------|-------|---------|
| grocery | ~6,250 | WHOLEFDS MKT #1523, TRADER JOES #456 |
| restaurant | ~6,250 | STARBUCKS #12345, CHIPOTLE ONLINE |
| transport | ~6,250 | UBER *TRIP, LYFT *1234567 |
| shopping | ~6,250 | AMZN MKTP US*2K4J9, TARGET #0234 |
| entertainment | ~6,250 | NETFLIX.COM, SPOTIFY USA |
| utilities | ~6,250 | ATT*BILL PAYMENT, VERIZON*WIRELESS |
| healthcare | ~6,250 | CVS PHARMACY #1234, CIGNA HEALTH |
| general | ~6,250 | VENMO PAYMENT, ATM WITHDRAWAL |

## Generation method

Generated using Claude Haiku API from 80 seed merchants (10 per category).
Each seed produced ~50 realistic variations in real bank statement format.
See `ml/training/generate_dataset.py` for the full generation pipeline.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{username}/{config.dataset_repo}")
print(ds["train"][0])
# {{'text': 'WHOLEFDS MKT #1523', 'label': 0, 'category': 'grocery'}}
```

## Trained models

- [{username}/SpendingCategoryBERT](https://huggingface.co/{username}/{config.model_repo}) — BERT fine-tuned on this dataset

## License

MIT
"""


# ── Push model ───────────────────────────────────────────────────────────────


def push_model(config: HubConfig) -> str:
    """
    Push the fine-tuned model and tokenizer to HuggingFace Hub.

    Also uploads a model card (README.md) with metrics if eval results exist.
    Returns the full repo URL.
    """
    repo_id = f"{config.hf_username}/{config.model_repo}"
    log.info("Pushing model to %s …", repo_id)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_path)

    model.push_to_hub(repo_id, private=config.private)
    tokenizer.push_to_hub(repo_id, private=config.private)
    log.info("Model and tokenizer pushed")

    # Upload model card
    metrics: dict[str, Any] = {}
    results_file = Path(config.results_path)
    if results_file.is_file():
        with open(results_file) as f:
            metrics = json.load(f)
        log.info("Loaded eval metrics from %s", config.results_path)
    else:
        log.warning("No eval results found at %s — model card will have placeholder metrics", config.results_path)

    card_content = create_model_card(config, metrics)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    log.info("Model card uploaded")

    # Upload training results as artifact
    if results_file.is_file():
        api.upload_file(
            path_or_fileobj=str(results_file),
            path_in_repo="eval_results.json",
            repo_id=repo_id,
            repo_type="model",
        )
        log.info("Eval results uploaded as artifact")

    url = f"https://huggingface.co/{repo_id}"
    log.info("Model published: %s", url)
    return url


# ── Push dataset ─────────────────────────────────────────────────────────────


def push_dataset(config: HubConfig) -> str:
    """
    Push the transaction dataset to HuggingFace Hub with train/test splits.

    Also uploads a dataset card (README.md).
    Returns the full dataset URL.
    """
    repo_id = f"{config.hf_username}/{config.dataset_repo}"
    log.info("Pushing dataset to %s …", repo_id)

    train_ds = load_dataset("json", data_files=config.dataset_path, split="train")

    splits = {"train": train_ds}
    if Path(config.test_path).is_file():
        test_ds = load_dataset("json", data_files=config.test_path, split="train")
        splits["test"] = test_ds
        log.info("Including test split from %s (%d examples)", config.test_path, len(test_ds))

    dataset_dict = DatasetDict(splits)
    dataset_dict.push_to_hub(repo_id, private=config.private)
    log.info("Dataset pushed (%d train examples)", len(train_ds))

    # Upload dataset card
    card_content = create_dataset_card(config)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    log.info("Dataset card uploaded")

    url = f"https://huggingface.co/datasets/{repo_id}"
    log.info("Dataset published: %s", url)
    return url


# ── CLI entry-point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push SpendingCategoryBERT and dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("HF_USERNAME", ""),
        help="Your HuggingFace username (or set HF_USERNAME env var)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--model-path",
        default="ml/models/spending_bert",
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--dataset-path",
        default="ml/datasets/transactions_50k.jsonl",
        help="Path to the training JSONL",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Push only the model (skip dataset)",
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Push only the dataset (skip model)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repos (default: public)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight checks",
    )
    args = parser.parse_args()

    # ── Resolve token ────────────────────────────────────────────────────
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        log.error("No HuggingFace token provided.")
        log.error("  → Pass --token YOUR_TOKEN or set the HF_TOKEN env var")
        log.error("  → Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # ── Resolve username ─────────────────────────────────────────────────
    if not args.username:
        log.error("No HuggingFace username provided.")
        log.error("  → Pass --username YOUR_NAME or set HF_USERNAME env var")
        sys.exit(1)

    _check_deps()
    login(token=token)

    config = HubConfig(
        hf_username=args.username,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        private=args.private,
    )

    push_model_flag = not args.dataset_only
    push_dataset_flag = not args.model_only

    # ── Pre-flight ───────────────────────────────────────────────────────
    if not args.skip_preflight:
        log.info("Running pre-flight checks …")
        if not preflight_check(config, check_model=push_model_flag, check_dataset=push_dataset_flag):
            log.error("Pre-flight checks failed. Fix the issues above or pass --skip-preflight.")
            sys.exit(1)
        log.info("All pre-flight checks passed")

    # ── Push ─────────────────────────────────────────────────────────────
    model_url = ""
    dataset_url = ""

    if push_model_flag:
        model_url = push_model(config)

    if push_dataset_flag:
        dataset_url = push_dataset(config)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║     Published to HuggingFace Hub!            ║")
    print("╚══════════════════════════════════════════════╝")
    if model_url:
        print(f"  Model:   {model_url}")
    if dataset_url:
        print(f"  Dataset: {dataset_url}")
    print()


if __name__ == "__main__":
    main()
