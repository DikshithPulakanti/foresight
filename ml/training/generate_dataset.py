"""
Generate a 50,000-example dataset of bank transaction descriptions with spending
categories for fine-tuning SpendingCategoryBERT.

Raw bank transactions look like "SQ *BLUE BOTTLE 04/07" or "AMZN MKTP US*2K4J9".
A human can parse these, but a model needs labelled training data.  This script
uses Claude Haiku to expand a curated seed list into 50k realistic variations
across 8 spending categories, then writes the result as JSONL.

Usage:
    python -m ml.training.generate_dataset                       # full 50k run
    python -m ml.training.generate_dataset --dry-run             # ~80 examples
    python -m ml.training.generate_dataset --output my_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from typing import Any

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore[assignment,misc]

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kw):  # type: ignore[misc]
        """Fallback when tqdm is not installed."""
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Category schema ──────────────────────────────────────────────────────────

CATEGORY_LABELS: dict[str, int] = {
    "grocery": 0,
    "restaurant": 1,
    "transport": 2,
    "shopping": 3,
    "entertainment": 4,
    "utilities": 5,
    "healthcare": 6,
    "general": 7,
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "grocery": "supermarkets, food stores",
    "restaurant": "cafes, fast food, dining",
    "transport": "rideshare, gas, parking, airlines",
    "shopping": "retail, online, clothing",
    "entertainment": "streaming, events, subscriptions",
    "utilities": "phone, internet, electricity, water",
    "healthcare": "pharmacy, doctor, hospital",
    "general": "everything else",
}

# ── Seed merchants (real bank-statement format) ─────────────────────────────

SEED_MERCHANTS: dict[str, list[str]] = {
    "grocery": [
        "WHOLEFDS MKT #1523", "TRADER JOES #123", "SAFEWAY #0234",
        "KROGER #4521", "WALMART GROCERY", "TARGET GROCERY", "COSTCO WHSE",
        "ALDI #0412", "PUBLIX #1234", "HEB #0892",
    ],
    "restaurant": [
        "SQ *BLUE BOTTLE", "STARBUCKS #12345", "CHIPOTLE ONLINE",
        "MCDONALD'S F12345", "DOORDASH*SUBWAY", "UBER EATS*PIZZA",
        "GRUBHUB*THAI", "SQ *LOCAL CAFE", "TST*FINE DINING", "PANERA BREAD",
    ],
    "transport": [
        "UBER *TRIP", "LYFT *RIDE", "SHELL OIL 12345",
        "BP#1234567", "CHEVRON 0012", "SP PARKING #023",
        "DELTA AIR", "UNITED AIRLINES", "AMTRAK", "MTA*METROCARD",
    ],
    "shopping": [
        "AMZN MKTP US*2K4J9", "AMAZON.COM*AB1234",
        "TARGET #0234", "WALMART #1234", "BEST BUY #0234",
        "APPLE.COM/BILL", "ETSY.COM", "EBAY*12345", "ZARA USA", "H&M #1234",
    ],
    "entertainment": [
        "NETFLIX.COM", "SPOTIFY USA", "HULU", "DISNEY+",
        "APPLE ITUNES", "STEAM GAMES", "AMC THEATRES",
        "TICKETMASTER", "HBO MAX", "YOUTUBE PREMIUM",
    ],
    "utilities": [
        "AT&T*BILL", "VERIZON*WIRELESS", "COMCAST*CABLE",
        "T-MOBILE*AUTO PAY", "PG&E ELECTRIC", "CON EDISON",
        "WATER DEPT", "SPECTRUM*INTERNET", "GOOGLE FI", "XFINITY MOBILE",
    ],
    "healthcare": [
        "CVS PHARMACY #1234", "WALGREENS #5678",
        "RITE AID #0234", "CIGNA HEALTH", "AETNA PREMIUM",
        "LABCORP", "QUEST DIAGNOSTICS", "URGENT CARE #123",
        "KAISER PERMANENTE", "BLUE CROSS PAYMENT",
    ],
    "general": [
        "VENMO PAYMENT", "ZELLE TRANSFER", "ATM WITHDRAWAL",
        "BANK FEE", "INTEREST CHARGE", "CHECK #1234",
        "ACH TRANSFER", "WIRE TRANSFER", "PAYPAL TRANSFER", "CASHAPP*PAYMENT",
    ],
}

# ── Hand-verified test examples ─────────────────────────────────────────────

VERIFIED_EXAMPLES: dict[str, list[str]] = {
    "grocery": [
        "WHOLEFDS MKT #1523", "TRADER JOES #456", "SAFEWAY STORE 0001",
    ],
    "restaurant": [
        "STARBUCKS #99999", "CHIPOTLE ONLINE ORDER", "MCDONALDS #12345",
    ],
    "transport": [
        "UBER *TRIP HELP", "LYFT *1234567", "SHELL SERVICE STN",
    ],
    "shopping": [
        "AMAZON.COM*PURCHASE", "TARGET STORE #0001", "BEST BUY #00001",
    ],
    "entertainment": [
        "NETFLIX.COM", "SPOTIFY AB1CD2", "HULU 123456789",
    ],
    "utilities": [
        "ATT*BILL PAYMENT", "VERIZON WIRELESS", "COMCAST CABLE",
    ],
    "healthcare": [
        "CVS PHARMACY #0001", "WALGREENS #00001", "CIGNA HEALTH INS",
    ],
    "general": [
        "VENMO PAYMENT", "ATM WITHDRAWAL", "BANK SERVICE FEE",
    ],
}

# ── Cost estimation ──────────────────────────────────────────────────────────

HAIKU_INPUT_PRICE_PER_1K = 0.00025   # USD per 1k input tokens
HAIKU_OUTPUT_PRICE_PER_1K = 0.00125  # USD per 1k output tokens
AVG_INPUT_TOKENS_PER_CALL = 120
AVG_OUTPUT_TOKENS_PER_CALL = 600


def estimate_cost(total_api_calls: int) -> dict[str, float]:
    """Return estimated input, output, and total cost in USD."""
    input_cost = (total_api_calls * AVG_INPUT_TOKENS_PER_CALL / 1000) * HAIKU_INPUT_PRICE_PER_1K
    output_cost = (total_api_calls * AVG_OUTPUT_TOKENS_PER_CALL / 1000) * HAIKU_OUTPUT_PRICE_PER_1K
    return {
        "api_calls": total_api_calls,
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(input_cost + output_cost, 4),
    }


# ── Claude-powered variation generator ──────────────────────────────────────

def generate_variations(
    client: Anthropic,
    merchant: str,
    category: str,
    n: int = 50,
    *,
    max_retries: int = 3,
) -> list[str]:
    """
    Ask Claude Haiku to generate *n* realistic bank-statement variations of
    *merchant* within *category*.  Returns a list of stripped, non-empty strings.
    """
    prompt = (
        f"Generate {n} realistic bank transaction description variations "
        f"for a {category} merchant similar to \"{merchant}\".\n\n"
        "Rules:\n"
        "- Format like real bank statements (ALL CAPS, truncated, with codes)\n"
        "- Include realistic suffixes: store numbers (#1234), dates (04/07), "
        "transaction IDs (*AB1234), location codes\n"
        f"- Vary the format but keep it recognizable as {category}\n"
        "- Each on a new line, no numbering, no explanations\n"
        "- Examples of real bank format: SQ *COFFEE #123, WHOLEFDS MKT #0892 04/07\n\n"
        "Return ONLY the transaction strings, one per line."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model="claude-haiku-20240307",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            lines = response.content[0].text.strip().split("\n")
            results = [line.strip() for line in lines if line.strip()]
            return results[:n]
        except Exception as exc:
            log.warning(
                "API call failed (attempt %d/%d) for '%s': %s",
                attempt, max_retries, merchant, exc,
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    log.error("All retries exhausted for '%s' — returning empty list", merchant)
    return []


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_dataset(
    output_path: str,
    target_per_category: int = 6250,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Generate the full training dataset and write it to *output_path* as JSONL.

    Each line is ``{"text": "...", "label": 0-7, "category": "...", "source": "..."}``

    Returns the list of all examples.
    """
    if Anthropic is None:
        raise RuntimeError(
            "The 'anthropic' package is required. Install it with: pip install anthropic"
        )

    client = Anthropic()

    total_seeds = sum(len(v) for v in SEED_MERCHANTS.values())
    cost = estimate_cost(total_seeds)

    log.info("─── Cost Estimate ───────────────────────────────")
    log.info("  API calls        : %d", cost["api_calls"])
    log.info("  Input cost (est) : $%.4f", cost["input_cost"])
    log.info("  Output cost (est): $%.4f", cost["output_cost"])
    log.info("  Total cost (est) : $%.4f", cost["total_cost"])
    log.info("  Target examples  : %d (%d per category × %d categories)",
             target_per_category * len(CATEGORY_LABELS),
             target_per_category, len(CATEGORY_LABELS))
    log.info("─────────────────────────────────────────────────")

    if dry_run:
        log.info("DRY RUN — generating ~%d examples per category", target_per_category)

    all_examples: list[dict[str, Any]] = []

    for category, seeds in SEED_MERCHANTS.items():
        label = CATEGORY_LABELS[category]
        category_examples: list[dict[str, Any]] = []

        log.info("Generating category: %s (label=%d)", category, label)

        for seed in tqdm(seeds, desc=f"  {category}", unit="seed"):
            variations_n = max(1, target_per_category // len(seeds))
            if dry_run:
                variations_n = min(variations_n, 5)

            variations = generate_variations(client, seed, category, n=variations_n)
            for v in variations:
                category_examples.append({
                    "text": v,
                    "label": label,
                    "category": category,
                    "source": "generated",
                })

        for seed in seeds:
            category_examples.append({
                "text": seed,
                "label": label,
                "category": category,
                "source": "seed",
            })

        if len(category_examples) > target_per_category:
            category_examples = random.sample(category_examples, target_per_category)

        log.info("  %s: %d examples", category, len(category_examples))
        all_examples.extend(category_examples)

    random.shuffle(all_examples)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    log.info("Total: %d examples saved to %s", len(all_examples), output_path)
    return all_examples


# ── Test-set generation ──────────────────────────────────────────────────────

def generate_test_set(output_path: str) -> list[dict[str, Any]]:
    """
    Write a small hand-verified test set (~24 examples) to *output_path* as JSONL.
    """
    test_examples: list[dict[str, Any]] = []

    for category, examples in VERIFIED_EXAMPLES.items():
        label = CATEGORY_LABELS[category]
        for text in examples:
            test_examples.append({
                "text": text,
                "label": label,
                "category": category,
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for ex in test_examples:
            f.write(json.dumps(ex) + "\n")

    log.info("Test set: %d examples saved to %s", len(test_examples), output_path)
    return test_examples


# ── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SpendingCategoryBERT training data via Claude Haiku",
    )
    parser.add_argument(
        "--output",
        default="ml/datasets/transactions_50k.jsonl",
        help="Path for the training JSONL (default: ml/datasets/transactions_50k.jsonl)",
    )
    parser.add_argument(
        "--test-output",
        default="ml/datasets/transactions_test.jsonl",
        help="Path for the test JSONL (default: ml/datasets/transactions_test.jsonl)",
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=6250,
        help="Examples per category (default: 6250 → 50k total)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate only ~10 examples per category to test the pipeline",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.target_per_category = 10

    os.makedirs("ml/datasets", exist_ok=True)

    generate_dataset(
        args.output,
        args.target_per_category,
        dry_run=args.dry_run,
    )
    generate_test_set(args.test_output)

    log.info("Dataset generation complete!")


if __name__ == "__main__":
    main()
