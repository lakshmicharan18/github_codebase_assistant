"""
Evaluates utils/query_router.py's route_question() against a labeled
golden question set.

Usage:
    python -m eval.evaluate_router

Requires GROQ_API_KEY to be set (via .env or environment variable), since
route_question() makes a live Groq call for each question.

Outputs:
    - Overall accuracy
    - Per-category accuracy
    - Confusion matrix (expected -> what it actually predicted)
    - List of every misclassified question, for manual review
    - A CSV of raw results: eval/router_eval_results.csv
"""

import csv
import os
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv

# Allow running this as `python -m eval.evaluate_router` or `python eval/evaluate_router.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.query_router import route_question, VALID_CATEGORIES
from eval.golden_router_questions import GOLDEN_QUESTIONS

load_dotenv()


def run_eval():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in your .env file or environment.")
        sys.exit(1)

    results = []
    correct = 0
    total = len(GOLDEN_QUESTIONS)

    # expected_category -> {predicted_category: count}
    confusion = defaultdict(lambda: defaultdict(int))
    per_category_total = defaultdict(int)
    per_category_correct = defaultdict(int)

    print(f"Running router evaluation on {total} questions...\n")

    for i, (question, expected) in enumerate(GOLDEN_QUESTIONS, start=1):
        try:
            predicted = route_question(question, groq_api_key)
        except Exception as e:
            predicted = f"ERROR: {e}"

        is_correct = predicted == expected
        if is_correct:
            correct += 1
            per_category_correct[expected] += 1

        per_category_total[expected] += 1
        confusion[expected][predicted] += 1

        results.append({
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct
        })

        status = "✓" if is_correct else "✗"
        print(f"[{i}/{total}] {status}  expected={expected:<14} predicted={predicted:<14} | {question}")

        # Small delay to be gentle on rate limits
        time.sleep(0.3)

    # --- Write raw results to CSV ---
    csv_path = os.path.join(os.path.dirname(__file__), "router_eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "expected", "predicted", "correct"])
        writer.writeheader()
        writer.writerows(results)

    # --- Summary ---
    accuracy = correct / total * 100

    print("\n" + "=" * 60)
    print(f"OVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print("=" * 60)

    print("\nPer-category accuracy:")
    for category in VALID_CATEGORIES:
        cat_total = per_category_total.get(category, 0)
        if cat_total == 0:
            continue
        cat_correct = per_category_correct.get(category, 0)
        cat_acc = cat_correct / cat_total * 100
        print(f"  {category:<14} {cat_correct}/{cat_total} ({cat_acc:.0f}%)")

    print("\nConfusion matrix (expected -> predicted: count):")
    for expected in sorted(confusion.keys()):
        for predicted, count in sorted(confusion[expected].items(), key=lambda x: -x[1]):
            marker = "" if expected == predicted else "  <-- MISCLASSIFIED"
            print(f"  {expected:<14} -> {predicted:<14} : {count}{marker}")

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        print(f"\nMisclassified questions ({len(misclassified)}):")
        for r in misclassified:
            print(f"  '{r['question']}'")
            print(f"    expected: {r['expected']}, got: {r['predicted']}")
    else:
        print("\nNo misclassifications — router got everything right on this set.")

    print(f"\nFull results saved to: {csv_path}")

    if accuracy < 85:
        print(
            "\n⚠️  Accuracy is below 85% — consider tightening the category "
            "definitions in utils/query_router.py's prompt, especially "
            "around the categories with the most confusion above."
        )

    return accuracy, results


if __name__ == "__main__":
    run_eval()