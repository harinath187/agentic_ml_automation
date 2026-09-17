"""CLI entry point: run the full pipeline without the Streamlit UI.

Example:
    python main.py --data sample_data/churn.csv \\
        --description "Predict which customers will churn next month"

Note: the LLM (Planner/Evaluator/Recommender/Reporter) only ever sees column
names/dtypes/aggregated stats/metrics/logs, never raw data rows - see
tests/test_llm_safety.py.
"""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from orchestration.graph import run_pipeline
from tools.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic AI ML pipeline.")
    parser.add_argument("--data", required=True, help="Path to a CSV or Excel file.")
    parser.add_argument("--description", required=True, help="Business problem description.")
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    configure_logging()
    args = parse_args()

    result = run_pipeline(
        file_path=args.data,
        business_description=args.description,
        max_retries=args.max_retries,
    )

    if result.get("needs_clarification"):
        print("Planner needs clarification:")
        print(result["plan"].clarification_question)
        return

    print("Pipeline complete.")
    print(f"Best model: {result['decision'].best_model}")
    print(f"Report: {result['report_path']}")
    print("\nMetrics:")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
