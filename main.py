"""CLI entry point: run the full pipeline without the Streamlit UI.

Example:
    python main.py --data sample_data/churn.csv \\
        --description "Predict which customers will churn next month" \\
        --sensitive-columns customer_name,ssn
"""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from orchestration.graph import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic AI ML pipeline.")
    parser.add_argument("--data", required=True, help="Path to a CSV or Excel file.")
    parser.add_argument("--description", required=True, help="Business problem description.")
    parser.add_argument(
        "--sensitive-columns",
        default="",
        help="Comma-separated column names to exclude from anything sent to the LLM.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=60, help="AutoML training time budget per attempt, in seconds.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    sensitive_columns = [c.strip() for c in args.sensitive_columns.split(",") if c.strip()]

    result = run_pipeline(
        file_path=args.data,
        business_description=args.description,
        sensitive_columns=sensitive_columns,
        max_retries=args.max_retries,
        time_limit_s=args.time_limit,
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
