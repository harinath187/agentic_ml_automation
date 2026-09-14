"""Reporter Agent: writes the final business-friendly recommendation report.

Renders reports/template.html via Jinja2 into a standalone HTML file. Only
receives plan/metrics/decision context - never raw data.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from agents.llm_client import call_llm_json
from agents.schemas import EvaluatorDecision, PipelinePlan, ReportContent

SYSTEM_PROMPT = """You are the Reporter Agent in an automated ML pipeline.
Given the business problem, the pipeline plan, the model metrics, and the
evaluator's decision, write a business-friendly report. Avoid jargon where
possible, and never reference individual data rows - only aggregated
findings. Fill every field of the requested schema.
"""

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "reports"
OUTPUT_DIR = TEMPLATE_DIR / "generated"


def generate_report(
    business_description: str,
    plan: PipelinePlan,
    metrics: dict,
    decision: EvaluatorDecision,
    output_path: str | None = None,
) -> str:
    user_prompt = (
        f"Business description:\n{business_description}\n\n"
        f"Pipeline plan (JSON):\n{plan.model_dump_json(indent=2)}\n\n"
        f"Model metrics (JSON):\n{json.dumps(metrics, indent=2)}\n\n"
        f"Evaluator decision (JSON):\n{decision.model_dump_json(indent=2)}"
    )
    content = call_llm_json(SYSTEM_PROMPT, user_prompt, ReportContent)

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("template.html")
    html = template.render(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        business_description=business_description,
        plan=plan,
        metrics=metrics,
        decision=decision,
        content=content,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else OUTPUT_DIR / f"report_{datetime.now():%Y%m%d_%H%M%S}.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)
