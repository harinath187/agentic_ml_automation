// Renders agents/schemas.py's ExperimentPlan - the same shape whether it
// arrives mid-run (api/run_store.py surfaces it as soon as validate_plan
// finishes, well before training completes) or on a finished run's results.
export default function PlanSummary({ plan }) {
  if (!plan) return null;

  const rows = [
    ["Problem type", plan.problem_type],
    ["Target column", plan.target_column],
    [
      "Feature columns",
      plan.feature_columns?.length ? plan.feature_columns.join(", ") : null,
    ],
  ];

  if (plan.scope_strategy) {
    rows.push([
      "Entity scope",
      `${plan.scope_strategy}${plan.entity_column ? ` (${plan.entity_column})` : ""}${
        plan.entity_filter_value ? ` = ${plan.entity_filter_value}` : ""
      }${plan.entity_selection_reasoning ? ` — ${plan.entity_selection_reasoning}` : ""}`,
    ]);
  }

  rows.push(
    ["Models to test", plan.candidate_model_families?.length ? plan.candidate_model_families.join(", ") : null]
  );

  return (
    <div className="plan-summary">
      <h3>Experiment plan</h3>
      <dl className="plan-grid">
        {rows
          .filter(([, value]) => value)
          .map(([label, value]) => (
            <div className="plan-row" key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
      </dl>
      {plan.reasoning && <p className="muted plan-reasoning">{plan.reasoning}</p>}
    </div>
  );
}
