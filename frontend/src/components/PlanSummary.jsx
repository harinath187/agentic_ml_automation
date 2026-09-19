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

  // For classification/regression, orchestration/graph.py::node_train trains
  // every registered model for the problem type, not just the Planner's
  // shortlist - showing that shortlist here reads as "only these models
  // ran" when the Model comparison tab shows more, so it's hidden entirely
  // for those problem types rather than relabeled.
  const trainsEveryRegisteredModel = plan.problem_type === "classification" || plan.problem_type === "regression";

  if (!trainsEveryRegisteredModel) {
    rows.push([
      "Models to test",
      plan.candidate_model_families?.length ? plan.candidate_model_families.join(", ") : null,
    ]);
  }

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
      {trainsEveryRegisteredModel && (
        <p className="muted plan-reasoning">
          Every available model for this problem type is actually trained and scored — see the Model comparison tab
          for the full results.
        </p>
      )}
    </div>
  );
}
