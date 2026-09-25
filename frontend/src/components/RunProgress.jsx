// Maps orchestration/graph.py's raw LangGraph node names (what api/run_store.py
// persists as current_step, see CLAUDE.md's "Progress reporting") onto a short,
// human-friendly sequence of pipeline stages, so the user sees "Training models"
// instead of "per_entity_pipeline". Node names are grouped many-to-one where the
// backend has several internal nodes that read as one stage to a user (e.g. the
// scope-dependent train/per_entity_pipeline/hierarchical_train branches all mean
// "training" regardless of which one actually ran for this dataset).
const STAGES = [
  { key: "prepare", label: "Preparing data", nodes: ["ingest", "profile_data", "quality_analysis", "detect_problem"] },
  { key: "plan", label: "Planning", nodes: ["planner_agent", "validate_plan"] },
  { key: "select", label: "Cleaning & feature selection", nodes: ["filter_entity", "feature_selection", "eda", "clean"] },
  { key: "clean", label: "Feature engineering", nodes: ["feature_engineer"] },
  { key: "split", label: "Splitting data", nodes: ["split"] },
  { key: "tune", label: "Tuning models", nodes: ["tune_models"] },
  { key: "train", label: "Training models", nodes: ["train", "per_entity_pipeline", "hierarchical_train"] },
  { key: "evaluate", label: "Evaluating models", nodes: ["evaluate"] },
  { key: "recommend", label: "Recommending", nodes: ["recommend"] },
  { key: "report", label: "Finalizing report", nodes: ["report"] },
];

const NODE_TO_STAGE_INDEX = STAGES.reduce((acc, stage, index) => {
  for (const node of stage.nodes) acc[node] = index;
  return acc;
}, {});

export default function RunProgress({ status, currentStep }) {
  const normalizedStep = currentStep?.startsWith("node_") ? currentStep.slice(5) : currentStep;
  const activeIndex = status === "running" ? NODE_TO_STAGE_INDEX[normalizedStep] ?? -1 : -1;

  return (
    <div className="run-progress">
      <ol className="run-progress-steps">
        {STAGES.map((stage, index) => {
          const state =
            status === "queued" || activeIndex === -1
              ? "pending"
              : index < activeIndex
              ? "done"
              : index === activeIndex
              ? "active"
              : "pending";
          return (
            <li key={stage.key} className={`run-progress-step run-progress-step--${state}`}>
              <span className="run-progress-marker" aria-hidden="true">
                {state === "done" ? "✓" : state === "active" ? <span className="run-progress-spinner" /> : index + 1}
              </span>
              <span className="run-progress-label">{stage.label}</span>
            </li>
          );
        })}
      </ol>
      <p className="muted run-progress-status">
        {status === "queued" && "Waiting for a free worker..."}
        {status === "running" && activeIndex === -1 && "Starting..."}
        {status === "running" && activeIndex !== -1 && `Current step: ${STAGES[activeIndex].label} (${normalizedStep}).`}
      </p>
    </div>
  );
}
