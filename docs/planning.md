# Structured hypothesis planning and critique

Every new campaign freezes a hypothesis before training. The default offline provider emits a deterministic, explicitly labeled hypothesis. `planner-input.json`, `planning.json`, and `hypothesis.json` preserve its inputs, call reservation/receipt, and exact output.

The allowed action space currently contains only `gcn → ego_gcn`. The planner must state a mechanism, competing explanation, falsification condition, next experiment, and a minimum practical macro-F1 effect. Unknown fields, code, metric substitutions, arbitrary models, nonfinite effects, or out-of-range thresholds are rejected. Planner text does not control evaluation, job budgets, code execution, or splits.

## Optional OpenAI provider

Copy an example configuration and add:

```json
{
  "planner": "openai",
  "planner_model": "YOUR_SUPPORTED_MODEL_ID",
  "max_llm_calls": 1
}
```

Set `OPENAI_API_KEY` in the invoking process environment using your normal secret-management method. Never put it in a JSON config or commit it. No model is silently selected. The adapter sends only the public training-scope graph profile and the allowed intervention to `https://api.openai.com/v1/responses`.

The request uses strict JSON-schema output, no tools, `store: false`, a 2,000-output-token limit, and a 45-second HTTP timeout. There is one planning call per campaign. Its reservation is persisted before sending; failures count, remain recorded, and are never automatically retried. The configured call ceiling must permit the call. Missing credentials fail clearly. Response IDs and usage are saved, credentials are not. Worker environments omit the API key.

The implementation follows the [official Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs). The adapter is tested with fixture transports, including refusal and malformed output. A live API run requires credentials; the checked-in tests do not spend API credits.

## Critic policy

The critic is deterministic and evidence-backed in this version. It compares the predeclared minimum effect with every paired seed difference:

- Fewer than five complete predeclared paired seeds: `inconclusive`.
- Every difference meets the threshold: `supported_exploratory`.
- Every difference is below the threshold: `rejected_exploratory`.
- Mixed results: `inconclusive`.

These are conservative engineering labels, not significance tests. Seed agreement on one adaptively used validation split is not independent confirmation. Capacity confounds and graph/split limitations remain in the critique and report. Tests exercise rejected and inconclusive outcomes as well as support.

This release runs one bounded hypothesis/intervention cycle. Adaptive replanning, evidence retrieval across campaigns, and additional intervention families remain future work.
