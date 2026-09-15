# Structured hypothesis planning and critique

Every new campaign freezes a hypothesis before training. The default offline provider emits a deterministic, explicitly labeled hypothesis. `planner-input.json`, `planning.json`, and `hypothesis.json` preserve its inputs, call reservation/receipt, and exact output.

The allowed action space currently contains only `gcn → ego_gcn`. The planner must state a mechanism, competing explanation, falsification condition, next experiment, and a minimum practical macro-F1 effect. Unknown fields, code, metric substitutions, arbitrary models, nonfinite effects, or out-of-range thresholds are rejected. Planner text does not control evaluation, job budgets, code execution, or splits.

## Codex planner with ChatGPT sign-in

Codex CLI can run the planner using your existing ChatGPT sign-in. No API key is needed.

```sh
codex login status
# If needed: codex login
python -m graphsmith research examples/codex-pairs.json
```

The example sets `planner` to `codex`, `max_llm_calls` to 1, and leaves `planner_model` empty to use the CLI default. Set a model ID explicitly if desired. Codex uses your account's Codex access and usage limits, as described in the [official non-interactive-mode documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

The adapter requires ChatGPT authentication and removes `OPENAI_API_KEY` and `CODEX_API_KEY` from the child environment. It does not read, copy, or export credentials; the CLI handles authentication. It uses an ephemeral temporary working directory, schema-constrained output, read-only execution, and disabled shell, apps, plugins, browser, image, and multi-agent capabilities. The invocation ignores user config and requests host-skill discovery to be skipped; authentication still uses the existing Codex login. Codex can still supply built-in or system-managed context, and its nonfatal diagnostics are recorded in the receipt. Only the training-scope profile is supplied as input. Unsupported tool events cause the proposal to be rejected.

The maximum planning duration is 120 seconds. There is one reserved CLI planning turn and no application-level retry. A CLI turn can contain internal model requests or connection retries, so `max_llm_calls` counts CLI turns for this provider, not individual backend requests. Usage and the ephemeral thread ID are retained in the planning receipt. Existing training and final-evaluation budgets are unchanged.

The standard offline example stays available for deterministic tests and runs without a Codex installation. To use Codex with a sealed campaign, copy `sealed-pairs.json`, set `planner: "codex"` and `max_llm_calls: 1`, and build the evaluator image from the updated source before starting a new evaluator volume.

## Optional OpenAI API provider

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
