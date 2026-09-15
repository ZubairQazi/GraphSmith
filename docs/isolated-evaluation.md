# Isolated final evaluation (macOS development backend)

The evaluator stores labels and its private generator seed in a named Docker volume. It exposes only the public dataset, lifecycle status, method freezing, and one final score. Neither training nor inference mounts this volume. Research workers run under a tested macOS sandbox with no networking, no Docker socket access, and no read access outside approved runtime/source/campaign paths.

## Start a new evaluator

Docker Desktop must be running. From the project directory:

```sh
docker build -f deploy/evaluator.Dockerfile -t graphsmith-evaluator:0.2 .
docker run -d --name graphsmith-evaluator \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 128m --cpus 1 \
  -p 127.0.0.1:8765:8080 \
  -v graphsmith-evaluator-v1:/vault graphsmith-evaluator:0.2
```

The Python base image is pinned by digest. The container runs as UID 65532 and has no host bind mounts. Keep the named volume as the private audit record. Each independent campaign needs a fresh evaluator volume; a frozen or evaluated volume cannot start a new search. Stop the service with `docker stop graphsmith-evaluator` when finished. Restart the same container to recover an interrupted transport request without resetting its evaluation ledger.

## Research, freeze, evaluate

```sh
python -m graphsmith research examples/sealed-pairs.json --evaluator http://127.0.0.1:8765
python -m graphsmith freeze runs/RUN_ID
python -m graphsmith finalize runs/RUN_ID
```

`finalize` also performs the local freeze if needed. The example reserves 26 jobs: 25 development jobs and one final ensemble-inference job. It uses the offline planner; no API credential is needed.

The trusted service generates a 60/20/20 train/validation/test partition. Test query endpoints are public, but labels are omitted. A private random generator seed replaces `data_seed` for sealed campaigns and is not sent to the planner or saved in the public bundle. This prevents trivial recovery of withheld labels from the public seed. Reproduction uses the saved public dataset and private service audit record, not regeneration from the example config.

Selection maximizes mean development macro-F1 across every predeclared seed, with model name breaking ties. The selected checkpoints are frozen by content hash. Their probabilities are averaged into one ensemble submission, using the threshold 0.5 declared before search. Required baselines and all planned jobs must complete before freezing.

Before exposure, the evaluator freezes the dataset, code, model, seed list, checkpoint hashes, threshold, selection rule, and exact prediction hash. It rejects changed submissions and returns a stored receipt only for an exact request replay. That replay is transport recovery, not another test evaluation. The report labels the single-graph generalization limit explicitly.

## Trust boundary and current limits

- The local OS user, Docker administrator, source code, and orchestration CLI are trusted setup/control components. They can administer the evaluator; this is not protection against a malicious machine owner.
- The LLM receives a training-scope profile through a tool-free structured-output API call. It cannot execute code, modify evaluation rules, or read files.
- The only intervention is preimplemented. Arbitrary generated code and a general-purpose coding agent are not supported inside this boundary.
- The HTTP service binds to localhost on the host. It is a single-user development service without multi-user authentication or TLS; do not expose it on a public interface.
- The worker sandbox is macOS-specific and fails closed elsewhere. Linux CI tests service semantics and the deterministic engine; macOS CI additionally exercises filesystem and network denial.
- A literal root-directory read is allowed for Python startup. It does not recursively allow descendant reads; regression tests verify that private sibling files remain unreadable.
- Test labels never appear in public responses. Their private storage and the one-shot ledger survive service restarts.
- The demonstration is a single held-out test on one synthetic graph. It does not independently confirm the hypothesis mechanism or establish population-level uncertainty.
