# Implementation roadmap

## Working slice — deterministic pair experiments

Synthetic pair-held-out datasets, baseline training, ego-path ablation, budgeted workers, resumability, and development reports. Small dense CPU graphs only.

## Working layer — planning and isolated final scoring

Structured offline/OpenAI hypothesis planning, deterministic evidence critique, private-volume evaluator, macOS worker sandbox, frozen ensemble selection, and idempotent one-shot scoring. One bounded hypothesis cycle; no adaptive replanning or arbitrary code generation.

## Next — complete the experiment foundation

1. Add a capacity-matched ego-path control and independent synthetic graph draws across neighborhood regimes.
2. Define file-backed DatasetManifest and TaskSpec contracts for real datasets; validate context/group and provenance rules.
3. Add the node-classification task adapter with explicit transductive/inductive protocols.
4. Extend the tested macOS isolation backend to Linux and add authenticated remote evaluator deployment. Retain mutation tests showing test-label changes cannot affect search artifacts.
5. Introduce frozen method selection and a separate confirmation stage; add memory instrumentation and limits.

## Then — bounded research loop

Structured hypothesis plans and approved configuration interventions; provider adapter; evidence-linked critique; explicit rejected/inconclusive outcomes. Only after infrastructure is trusted: generated model code in an actual sandbox. Reuse the deterministic training/evaluation engine without an LLM dependency.

## Later — system evaluation

Compare fixed-budget search strategies on datasets withheld from system development. Measure invalid runs, reproducibility, cost, and independently confirmed hypothesis quality. Review related work before claiming novelty.
