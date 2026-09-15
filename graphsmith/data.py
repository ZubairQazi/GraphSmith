"""Development-only synthetic data and strict pair-held-out validation."""
from dataclasses import dataclass
import hashlib
import json
import math
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    name: str = "synthetic-pairs"
    nodes: int = 80
    features: int = 8
    pairs: int = 600
    data_seed: int = 41
    neighbor_agreement: float = 0.15
    edge_probability: float = 0.12
    label_noise: float = 0.05
    seeds: tuple = (11, 23, 37, 53, 71)
    models: tuple = ("prior", "mlp", "gcn", "sage", "ego_gcn")
    hidden: int = 16
    epochs: int = 100
    patience: int = 15
    learning_rate: float = 0.01
    max_jobs: int = 25
    job_timeout_seconds: int = 60
    campaign_seconds: int = 600
    topology_visibility: str = "independent_background_transductive"
    split_policy: str = "pair_held_out"
    primary_metric: str = "macro_f1"
    planner: str = "offline"
    planner_model: str = ""
    max_llm_calls: int = 0

    @classmethod
    def parse(cls, raw):
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        if "topology_visibility" not in raw:
            raise ValueError("An explicit topology_visibility policy is required")
        raw = dict(raw)
        for key in ("seeds", "models"):
            if key in raw:
                raw[key] = tuple(raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        for key in ("nodes", "features", "pairs", "hidden", "epochs", "patience",
                    "max_jobs", "job_timeout_seconds", "campaign_seconds"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be nonempty text")
        if type(self.data_seed) is not int:
            raise ValueError("data_seed must be an integer")
        if not 8 <= self.nodes <= 1000 or not 2 <= self.features <= 128:
            raise ValueError("Toy engine supports 8–1000 nodes and 2–128 features")
        if not 20 <= self.pairs <= self.nodes * (self.nodes - 1) // 2:
            raise ValueError("pairs must be >=20 and fit the unique unordered pair universe")
        for key in ("neighbor_agreement", "edge_probability", "label_noise"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if type(self.learning_rate) not in (float, int) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not self.seeds or any(type(s) is not int for s in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain distinct integers")
        if self.planner not in {"offline", "openai", "codex"}:
            raise ValueError("planner must be offline, codex, or openai")
        if type(self.max_llm_calls) is not int or not 0 <= self.max_llm_calls <= 10:
            raise ValueError("max_llm_calls must be an integer in [0, 10]")
        if self.planner == "openai" and (not isinstance(self.planner_model, str) or not self.planner_model or self.max_llm_calls < 1):
            raise ValueError("OpenAI planning requires planner_model and at least one declared LLM call")
        if self.planner == "codex" and (not isinstance(self.planner_model, str) or self.max_llm_calls < 1):
            raise ValueError("Codex planning requires a string planner_model and at least one declared planning turn")
        allowed = {"prior", "mlp", "gcn", "sage", "ego_gcn"}
        if not self.models or not set(self.models) <= allowed or len(set(self.models)) != len(self.models):
            raise ValueError("models must be distinct supported model names")
        if self.topology_visibility != "independent_background_transductive":
            raise ValueError("Only independent_background_transductive topology is implemented")
        if self.split_policy != "pair_held_out" or self.primary_metric != "macro_f1":
            raise ValueError("Only pair_held_out splits and macro_f1 are implemented")


def generate(config):
    config.validate()
    rng = random.Random(config.data_seed)
    groups = [rng.randrange(2) for _ in range(config.nodes)]
    features = [[(2 * g - 1) + rng.gauss(0, 0.45)] +
                [rng.gauss(0, 1) for _ in range(config.features - 1)] for g in groups]
    universe = [(a, b) for a in range(config.nodes) for b in range(a + 1, config.nodes)]
    edges = []
    for a, b in universe:
        compatibility = config.neighbor_agreement if groups[a] == groups[b] else 1 - config.neighbor_agreement
        if rng.random() < config.edge_probability * compatibility:
            edges.extend([[a, b], [b, a]])
    rng.shuffle(universe)
    queries = []
    for i, (a, b) in enumerate(universe[:config.pairs]):
        target = int(groups[a] == groups[b])
        if rng.random() < config.label_noise:
            target = 1 - target
        queries.append({"id": f"pair-{a}-{b}", "source": a, "destination": b,
                        "target": target, "split": "train" if i < int(config.pairs * .7) else "validation"})
    bundle = {
        "manifest": {
            "dataset_id": config.name, "schema_version": 1,
            "source": "GraphSmith synthetic generator", "license": "CC0-1.0 (generated data)",
            "task": "unordered_pair_classification", "split_policy": config.split_policy,
            "topology_visibility": config.topology_visibility,
            "target_definition": "Same latent binary group, independently flipped with label_noise probability",
            "feature_provenance": "Noisy latent group signal in first coordinate; independent Gaussian remaining coordinates",
            "topology_provenance": "Background edges sampled from latent groups before query outcomes; no outcome-derived edges",
            "preprocessing_fit_scope": "none", "missing_label_policy": "All generated queries labeled; absent queries unknown",
            "evaluation": "development_validation_only", "primary_metric": "macro_f1",
            "threshold": 0.5, "threshold_policy": "fixed a priori", "directed": False,
            "generator": {"data_seed": config.data_seed, "neighbor_agreement": config.neighbor_agreement,
                          "edge_probability": config.edge_probability, "label_noise": config.label_noise},
        },
        "node_ids": [f"node-{i}" for i in range(config.nodes)],
        "node_features": features, "message_edges": edges, "queries": queries,
    }
    validate(bundle)
    return bundle


def validate(bundle):
    manifest = bundle["manifest"]
    if manifest.get("topology_visibility") != "independent_background_transductive":
        raise ValueError("Explicit supported topology-visibility policy required")
    if manifest.get("split_policy") != "pair_held_out" or manifest.get("directed") is not False:
        raise ValueError("Only unordered pair-held-out data supported")
    if manifest.get("evaluation") != "development_validation_only":
        raise ValueError("This engine accepts development data only")
    ids = bundle["node_ids"]
    features = bundle["node_features"]
    if not ids or len(set(ids)) != len(ids) or len(ids) != len(features):
        raise ValueError("Node IDs must be unique and align with features")
    width = len(features[0])
    if width == 0 or any(len(row) != width or not all(type(x) in (int, float) and math.isfinite(x) for x in row) for row in features):
        raise ValueError("Features must be a finite rectangular matrix")
    n = len(ids)
    seen_edges = set()
    for edge in bundle["message_edges"]:
        if len(edge) != 2 or any(type(v) is not int or not 0 <= v < n for v in edge):
            raise ValueError("Message edge endpoint is invalid")
        if tuple(edge) in seen_edges:
            raise ValueError("Duplicate message edge")
        seen_edges.add(tuple(edge))
    if any((b, a) not in seen_edges for a, b in seen_edges):
        raise ValueError("Undirected background edges must include both orientations")
    seen_ids, seen_pairs = set(), set()
    classes = {"train": set(), "validation": set()}
    for q in bundle["queries"]:
        a, b = q["source"], q["destination"]
        if any(type(v) is not int or not 0 <= v < n for v in (a, b)) or a == b:
            raise ValueError("Query endpoints are invalid")
        pair = tuple(sorted((a, b)))
        if q["id"] in seen_ids or pair in seen_pairs:
            raise ValueError("Duplicate/reversed query pair or query ID; potential split leakage")
        if q["split"] not in classes or type(q["target"]) is not int or q["target"] not in (0, 1):
            raise ValueError("Only binary train/validation targets are allowed")
        classes[q["split"]].add(q["target"])
        seen_ids.add(q["id"])
        seen_pairs.add(pair)
    for q in bundle.get("test_queries", []):
        if set(q) != {"id", "source", "destination"}:
            raise ValueError("Test query schema forbids labels and extra fields")
        a, b = q["source"], q["destination"]
        if any(type(v) is not int or not 0 <= v < n for v in (a, b)) or a == b:
            raise ValueError("Test query endpoints are invalid")
        pair = tuple(sorted((a, b)))
        if q["id"] in seen_ids or pair in seen_pairs:
            raise ValueError("Test query overlaps development queries")
        seen_ids.add(q["id"])
        seen_pairs.add(pair)
    if any(labels != {0, 1} for labels in classes.values()):
        raise ValueError("Both classes must occur in each development partition")
    return {"dataset_hash": digest(bundle), "split_hash": digest([(q["id"], q["split"]) for q in bundle["queries"]])}


def profile(bundle):
    hashes = validate(bundle)
    degree = [0] * len(bundle["node_ids"])
    for a, _ in bundle["message_edges"]:
        degree[a] += 1
    train = [q for q in bundle["queries"] if q["split"] == "train"]
    return {**hashes, "nodes": len(degree), "undirected_background_edges": len(bundle["message_edges"]) // 2,
            "isolated_nodes": degree.count(0), "mean_degree": sum(degree) / len(degree),
            "training_prevalence": sum(q["target"] for q in train) / len(train),
            "query_counts": {s: sum(q["split"] == s for q in bundle["queries"]) for s in ("train", "validation")},
            "scope": {"topology": "all permitted transductive background edges", "labels": "training queries only"}}
