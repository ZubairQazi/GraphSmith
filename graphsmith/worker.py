"""Fixed-code training worker. No generated code or final-test data supported."""
import copy
from pathlib import Path
import random
import sys
import time

import signal

# Bound orphan workers even if the orchestrator is killed. POSIX/macOS CPU engine.
if __name__ == "__main__":
    signal.setitimer(signal.ITIMER_REAL, float(sys.argv[3]))

import torch
from torch import nn

from .data import validate
from .io import read, write
from .metrics import evaluate
from .models import PairModel, operators


def run(run_dir, job_id):
    run_dir = Path(run_dir)
    job = read(run_dir / "jobs" / job_id / "request.json")
    bundle = read(run_dir / "dataset.json")
    validate(bundle)
    config = read(run_dir / "config.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(job["seed"])
    random.seed(job["seed"])
    features = torch.tensor(bundle["node_features"], dtype=torch.float32)
    normalized, mean = operators(features, bundle["message_edges"])
    train = [q for q in bundle["queries"] if q["split"] == "train"]
    valid = [q for q in bundle["queries"] if q["split"] == "validation"]
    def pairs(rows):
        return torch.tensor([[q["source"], q["destination"]] for q in rows], dtype=torch.long)
    train_pairs, valid_pairs = pairs(train), pairs(valid)
    labels = torch.tensor([q["target"] for q in train], dtype=torch.float32)
    def predictions(probabilities):
        return [{"id": q["id"], "probability": float(p)} for q, p in zip(valid, probabilities, strict=True)]
    start = time.monotonic()
    history = []
    best_epoch = 0
    parameter_count = 0
    if job["model"] == "prior":
        prior = labels.mean().item()
        output = predictions([prior] * len(valid))
        write(run_dir / "jobs" / job_id / "prior.json", {"probability": prior})
    else:
        model = PairModel(job["model"], features.shape[1], config["hidden"])
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_score, best_state, stale = -1.0, None, 0
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(features, normalized, mean, train_pairs), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                rows = predictions(model(features, normalized, mean, valid_pairs).sigmoid().tolist())
            score = evaluate(valid, rows)["macro_f1"]
            history.append({"epoch": epoch, "training_loss": loss.item(), "validation_macro_f1": score})
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= config["patience"]:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            output = predictions(model(features, normalized, mean, valid_pairs).sigmoid().tolist())
        torch.save(best_state, run_dir / "jobs" / job_id / "checkpoint.pt")
    write(run_dir / "jobs" / job_id / "predictions.json", output)
    write(run_dir / "jobs" / job_id / "result.json", {
        "job_id": job_id, "model": job["model"], "seed": job["seed"],
        "metrics": evaluate(valid, output), "history": history, "selected_epoch": best_epoch,
        "parameter_count": parameter_count, "training_seconds": time.monotonic() - start,
        "validation_accesses": len(history) + 1,
        "selection_policy": "maximum validation macro-F1; earliest epoch breaks ties",
        "peak_memory": None, "peak_memory_note": "Not instrumented in this development slice",
    })


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
