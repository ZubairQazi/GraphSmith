"""No-training inference of the frozen ensemble; executed without network access."""
from pathlib import Path
import signal
import sys
if __name__ == '__main__':
    signal.setitimer(signal.ITIMER_REAL, float(sys.argv[2]))
import torch

from .data import validate
from .io import read, write
from .models import PairModel, operators


def run(root):
    root = Path(root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    bundle = read(root / 'dataset.json')
    validate(bundle)
    selected, config = read(root / 'selection.json'), read(root / 'config.json')
    x = torch.tensor(bundle['node_features'], dtype=torch.float32)
    normalized, mean = operators(x, bundle['message_edges'])
    queries = bundle['test_queries']
    pairs = torch.tensor([[q['source'], q['destination']] for q in queries], dtype=torch.long)
    outputs = []
    for seed in selected['seeds']:
        job = root / 'jobs' / f"{selected['model']}-{seed}"
        if selected['model'] == 'prior':
            probabilities = torch.full((len(queries),), read(job / 'prior.json')['probability'])
        else:
            model = PairModel(selected['model'], x.shape[1], config['hidden'])
            model.load_state_dict(torch.load(job / 'checkpoint.pt', weights_only=True, map_location='cpu'))
            model.eval()
            with torch.no_grad():
                probabilities = model(x, normalized, mean, pairs).sigmoid()
        outputs.append(probabilities)
    probabilities = torch.stack(outputs).mean(0).tolist()
    write(root / 'test-predictions.json', [{'id': q['id'], 'probability': p} for q, p in zip(queries, probabilities, strict=True)])


if __name__ == '__main__':
    run(sys.argv[1])
