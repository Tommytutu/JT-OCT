"""Sequential, cold-process validation of the three hard Table 3 cases."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'


def audit_tree(result, data_path):
    """Evaluate raw NumPy rows, independently of the solver's packed masks."""
    import numpy as np
    with np.load(data_path, allow_pickle=False) as data:
        x, y = data['X'], data['y']
    prediction = np.empty_like(y)
    labels = set(y.tolist())
    splits = 0
    depth = 0

    def visit(tree, rows, ancestors):
        nonlocal splits, depth
        depth = max(depth, len(ancestors))
        if 'label' in tree:
            assert tree['label'] in labels
            prediction[rows] = tree['label']
            return
        feature = tree['feature']
        assert 0 <= feature < x.shape[1] and feature not in ancestors
        splits += 1
        zero = x[rows, feature] == 0
        visit(tree['left'], rows[zero], ancestors + (feature,))
        visit(tree['right'], rows[~zero], ancestors + (feature,))

    visit(result['tree'], np.arange(len(y)), ())
    errors = int(np.count_nonzero(prediction != y))
    objective = errors / len(y) + result['manifest']['penalty'] * splits
    assert depth <= result['manifest']['depth']
    assert abs(objective - result['UB']) < 1e-10
    assert errors == result['metrics']['misclassified']
    assert splits == result['metrics']['split_nodes']
    assert result['LB'] <= objective + 1e-9
    if result['status'] == 'OPT':
        assert objective - result['LB'] <= 1e-7
    return dict(errors=errors, splits=splits, depth=depth, objective=objective,
                independent_tree_audit=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=600.)
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds <= 0:
        parser.error('positive repeats and seconds required')
    root = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for repeat in range(1, args.repeats + 1):
        for dataset, penalty in (('diabetic', 0.), ('diabetic', .01), ('transactions', 0.)):
            output = (args.output / f'{dataset}_D5_L{penalty:g}_r{repeat}.json').resolve()
            command = [sys.executable, str(root / 'benchmark_hard_case.py'),
                       '--dataset', dataset, '--penalty', str(penalty),
                       '--seconds', str(args.seconds), '--backend', 'auto',
                       '--options', '{"native_d3":true,"threads":8}', '--output', str(output)]
            start = time.perf_counter()
            print(f'START {output.name}', flush=True)
            with output.with_suffix('.console.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            process_wall = time.perf_counter() - start
            result = json.loads(output.read_text(encoding='utf-8'))
            result['process_wall_seconds'] = process_wall
            result['independent_audit'] = audit_tree(result, root / 'datasets' / (dataset + '.npz'))
            output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            row = dict(dataset=dataset, depth=5, penalty=penalty, repeat=repeat,
                       status=result['status'], solver_seconds=result['call_wall_seconds'],
                       process_seconds=process_wall, LB=result['LB'], UB=result['UB'],
                       **result['independent_audit'])
            results.append(row)
            (args.output / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(results)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
