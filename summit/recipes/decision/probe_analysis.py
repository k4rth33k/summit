"""Compare matched stability artifacts against the predeclared exploratory gates."""

import argparse
import json
from pathlib import Path


def read_predictions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError("predictions must be nonempty with unique IDs")
    return {r['id']: r for r in rows}


def metrics(rows):
    rows = list(rows)
    return {"examples": len(rows), "correct": sum(r['choice'] == r['target'] for r in rows),
            "accuracy": sum(r['choice'] == r['target'] for r in rows) / len(rows),
            "nll": sum(r['nll'] for r in rows) / len(rows),
            "brier": sum(r['brier'] for r in rows) / len(rows),
            "groups": len({r['group_id'] for r in rows})}


def summarize(path):
    names = {"before": "predictions-before.jsonl", "after": "predictions.jsonl",
             "reversed_before": "predictions-reversed-before.jsonl", "reversed_after": "predictions-reversed.jsonl"}
    predictions = {name: read_predictions(path / filename) for name, filename in names.items()}
    before = predictions['before']
    for rows in predictions.values():
        if set(rows) != set(before) or any(rows[k]['target'] != before[k]['target'] for k in before):
            raise ValueError("evaluation IDs/targets are not matched")
    report = {name: {"overall": metrics(rows.values()), "sources": {
        source: metrics(r for r in rows.values() if r['source'] == source)
        for source in sorted({r['source'] for r in rows.values()})}} for name, rows in predictions.items()}
    report['order_disagreement'] = {stage: sum(predictions[stage][k]['choice'] != predictions['reversed_' + stage][k]['choice']
                                              for k in before) / len(before) for stage in ('before', 'after')}
    report['completion'] = json.loads((path / 'complete.json').read_text())
    return report, predictions


def compare(ce_path, kd_path):
    ce, ce_rows = summarize(ce_path)
    kd, kd_rows = summarize(kd_path)
    for stage in ('before', 'reversed_before'):
        for k, row in ce_rows[stage].items():
            other = kd_rows[stage].get(k)
            if not other or row['input_sha256'] != other['input_sha256'] or row['probabilities'] != other['probabilities']:
                raise ValueError("CE/KD baseline predictions differ; not a matched comparison")
        if set(ce_rows[stage]) != set(kd_rows[stage]):
            raise ValueError("CE/KD evaluation IDs differ")
    gates = {"reload_verified": all(r['completion'].get('reload_verified') for r in (ce, kd))}
    for prefix in ('', 'reversed_'):
        before, after = kd[prefix+'before'], kd[prefix+'after']
        gates[prefix+'no_accuracy_decline'] = after['overall']['correct'] >= before['overall']['correct']
        gates[prefix+'no_nll_increase'] = after['overall']['nll'] <= before['overall']['nll']
        gates[prefix+'source_stability'] = all(after['sources'][s]['accuracy'] >= before['sources'][s]['accuracy'] - 3/64
                                              for s in before['sources'])
    before, after = kd['before']['overall'], kd['after']['overall']
    gates['kd_not_worse_than_ce_accuracy'] = after['correct'] >= ce['after']['overall']['correct']
    gates['kd_not_worse_than_ce_nll'] = after['nll'] <= ce['after']['overall']['nll']
    gates['positive_learning_signal'] = after['correct'] >= before['correct'] and (
        after['correct'] >= before['correct'] + 2 or after['nll'] <= .95 * before['nll'])
    return {"ce": ce, "kd": kd, "gates": gates, "technical_scaling_gate_passed": all(gates.values()),
            "scope": "exploratory balanced development probe; not a final test, parity, or significance claim",
            "budget_gate": "Re-estimate full collection against the latest approved total and remaining spend; quality gates do not authorize extra spending."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ce', type=Path, required=True)
    parser.add_argument('--kd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.ce, args.kd)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
