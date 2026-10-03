import json
import pytest


def test_matched_checkpoint_selection_is_predeclared_and_deterministic():
    from summit.recipes.decision.matched import selected_arm
    report = {name:{'after':{'overall':{'accuracy':.75,'nll':.8}}} for name in ('ce','kd')}
    assert selected_arm(report)=='ce'
    report['kd']['after']['overall']['nll']=.7
    assert selected_arm(report)=='kd'
    report['kd']['after']['overall']['accuracy']=.74
    assert selected_arm(report)=='ce'


def test_scaling_gate_rejects_regression_and_mismatched_baselines(tmp_path):
    from summit.recipes.decision.probe_analysis import compare
    ce, kd = tmp_path / 'ce', tmp_path / 'kd'
    for path in (ce, kd):
        path.mkdir()
        (path / 'complete.json').write_text(json.dumps({'reload_verified': True}))
        for suffix in ('-before', '', '-reversed-before', '-reversed'):
            rows = [{'id':str(i), 'group_id':str(i), 'source':'fixture', 'input_sha256':str(i),
                     'choice':'x', 'target':'x', 'probabilities':{'x':.8,'y':.2},
                     'nll':.2 if 'before' in suffix else .18, 'brier':.1} for i in range(128)]
            (path / f'predictions{suffix}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert compare(ce,kd)['technical_scaling_gate_passed']
    rows[0]['choice'] = 'y'
    (kd / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert not compare(ce,kd)['technical_scaling_gate_passed']
    (kd / 'predictions-before.jsonl').write_text(json.dumps({**rows[0], 'input_sha256':'changed'})+'\n')
    with pytest.raises(ValueError, match='matched'):
        compare(ce,kd)
    (kd / 'predictions-before.jsonl').write_text((ce / 'predictions-before.jsonl').read_text())
    (kd / 'predictions-reversed-before.jsonl').write_text(json.dumps({**rows[0], 'input_sha256':'changed'})+'\n')
    with pytest.raises(ValueError, match='matched'):
        compare(ce,kd)
