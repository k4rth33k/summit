"""Sequential matched CE/KD probe; publish both diagnostics and one selected model."""

import gc
import json
import os
from pathlib import Path
import shutil
import time

from .data import canonical_json
from .probe_analysis import compare


def selected_arm(report):
    # Predeclared artifact retention rule, not a test of statistical superiority.
    return max(('ce', 'kd'), key=lambda name: (
        report[name]['after']['overall']['accuracy'],
        -report[name]['after']['overall']['nll'], name == 'ce'))


def copy_link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def train_matched(cfg):
    import torch
    from .train import train
    from .publication import publish_artifact
    from .recipe import preflight

    report = preflight(cfg)
    if not report.passed:
        raise ValueError(report.format_text())
    output = cfg.resolve_path(cfg.summitConfig.output.directory)
    work = output.with_name(output.name + '-arms')
    if output.exists() or work.exists():
        raise FileExistsError('matched output/work directory already exists')
    output.mkdir(parents=True)
    work.mkdir()
    started = time.monotonic()
    completions = {}
    try:
        for name, objective in (('ce', 'candidate_ce'), ('kd', 'candidate_distillation')):
            child = cfg.model_copy(deep=True)
            child.training.comparison = 'single'
            child.objective.type = objective
            child.experiment_name += '-' + name
            child.summitConfig.output.directory = work / name
            child.summitConfig.output.hf_push = None
            print(canonical_json({'event': 'comparison_arm_started', 'arm': name}), flush=True)
            completions[name] = train(child)
            gc.collect()
            if cfg.training.device == 'cuda':
                torch.cuda.empty_cache()
        comparison = compare(work / 'ce', work / 'kd')
        chosen = selected_arm(comparison)
        comparison['checkpoint_selection'] = {
            'selected': chosen, 'rule': 'highest original-order development accuracy; then lowest NLL; CE on exact tie',
            'retention': 'one selected model plus both diagnostics; nonselected weights remain local to this job and are not published',
            'scope': 'artifact selection only; not a parity or statistical superiority claim'}
        for source in (work / chosen).rglob('*'):
            if source.is_file() and source.name not in ('complete.json', 'failure.json'):
                copy_link(source, output / source.relative_to(work / chosen))
        for name in ('ce', 'kd'):
            for source in (work / name).iterdir():
                if source.is_file() and source.suffix in ('.json', '.jsonl'):
                    copy_link(source, output / 'comparisons' / name / source.name)
        (output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        (output / 'comparison-config.json').write_text(cfg.model_dump_json(indent=2) + '\n')
        destination = cfg.summitConfig.output.hf_push
        complete = {**completions[chosen], 'hf_repo': destination.repo if destination else None,
                    'comparison': 'ce_kd', 'selected_arm': chosen,
                    'total_optimizer_steps': sum(c['steps'] for c in completions.values()),
                    'peak_allocated_bytes': max((c['peak_allocated_bytes'] for c in completions.values()
                                                 if c['peak_allocated_bytes'] is not None), default=None),
                    'elapsed_seconds': time.monotonic() - started,
                    'elapsed_scope': 'both training/evaluation/reload arms and packaging; excludes publication'}
        if destination:
            revision = publish_artifact(output, destination, complete)
            print(canonical_json({'event':'checkpoint_published', 'repo':destination.repo, 'revision':revision,
                                  'selected_arm':chosen}), flush=True)
        (output / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
        print(canonical_json({'event':'comparison_completed','selected_arm':chosen,
                              'gates':comparison['gates']}), flush=True)
        return complete
    except Exception as exc:
        (output / 'failure.json').write_text(json.dumps({'status':'failed','error_type':type(exc).__name__,
                                                        'message':str(exc)}, indent=2) + '\n')
        raise
