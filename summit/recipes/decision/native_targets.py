"""Encode completed native teacher answers as hard labels, not confidence scores."""

import argparse
import json
from pathlib import Path

import yaml

from .data import TeacherTarget, canonical_json, digest, read_records
from .reasoning_reference import parse_answer
from .teacher import TeacherConfig, load_rationales


def native_targets(records, config, reference, repairs=()):
    # Reuse strict model/input/repair-chain validation. No API calls occur here.
    rationales = load_rationales(reference, records, config, repair=list(repairs))
    targets = []
    for row in records:
        selected = None
        for directory in [reference, *repairs]:
            path = directory / f'response-{digest(row.id)}.json'
            if not path.exists():
                continue
            response = json.loads(path.read_text())
            choice = response['choices'][0]
            answer = parse_answer(choice, row.candidate_options)
            if answer is None or '</think>' not in choice['text']:
                continue
            rationale = choice['text'].rsplit('</think>',1)[0] + '</think>'
            if rationale != rationales[row.id]:
                raise ValueError('native answer does not match validated repair selection')
            selected = TeacherTarget(input_sha256=row.input_hash(), model=config.model,
                method='native_answer_hard_label',
                probabilities={option.id:float(option.id==answer) for option in row.candidate_options},
                metadata={'label_encoding':'one_hot_not_model_confidence', 'tokenizer':config.tokenizer,
                    'tokenizer_revision':config.tokenizer_revision, 'reference_directory':str(directory),
                    'response_sha256':digest(response), 'rationale_sha256':digest(rationale)})
            break
        if selected is None:
            raise ValueError('no completed native answer: '+row.id)
        targets.append({'id':row.id,'teacher':selected.model_dump(mode='json')})
    return targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f','--config',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--repair',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=TeacherConfig.model_validate(yaml.safe_load(args.config.read_text()))
    targets=native_targets(read_records(args.data),cfg,args.reference,args.repair)
    with args.output.open('x') as stream:
        stream.writelines(canonical_json(row)+'\n' for row in targets)
    print(json.dumps({'records':len(targets),'method':'native_answer_hard_label','api_requests':0}))


if __name__=='__main__':
    main()
