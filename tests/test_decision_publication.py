import json
import sys
from types import SimpleNamespace

import pytest

from summit.recipes.decision.config import DecisionHFPushConfig
from summit.recipes.decision import publication
from summit.recipes.decision.publication import publish_artifact


@pytest.fixture
def fake_hub(monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: None,
        CommitOperationAdd=lambda **kw: SimpleNamespace(kind='add', **kw),
        CommitOperationDelete=lambda **kw: SimpleNamespace(kind='delete', **kw)))


class API:
    def __init__(self, private=True):
        self.private = private
        self.commits = []

    def create_repo(self, **kw):
        pass

    def model_info(self, repo):
        return SimpleNamespace(private=self.private, sha='old-revision', siblings=[
            SimpleNamespace(rfilename=f) for f in ('.gitattributes', 'complete.json', 'stale.json', 'model.bin')])

    def create_commit(self, **kw):
        self.commits.append(kw)
        return SimpleNamespace(oid='new-revision')


def test_publication_is_one_guarded_snapshot_commit(tmp_path, fake_hub):
    (tmp_path / 'model.bin').write_bytes(b'new weights')
    (tmp_path / 'manifest.json').write_text('{}')
    (tmp_path / 'failure.json').write_text('{}')
    (tmp_path / 'complete.json').write_text('{"stale":true}')
    api = API()
    complete = {'status':'completed', 'reload_verified':True}
    result = publish_artifact(tmp_path, DecisionHFPushConfig(repo='test/slot',replace=True), complete, api=api)
    assert result == 'new-revision' and len(api.commits) == 1
    commit = api.commits[0]
    assert commit['parent_commit'] == 'old-revision'
    ops = commit['operations']
    assert {o.path_in_repo for o in ops if o.kind == 'delete'} == {'stale.json'}
    assert {o.path_in_repo for o in ops if o.kind == 'add'} == {'model.bin','manifest.json','complete.json'}
    assert json.loads(next(o.path_or_fileobj for o in ops if o.path_in_repo == 'complete.json')) == complete


def test_publication_requires_explicit_replacement_and_matching_visibility(tmp_path, fake_hub):
    api = API()
    with pytest.raises(ValueError, match='archive it locally'):
        publish_artifact(tmp_path, DecisionHFPushConfig(repo='test/slot'), {}, api=api)
    assert not api.commits
    with pytest.raises(ValueError, match='visibility'):
        publish_artifact(tmp_path, DecisionHFPushConfig(repo='test/slot',replace=True), {}, api=API(private=False))


def test_large_publication_stages_small_diagnostics_before_weights(tmp_path, fake_hub, monkeypatch):
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model/model.safetensors').write_bytes(b'weights')
    (tmp_path / 'evaluation.json').write_text('{}')
    (tmp_path / 'predictions.jsonl').write_text('{}\n')
    monkeypatch.setattr(publication, '_EVIDENCE_STAGE_THRESHOLD_BYTES', 0)
    api = API()
    publish_artifact(tmp_path, DecisionHFPushConfig(repo='test/slot', replace=True),
                     {'status': 'completed'}, api=api)
    assert len(api.commits) == 2
    staged, final = api.commits
    staged_paths = {o.path_in_repo for o in staged['operations']}
    assert len(staged_paths) == 2
    assert all(path.startswith('_summit_attempts/') for path in staged_paths)
    assert all(not path.endswith('.safetensors') for path in staged_paths)
    assert final['parent_commit'] == 'new-revision'
    assert staged_paths <= {o.path_in_repo for o in final['operations'] if o.kind == 'delete'}
