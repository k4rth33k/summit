"""Atomic publication into an explicitly managed checkpoint slot."""

from pathlib import Path
import uuid

from .data import canonical_json


_EVIDENCE_STAGE_THRESHOLD_BYTES = 1 << 30
_MAX_EVIDENCE_FILE_BYTES = 64 << 20


def _is_evidence(path: Path, output: Path) -> bool:
    relative = path.relative_to(output)
    return (
        path.stat().st_size <= _MAX_EVIDENCE_FILE_BYTES
        and relative.parts[0] not in ("model", "tokenizer")
        and path.suffix not in (".bin", ".safetensors")
    )


def publish_artifact(output, destination, complete, *, api=None):
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = api or HfApi()
    api.create_repo(repo_id=destination.repo, private=destination.private, exist_ok=True)
    info = api.model_info(destination.repo)
    if info.private != destination.private:
        raise ValueError('existing repository visibility differs from requested visibility')
    existing = {file.rfilename for file in info.siblings}
    if existing - {'.gitattributes'} and not destination.replace:
        raise ValueError('repository already contains files; archive it locally before enabling hf_push.replace')
    files = {str(path.relative_to(output)): path for path in sorted(output.rglob('*')) if path.is_file()
             and path.relative_to(output).parts[0] not in ('complete.json', 'failure.json')}
    parent_commit = info.sha
    staged_paths = []
    if sum(path.stat().st_size for path in files.values()) >= _EVIDENCE_STAGE_THRESHOLD_BYTES:
        # Preserve small diagnostics before beginning a multi-GB checkpoint commit.
        # They live outside the root checkpoint snapshot, so the prior root remains
        # valid if the weight upload fails. A successful final commit removes them.
        attempt = uuid.uuid4().hex[:12]
        evidence = {name: path for name, path in files.items() if _is_evidence(path, output)}
        staged_paths = [f'_summit_attempts/{attempt}/{name}' for name in evidence]
        if staged_paths:
            staged = api.create_commit(
                repo_id=destination.repo,
                operations=[CommitOperationAdd(path_in_repo=staged_name, path_or_fileobj=evidence[name])
                            for staged_name, name in zip(staged_paths, evidence)],
                parent_commit=parent_commit,
                commit_message='Stage Summit decision diagnostics',
            )
            parent_commit = staged.oid
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path) for name, path in files.items()]
    operations.append(CommitOperationAdd(path_in_repo='complete.json',
                                        path_or_fileobj=(canonical_json(complete) + '\n').encode()))
    if destination.replace:
        # Preserve HF's repository metadata; remove stale checkpoint sidecars.
        operations.extend(CommitOperationDelete(path_in_repo=name)
                          for name in sorted(existing - set(files) - {'complete.json', '.gitattributes'}))
    operations.extend(CommitOperationDelete(path_in_repo=name) for name in staged_paths)
    # One Git commit publishes all weights, diagnostics, and the success marker.
    # Upload failure leaves the previous snapshot intact. A moved HEAD fails the
    # optimistic concurrency guard instead of overwriting another publisher.
    commit = api.create_commit(repo_id=destination.repo, operations=operations,
                               parent_commit=parent_commit, commit_message='Publish Summit decision checkpoint')
    return commit.oid
