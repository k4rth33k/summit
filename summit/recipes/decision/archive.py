"""Download a pinned HF checkpoint into a local archive and verify every file."""

import argparse
import hashlib
import json
from pathlib import Path


def verify_file(path, *, size, blob_id, lfs_sha256=None):
    if path.stat().st_size != size:
        raise ValueError(f"archive size mismatch: {path}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha256.update(chunk)
            git_blob.update(chunk)
    if lfs_sha256:
        if sha256.hexdigest() != lfs_sha256:
            raise ValueError(f"archive LFS hash mismatch: {path}")
    elif git_blob.hexdigest() != blob_id:
        raise ValueError(f"archive Git blob mismatch: {path}")
    return sha256.hexdigest()


def archive(repo, output, *, token, revision=None):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)
    info = api.model_info(repo, revision=revision, files_metadata=True)
    destination = Path(output).resolve() / repo.replace('/', '--') / info.sha
    destination.mkdir(parents=True, exist_ok=True)
    # Resuming only this immutable revision cannot mix checkpoints.
    snapshot_download(repo, revision=info.sha, token=token, local_dir=destination)
    files = {}
    for sibling in info.siblings:
        path = (destination / sibling.rfilename).resolve()
        if not path.is_relative_to(destination):
            raise ValueError('remote filename escapes archive directory')
        sha = verify_file(path, size=sibling.size, blob_id=sibling.blob_id,
                          lfs_sha256=sibling.lfs.sha256 if sibling.lfs else None)
        files[sibling.rfilename] = {'size': sibling.size, 'sha256': sha}
    manifest = {'repo': repo, 'revision': info.sha, 'private': info.private,
                'files': files, 'verified': True, 'bytes': sum(f['size'] for f in files.values()),
                'scope': 'complete snapshot of this revision; not Git history, discussions, or settings'}
    # Keep archival bookkeeping outside the downloaded model snapshot.
    with (destination.parent / f'{info.sha}.verified.json').open('w') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    return {'repo': repo, 'revision': info.sha, 'directory': str(destination),
            'bytes': manifest['bytes'], 'verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/hf-archives'))
    parser.add_argument('--revision')
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    args = parser.parse_args()
    from summit.env import load_env
    token = load_env(args.env_file).get('HF_TOKEN')
    if not token:
        parser.error('HF_TOKEN required')
    print(json.dumps(archive(args.repo, args.output, token=token, revision=args.revision), indent=2))


if __name__ == '__main__':
    main()
