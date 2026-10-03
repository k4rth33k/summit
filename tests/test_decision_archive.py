import hashlib

import pytest

from summit.recipes.decision.archive import verify_file


def test_archive_verifies_git_and_lfs_content(tmp_path):
    path = tmp_path / 'file'
    value = b'checkpoint contents'
    path.write_bytes(value)
    git_hash = hashlib.sha1(f'blob {len(value)}\0'.encode() + value).hexdigest()
    sha = hashlib.sha256(value).hexdigest()
    assert verify_file(path, size=len(value), blob_id=git_hash) == sha
    assert verify_file(path, size=len(value), blob_id='unused', lfs_sha256=sha) == sha
    with pytest.raises(ValueError, match='size mismatch'):
        verify_file(path, size=1, blob_id=git_hash)
    with pytest.raises(ValueError, match='Git blob mismatch'):
        verify_file(path, size=len(value), blob_id='bad')
    with pytest.raises(ValueError, match='LFS hash mismatch'):
        verify_file(path, size=len(value), blob_id=git_hash, lfs_sha256='bad')
