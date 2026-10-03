#!/usr/bin/env python3
"""Audit/export the source inventory without reading credentials or changing Git."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit


EXAMPLES = {"examples/deepswe_opd.yaml", "examples/decision_9b.yaml", "examples/specialize_9b.yaml"}
PRIVATE_ROOTS = {"notes", "research", "data", "outputs", "artifacts", "checkpoints",
                 "summit-checks", "summit-dry-run", "runs", "dist", "build"}


def inventory(root):
    # A separate empty index also audits workspaces with no .git. Ignore rules
    # from the workspace apply; user-global exclusions must not hide problems.
    with tempfile.TemporaryDirectory(prefix="summit-release-git-") as temporary:
        git_dir = Path(temporary) / "repo.git"
        subprocess.run(["git", "init", "--bare", "--quiet", str(git_dir)], check=True,
                       capture_output=True)
        result = subprocess.run([
            "git", "--git-dir", str(git_dir), "--work-tree", str(root),
            "-c", "core.excludesFile=/dev/null", "ls-files", "--others",
            "--exclude-standard", "-z",
        ], cwd=root, check=True, capture_output=True)
    return sorted(result.stdout.decode().rstrip("\0").split("\0")) if result.stdout else []


def audit(root, names):
    errors = []
    files = set(names)
    if {name for name in names if name.startswith("examples/")} != EXAMPLES:
        errors.append("release must contain exactly the three supported example YAMLs")
    for name in names:
        path = Path(name)
        if (path.parts[0] in PRIVATE_ROOTS or any(part.startswith(".venv") for part in path.parts)
                or "__pycache__" in path.parts or path.suffix in {".safetensors", ".pt", ".pth", ".ckpt"}
                or (path.name.startswith(".env") and name != ".env.example")):
            errors.append(f"private/generated path in release: {name}")
        if (root / name).is_symlink():
            errors.append(f"release symlink requires explicit review: {name}")
    # Stop before opening any file when the inventory itself is unsafe.
    if errors:
        return errors
    for name in names:
        if not name.endswith(".md"):
            continue
        content = (root / name).read_text()
        content = re.sub(r"^```.*?^```[^\n]*", "", content, flags=re.MULTILINE | re.DOTALL)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc or not url.path:
                continue
            linked = ((root / name).parent / unquote(url.path)).resolve()
            if not linked.is_relative_to(root) or linked.relative_to(root).as_posix() not in files:
                errors.append(f"Markdown link outside release: {name} -> {target}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--export", type=Path, help="copy source inventory into a NEW directory")
    args = parser.parse_args()
    root = args.root.resolve()
    names = inventory(root)
    errors = audit(root, names)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    if args.export:
        destination = args.export.resolve()
        if destination.is_relative_to(root):
            parser.error("export must be outside the source workspace")
        destination.mkdir(parents=True, exist_ok=False)
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        print(f"Exported {len(names)} source files to {destination}")
    if args.list:
        print("\n".join(names))
    print(f"Release inventory OK: {len(names)} files, exactly three example recipes")


if __name__ == "__main__":
    main()
