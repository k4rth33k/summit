"""Fetch pinned public source artifacts; no repository code is executed."""

import argparse
import hashlib
import json
from pathlib import Path

import httpx


DOWNLOADS = {
    "clinc150": {"revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de", "repo": "clinc/oos-eval",
                 "files": {"data_full.json": "data/data_full.json", "LICENSE": "LICENSE", "README.md": "README.md"}},
    "contractnli": {"revision": "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9", "repo": "stanfordnlp/contract-nli",
                    "files": {"contract-nli.zip": "resources/contract-nli.zip", "LICENSE": "LICENSE", "README.md": "README.md"}},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=DOWNLOADS, default=list(DOWNLOADS))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for source in args.sources:
            spec = DOWNLOADS[source]
            directory = args.output / source
            directory.mkdir()
            for name, relative in spec["files"].items():
                url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['revision']}/{relative}"
                response = client.get(url)
                response.raise_for_status()
                (directory / name).write_bytes(response.content)
                manifest[f"{source}/{name}"] = {"url": url, "revision": spec["revision"],
                    "sha256": hashlib.sha256(response.content).hexdigest(), "bytes": len(response.content)}
                print(f"fetched {source}/{name}: {len(response.content)} bytes", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
