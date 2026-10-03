"""Ensures ~/.dstack/server/config.yml contains a vast.ai backend with creds
from VAST_API_KEY (summit's server config merges, never clobbers)."""
import os, sys
from pathlib import Path
import yaml

cfg_path = Path.home() / ".dstack" / "server" / "config.yml"
cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {"projects": []}
projects = cfg.setdefault("projects", [])
main = next((p for p in projects if p.get("name") == "main"), None)
if main is None:
    main = {"name": "main", "backends": []}
    projects.append(main)
backends = main.setdefault("backends", [])
if any(b.get("type") == "vastai" for b in backends):
    print("vast.ai backend already configured")
    sys.exit(0)
key = os.environ.get("VAST_API_KEY")
if not key:
    print("VAST_API_KEY not set", file=sys.stderr)
    sys.exit(1)
backends.append({"type": "vastai", "creds": {"type": "api_key", "api_key": key}})
cfg_path.write_text(yaml.safe_dump(cfg))
print("vast.ai backend added")
