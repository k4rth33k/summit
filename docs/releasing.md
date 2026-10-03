# Release checklist

This checkout may contain private research history and large artifacts. Release
only the source inventory; never archive/upload the entire workspace directory.
Preparing a release does not authorize publishing it to GitHub, PyPI or HF.

## Inventory

Keep README.md, AGENTS.md, CLAUDE.md, docs/, runtime source, simulation build
inputs, maintained tests, preparation/evaluation scripts, dependency metadata,
the license and exactly these example jobs:

- `examples/deepswe_opd.yaml`
- `examples/decision_9b.yaml`
- `examples/specialize_9b.yaml`

Historical Markdown is preserved under ignored `notes/`. Private/experimental
YAMLs remain locally under ignored examples paths. Data, outputs, checkpoints,
research, credentials, virtual environments and caches must stay excluded.
Gitignore is an inclusion aid, **not** a secret scanner or a way to untrack
previously committed files. If importing into an existing repository, inspect
tracked files separately and explicitly review any untracking/removal.

```bash
python scripts/check_release.py
python scripts/check_release.py --list
```

The checker uses a temporary Git metadata directory, honors this checkout's
ignore rules, and does not initialize/change its Git history. It checks the
three-example allowlist, prohibited paths, symlinks and local Markdown links.
It does not read `.env`, inspect tokens or replace a dedicated secret audit.
Use `--export NEW_DIRECTORY` to create a fresh source-only copy for validation;
it refuses existing destinations. Review that copy before sharing it.

## Verification

1. Run the core suite, and the offline ML suite when installed. Record skips.
2. Run the README's static check and dry run from a clean source export. They
   should not require private files or cloud credentials.
3. Decision recipes must fail clearly if data/base assets are absent; after
   preparing those assets, verify schemas and dry-run bundles separately.
4. Run `uv lock --check`, then `uv build`. Inspect wheel and source-distribution
   contents for private paths, weights, credentials and extra example YAMLs.
   Source install is the documented deployment path; do not advertise a PyPI
   one-liner until wheel-only cloud-bundling support is qualified.
5. Build Phantora/runtime images and run explicit container integration tests
   when changing that path. Do not claim that unrun Docker/CUDA/cloud tests passed.
6. Review all example destinations, privacy settings, retries and budget limits.
   No personal HF repository or machine-specific base path should be included.
7. Check third-party attribution, model/data rights and package license. This
   source release does not grant rights to redistribute upstream datasets/models.
8. For publishing model results, separately prepare immutable public evidence,
   model cards and portable checkpoints. Local historical reports are not public
   reproducibility artifacts. Do not claim exact retraining without the frozen
   inputs/teacher cache or equivalent evidence.

Only initialize Git, commit, tag, push or publish when the owner requests it.
