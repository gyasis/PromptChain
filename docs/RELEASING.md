# Releasing / versioning

PromptChain uses **setuptools_scm** — the package version is derived from the latest **git tag**
(`pyproject.toml` → `[tool.setuptools_scm]`, `write_to = promptchain/_version.py`). There is no version
number to hand-edit; **creating a tag is the release.**

## Automatic (default)
The `.github/workflows/auto-version.yml` action runs on every merge to `main`, reads the **conventional
commits** since the last tag, and creates the next semver tag:

| commit prefix | bump | example |
|---|---|---|
| `feat:` | minor | v0.7.0 → v0.8.0 |
| `fix:` | patch | v0.7.0 → v0.7.1 |
| `feat!:` / `BREAKING CHANGE:` in body | major | v0.7.0 → v1.0.0 (pre-1.0: still a minor bump under semver 0.x) |
| `docs:` / `chore:` / `ci:` only | none | no tag created |

So: **write conventional commit messages** and versioning happens on merge. Between tags, setuptools_scm
reports a dev version like `0.7.1.devN+g<sha>`.

## Manual (catch-up or override)
```bash
git tag -a v0.8.0 origin/main -m "release notes"
git push origin v0.8.0
```
`v0.7.0` was cut manually as the catch-up baseline for the naked-callers + validity + OKF batch.
