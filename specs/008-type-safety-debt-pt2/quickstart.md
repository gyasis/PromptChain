# Quickstart: Verify 008-type-safety-debt-pt2 Fixes

**Purpose**: Step-by-step commands to verify each fix locally.
**Prereq**: Python 3.12+, mypy 1.7+, pytest installed in the environment.

---

## Setup

```bash
cd /home/gyasis/Documents/code/PromptChain
git checkout 008-type-safety-debt-pt2
pip install -e ".[dev]"
```

---

## Step 1: Capture baseline error counts (before any fixes)

```bash
python -m mypy promptchain/utils/strategies/state_agent.py --ignore-missing-imports 2>&1 | grep "error:" | wc -l
# Expected: 82

python -m mypy promptchain/cli/tui/app.py --ignore-missing-imports 2>&1 | grep "error:" | wc -l
# Expected: 63

python -m mypy promptchain/utils/promptchaining.py --ignore-missing-imports 2>&1 | grep "error:" | wc -l
# Expected: 32

python -m mypy promptchain/patterns/executors.py --ignore-missing-imports 2>&1 | grep "executors.py.*error:" | wc -l
# Expected: ~31
```

---

## Step 2: Verify each file fix independently

After fixing state_agent.py:
```bash
python -m mypy promptchain/utils/strategies/state_agent.py --ignore-missing-imports 2>&1 | grep "error:"
# Expected: (empty — 0 errors)
```

After fixing app.py:
```bash
python -m mypy promptchain/cli/tui/app.py --ignore-missing-imports 2>&1 | grep "app.py.*error:"
# Expected: (empty — 0 errors)
```

After fixing promptchaining.py:
```bash
python -m mypy promptchain/utils/promptchaining.py --ignore-missing-imports 2>&1 | grep "promptchaining.py.*error:"
# Expected: (empty — 0 errors)
```

After fixing executors.py:
```bash
python -m mypy promptchain/patterns/executors.py --ignore-missing-imports 2>&1 | grep "executors.py.*error:"
# Expected: (empty — 0 errors)
```

---

## Step 3: Verify total project error reduction

```bash
python -m mypy promptchain/ --ignore-missing-imports 2>&1 | grep "Found.*error" | tail -1
# Expected: Found NNN errors in ... files  (NNN < 220, down from 329)
```

---

## Step 4: Regression check

```bash
pytest --tb=short -q 2>&1 | tail -10
# Expected: same pass/fail ratio as pre-sprint baseline
# (pre-existing failures in tests/unit/patterns/ are acceptable)
```

---

## Step 5: Linting

```bash
black --check promptchain/utils/strategies/state_agent.py \
              promptchain/cli/tui/app.py \
              promptchain/utils/promptchaining.py \
              promptchain/patterns/executors.py

isort --check promptchain/utils/strategies/state_agent.py \
              promptchain/cli/tui/app.py \
              promptchain/utils/promptchaining.py \
              promptchain/patterns/executors.py
# Expected: exit 0 (no formatting issues)
```

If black or isort report issues, run without `--check` to auto-fix:
```bash
black promptchain/utils/strategies/state_agent.py promptchain/cli/tui/app.py \
      promptchain/utils/promptchaining.py promptchain/patterns/executors.py
isort promptchain/utils/strategies/state_agent.py promptchain/cli/tui/app.py \
      promptchain/utils/promptchaining.py promptchain/patterns/executors.py
```

---

## Step 6: Record results

Write results to `specs/008-type-safety-debt-pt2/test-results.md` in this format:

```markdown
# Test Results: 008-type-safety-debt-pt2

**Date**: YYYY-MM-DD
**Branch**: 008-type-safety-debt-pt2

## Mypy Results

| File | Before | After |
|------|--------|-------|
| state_agent.py | 82 | 0 |
| app.py | 63 | 0 |
| promptchaining.py | 32 | 0 |
| executors.py | 31 | 0 |
| **Total project** | **329** | **NNN** |

## Pytest Results

- Pre-sprint passing tests: NNN
- Post-sprint passing tests: NNN
- New failures introduced: 0

## Linting

- black: PASS
- isort: PASS
```
