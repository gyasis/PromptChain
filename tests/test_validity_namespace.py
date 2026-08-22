"""The promptchain.validity sub-package: namespace re-exports + OKF bundle conformance (issue #40)."""
import os
import re
from promptchain import validity as V


def test_namespace_reexports_both_layers():
    # procedural + statistical, one front door
    for name in ("technique_fired", "no_regression", "harness_faithful", "ValiditySuite",
                 "mcnemar", "wilson_ci", "holm_bonferroni", "compare_paired_binary", "okf_path"):
        assert hasattr(V, name), f"promptchain.validity missing {name}"
    assert callable(V.mcnemar) and callable(V.technique_fired)


def test_okf_path_exists():
    p = V.okf_path()
    assert os.path.isdir(p) and os.path.isfile(os.path.join(p, "index.md"))


def test_okf_bundle_is_spec_conformant():
    p = V.okf_path()
    # index.md and log.md MUST NOT have frontmatter (OKF spec §6/§7)
    for f in ("index.md", "log.md"):
        assert not open(os.path.join(p, f)).read().startswith("---"), f"{f} must have no frontmatter"
    # every concept file MUST have a non-empty `type:` in YAML frontmatter (OKF §4.1)
    concepts = 0
    for root, _, files in os.walk(p):
        for fn in files:
            if fn in ("index.md", "log.md") or not fn.endswith(".md"):
                continue
            txt = open(os.path.join(root, fn)).read()
            assert txt.startswith("---"), f"{fn} missing frontmatter"
            m = re.search(r"^type:\s*(\S+)", txt, re.M)
            assert m, f"{fn} missing required 'type:'"
            concepts += 1
    assert concepts >= 8  # workflow + 4 checks + 4 tests
