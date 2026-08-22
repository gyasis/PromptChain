"""OKFLoader tests (issue #42) — load & understand the real promptchain/validity/okf bundle,
and both injection modes (sequential file vs agentic folder/outline)."""
import os
from promptchain.utils.okf_loader import (
    OKFLoader, okf_step, okf_agentic_context, okf_reader_tool,
)

BUNDLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "promptchain", "validity", "okf")


def _loader():
    return OKFLoader(BUNDLE)


def test_scans_concepts_and_skips_reserved():
    ld = _loader()
    ids = ld.concepts()
    assert "validation-workflow" in ids
    assert "tests/mcnemar" in ids and "checks/technique-fired" in ids
    # reserved files are NOT concepts
    assert "index" not in ids and "log" not in ids


def test_frontmatter_type_parsed():
    c = _loader().get("tests/mcnemar")
    assert c is not None and c.type == "StatisticalTest"
    assert "McNemar" in c.title and c.description


def test_links_are_extracted():
    # the workflow concept links to other concepts (a link asserts a relationship)
    links = _loader().get("validation-workflow").links
    assert any("mcnemar" in l for l in links) and any("harness-faithful" in l for l in links)


def test_tree_and_outline_navigation():
    ld = _loader()
    tree = ld.tree()
    assert "tests" in tree and "checks" in tree
    outline = ld.outline()
    assert "`tests/mcnemar`" in outline and "[StatisticalTest]" in outline


def test_load_body_targeted():
    body = _loader().load("tests/mcnemar")
    assert "McNemar" in body and body and not body.startswith("---")   # body only, frontmatter stripped


def test_okf_step_sequential_injection():
    ld = _loader()
    instr = okf_step(ld, ["tests/mcnemar"], "decide if the delta is significant.")
    assert "McNemar" in instr and "decide if the delta is significant." in instr


def test_okf_agentic_context_outline_and_full():
    ld = _loader()
    outline_ctx = okf_agentic_context(ld, mode="outline")
    assert "okf_read(concept_id)" in outline_ctx and "tests/mcnemar" in outline_ctx
    full_ctx = okf_agentic_context(ld, ["tests/mcnemar"], mode="full")
    assert "McNemar" in full_ctx


def test_okf_reader_tool_on_demand():
    read = okf_reader_tool(_loader())
    assert "McNemar" in read("tests/mcnemar")
    assert "no concept" in read("does/not/exist")


def test_index_md_available():
    assert "Experiment Validity" in _loader().index_md()
