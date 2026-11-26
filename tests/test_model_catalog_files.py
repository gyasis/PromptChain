"""Catalog-file model picker: fixtures only, no network."""
import json
import re
from pathlib import Path

import pytest

from promptchain.cli import model_catalog as mc

CLOUD = {"models": [
    {"name": "alpha:cloud", "capabilities": ["completion", "thinking", "tools"], "context_length": 1048576},
    {"name": "vis:7b-cloud", "capabilities": ["completion", "vision"], "context_length": 65536},
    {"name": "emb:cloud", "capabilities": ["embedding"], "context_length": 512},
]}
LOCAL = {"host": "http://10.0.0.5:11434", "models": [
    {"name": "small:3b", "capabilities": ["completion", "tools"], "context_length": 32768},
    {"name": "vec:latest", "capabilities": ["embedding"], "context_length": 8192},
], "llamacpp_lane": {
    "source": "http://10.0.0.5:8799/llm/llamacpp/models",
    "models": [{"alias": "lane-a", "vision": True, "ctx": 8192}, {"alias": "lane-b", "vision": False, "ctx": None}],
}}


@pytest.fixture
def cat_dir(tmp_path, monkeypatch):
    (tmp_path / "models.json").write_text(json.dumps(CLOUD))
    (tmp_path / "local-models.json").write_text(json.dumps(LOCAL))
    monkeypatch.setenv(mc.CATALOG_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    return tmp_path


def test_entries_mapped(cat_dir):
    entries, problems = mc.load_catalog_files()
    assert problems == []
    by = {e.model_name: e for e in entries}
    assert set(by) == {"ollama_chat/alpha:cloud", "ollama_chat/vis:7b-cloud",
                       "ollama_chat/small:3b", "openai/lane-a", "openai/lane-b"}
    a = by["ollama_chat/alpha:cloud"]
    assert a.source == "ollama-cloud" and a.context_length == 1048576
    assert a.capabilities == ["completion", "thinking", "tools"]
    assert a.reasoning["supports"] is True
    assert by["ollama_chat/small:3b"].api_base == "http://10.0.0.5:11434"


def test_llamacpp_lane(cat_dir):
    by = {e.model_name: e for e in mc.load_catalog_files()[0]}
    lane = by["openai/lane-a"]
    assert lane.source == "llamacpp"
    assert lane.api_base == "http://10.0.0.5:8799/llm/llamacpp/v1"
    assert "vision" in lane.capabilities and lane.context_length == 8192
    assert by["openai/lane-b"].context_length is None


def test_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.CATALOG_DIR_ENV, str(tmp_path / "nope"))
    entries, problems = mc.load_catalog_files()
    assert entries == [] and len(problems) == 2
    msg = mc.no_catalog_message()
    assert "no model catalog found at" in msg and "type a provider/model string" in msg
    assert mc.build_catalog(include_local=False, include_cloud=False) == []


def test_malformed_files(tmp_path, monkeypatch):
    (tmp_path / "models.json").write_text("{not json")
    (tmp_path / "local-models.json").write_text(json.dumps([1, 2]))
    monkeypatch.setenv(mc.CATALOG_DIR_ENV, str(tmp_path))
    entries, problems = mc.load_catalog_files()
    assert entries == [] and all("unreadable" in p for p in problems)


def test_bad_rows_skipped(tmp_path, monkeypatch):
    (tmp_path / "models.json").write_text(json.dumps({"models": [None, {"capabilities": []}, {"name": "ok:cloud"}]}))
    monkeypatch.setenv(mc.CATALOG_DIR_ENV, str(tmp_path))
    names = [e.model_name for e in mc.load_catalog_files()[0]]
    assert names == ["ollama_chat/ok:cloud"]


def test_build_catalog_dedupes(cat_dir):
    cat = mc.build_catalog(include_local=False, include_cloud=False)
    assert len(cat) == len({e.model_name for e in cat}) == 5


def test_web_bridge_catalog_uses_files(cat_dir):
    pytest.importorskip("fastapi")
    from promptchain.server import web_bridge
    ids = {m["id"] for m in web_bridge._catalog()}
    assert "ollama_chat/alpha:cloud" in ids and "openai/lane-a" in ids


RETIRED = re.compile(r"gpt-4o|gpt-4\.1|gpt-3\.5|gemini-2\.5|gemini-1\.5|\bo3-mini\b|claude-3")


@pytest.mark.parametrize("path", ["promptchain/cli/model_catalog.py", "promptchain/server/web_bridge.py"])
def test_no_hardcoded_retired_models(path):
    root = Path(mc.__file__).resolve().parents[2]
    assert not RETIRED.search((root / path).read_text())


def test_no_curated_lists():
    assert not hasattr(mc, "fetch_curated_api")
    assert not [n for n in dir(mc) if n.startswith("_CURATED")]
