"""Every recipe under docs/llms/recipes/ must parse and import only real names."""
import ast
import importlib
import re
import textwrap
import types
from pathlib import Path

import pytest

RECIPES = sorted((Path(__file__).resolve().parent.parent / "docs" / "llms" / "recipes").glob("*.md"))
_BLOCK = re.compile(r"^[ \t]*```(?:python|py)\b[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.S | re.M)

# Known-broken recipe blocks: (file name, block index, substring of the expected problem) -> reason.
# A listed block must still fail with that problem; any other problem in the same file fails the test.
KNOWN_BROKEN: dict = {}


def _python_blocks(path: Path):
    return [textwrap.dedent(b) for b in _BLOCK.findall(path.read_text(encoding="utf-8"))]


def _is_pc(name: str) -> bool:
    return name == "promptchain" or name.startswith("promptchain.")


def _import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # noqa: BLE001 - any import failure is a finding
        return None, repr(exc)


def _resolve_attr(mod, name: str):
    """Return the attribute or submodule `name` of module `mod`, else None."""
    if hasattr(mod, name):
        return getattr(mod, name)
    sub, _ = _import(f"{mod.__name__}.{name}")
    return sub


def _check_block(tree: ast.AST):
    """Return a list of problems for one parsed block."""
    problems = []
    bound = {}  # local name -> module object, for names bound to promptchain modules
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and _is_pc(node.module):
            mod, err = _import(node.module)
            if mod is None:
                problems.append(f"cannot import {node.module}: {err}")
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                obj = _resolve_attr(mod, alias.name)
                if obj is None:
                    problems.append(f"{node.module} has no attribute {alias.name!r}")
                elif isinstance(obj, types.ModuleType):
                    bound[alias.asname or alias.name] = obj
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_pc(alias.name):
                    continue
                mod, err = _import(alias.name)
                if mod is None:
                    problems.append(f"cannot import {alias.name}: {err}")
                elif alias.asname:
                    bound[alias.asname] = mod
                else:
                    bound[alias.name.split(".")[0]] = importlib.import_module(alias.name.split(".")[0])
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = []
        cur = node
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name) or cur.id not in bound:
            continue
        obj = bound[cur.id]
        walked = cur.id
        for attr in reversed(chain):
            if not isinstance(obj, types.ModuleType):
                break
            nxt = _resolve_attr(obj, attr)
            walked += "." + attr
            if nxt is None:
                msg = f"{walked}: module {obj.__name__} has no attribute {attr!r}"
                if msg not in seen:
                    seen.add(msg)
                    problems.append(msg)
                break
            obj = nxt
    return problems


def _recipe_problems(path: Path):
    """Yield (block index, problem) for every problem in a recipe."""
    for i, block in enumerate(_python_blocks(path)):
        try:
            tree = ast.parse(block)
        except SyntaxError as exc:
            yield i, f"SyntaxError line {exc.lineno}: {exc.msg}"
            continue
        for problem in _check_block(tree):
            yield i, problem


def test_recipes_are_present():
    assert RECIPES, "no recipes found under docs/llms/recipes/"


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.name)
def test_recipe_python_blocks_parse_and_imports_resolve(path):
    expected = {k: v for k, v in KNOWN_BROKEN.items() if k[0] == path.name}
    unexpected = []
    hit = set()
    for i, problem in _recipe_problems(path):
        for key in expected:
            if key[1] == i and key[2] in problem:
                hit.add(key)
                break
        else:
            unexpected.append(f"block {i}: {problem}")
    assert not unexpected, f"{path.name}:\n" + "\n".join(unexpected)
    stale = set(expected) - hit
    assert not stale, f"{path.name}: KNOWN_BROKEN entries no longer fail: {sorted(stale)}"
