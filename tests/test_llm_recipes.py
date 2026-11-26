"""Every recipe under docs/llms/recipes/ must parse and import only real names."""
import ast
import importlib
import re
from pathlib import Path

import pytest

RECIPES = sorted((Path(__file__).resolve().parent.parent / "docs" / "llms" / "recipes").glob("*.md"))
_BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.S | re.M)

# Recipes whose python blocks do not yet satisfy the check: {file name: reason}.
KNOWN_BROKEN: dict = {
    "recipe-advanced-agentic.md": (
        "block 7 imports TUIAgenticStepProcessor from promptchain.utils.agentic_step_processor, "
        "which does not define it"
    ),
    "recipe-tool-calling-local.md": (
        "block 1 is not valid Python (a dictionary key without ':'); it is pseudo-code"
    ),
}


def _python_blocks(path: Path):
    return _BLOCK.findall(path.read_text(encoding="utf-8"))


def _promptchain_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "promptchain" or node.module.startswith("promptchain."):
                for alias in node.names:
                    yield node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "promptchain" or alias.name.startswith("promptchain."):
                    yield alias.name, None


def _params():
    for path in RECIPES:
        marks = []
        if path.name in KNOWN_BROKEN:
            marks.append(pytest.mark.xfail(reason=KNOWN_BROKEN[path.name], strict=True))
        yield pytest.param(path, id=path.name, marks=marks)


def test_recipes_are_present():
    assert RECIPES, "no recipes found under docs/llms/recipes/"


@pytest.mark.parametrize("path", _params())
def test_recipe_python_blocks_parse_and_imports_resolve(path):
    problems = []
    for i, block in enumerate(_python_blocks(path)):
        try:
            tree = ast.parse(block)
        except SyntaxError as exc:
            problems.append(f"block {i}: SyntaxError line {exc.lineno}: {exc.msg}")
            continue
        for module, name in _promptchain_imports(tree):
            try:
                mod = importlib.import_module(module)
            except Exception as exc:  # noqa: BLE001 - any import failure is a finding
                problems.append(f"block {i}: cannot import {module}: {exc!r}")
                continue
            if name is not None and name != "*" and not hasattr(mod, name):
                try:
                    importlib.import_module(f"{module}.{name}")
                except Exception:  # noqa: BLE001
                    problems.append(f"block {i}: {module} has no attribute {name!r}")
    assert not problems, f"{path.name}:\n" + "\n".join(problems)
