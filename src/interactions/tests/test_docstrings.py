"""Documentation-coverage guard.

Every field of every public spec class and every parameter of every public
function (entry points + encoding builders) must have an entry in the numpy
``Parameters`` section of its docstring -- either on its own line
(``name:``) or in a grouped entry (``name_a, name_b:``). This is what feeds
Pylance's signature help, so a missing entry is a UX regression.

Run under pytest (``pytest tests/``) or standalone (``python tests/test_docstrings.py``).
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]  # Repo root == package root.

PUBLIC_CLASSES = [
    "Encoding", "Layout", "Style", "Regression", "SaveSpec",
    "FingerprintStyle", "PlotResult",
]
PUBLIC_FUNCTION_MODULES = ["plot/plot.py", "plot/encodings.py"]


def _documented(name: str, doc: str) -> bool:
    """True if ``name`` appears as a (possibly grouped) Parameters entry."""
    return bool(re.search(rf"^    [\w, ]*\b{name}\b[\w, ]*:", doc or "", re.M))


def _iter_public_functions():
    """Yield (module, FunctionDef) for every public top-level function."""
    for mod in PUBLIC_FUNCTION_MODULES:
        tree = ast.parse((PKG_ROOT / mod).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                yield mod, node


def test_spec_classes_document_every_field():
    from mypackage.plot import specs
    problems = []
    for cname in PUBLIC_CLASSES:
        cls = getattr(specs, cname)
        doc = cls.__doc__ or ""
        missing = [f.name for f in dataclasses.fields(cls) if not _documented(f.name, doc)]
        if missing:
            problems.append(f"{cname}: {missing}")
    assert not problems, "Undocumented dataclass fields: " + "; ".join(problems)


def test_public_functions_document_every_parameter():
    problems = []
    for mod, fn in _iter_public_functions():
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg != "self"]
        doc = ast.get_docstring(fn) or ""
        # ast.get_docstring dedents: grouped/own-line entries start at column 0.
        missing = [p for p in params
                   if not re.search(rf"^[\w, ]*\b{p}\b[\w, ]*:", doc, re.M)]
        if missing:
            problems.append(f"{mod}:{fn.name}: {missing}")
    assert not problems, "Undocumented function parameters: " + "; ".join(problems)


if __name__ == "__main__":  # Standalone mode (no pytest needed).
    import sys
    sys.path.insert(0, str(PKG_ROOT.parent))  # Make `import mypackage` work.
    test_spec_classes_document_every_field()
    test_public_functions_document_every_parameter()
    print("docstring coverage OK")
