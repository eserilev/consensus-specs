#!/usr/bin/env python3
"""
Check that spec functions stay inside the safe Python subset.

The subset keeps every state write explicit, so tools can translate the spec into a language
without shared mutable references (for example a proof assistant). The rules:

- A local name that refers to part of a root object (``state``, ``store``, ...) is read-only.
  Write through the full path instead: ``state.validators[index].exit_epoch = epoch``, not
  ``validator = state.validators[index]`` followed by ``validator.exit_epoch = epoch``.
- No nested function definitions.
- Only root parameters change. A helper in ``MUTATING_HELPERS`` can change a list parameter,
  because every caller passes a path from a root object.

To accept a reviewed exception, end the line with ``# safe-python: allow``.
"""

import ast
import re
import sys
from pathlib import Path

# Parameters that a spec function is allowed to change.
ROOT_PARAMS = {"state", "store", "fcr_store", "seen", "block"}

# Helpers that change a list parameter. Every caller passes a path from a root object.
MUTATING_HELPERS = {"set_or_append_list"}

# Methods that change the object they are called on.
MUTATING_METHODS = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "remove",
    "sort",
    "update",
}

ALLOW_MARKER = "# safe-python: allow"


def base_name(node):
    """Return the name at the root of an attribute or subscript chain."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def find_aliases(fn):
    """Return the local names that refer to part of a root object."""
    aliases = set()
    for _ in range(3):
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(
                node.value, (ast.Attribute, ast.Subscript)
            ):
                if base_name(node.value) in ROOT_PARAMS | aliases:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            aliases.add(target.id)
            if isinstance(node, ast.For):
                iterated = node.iter
                is_enumerate = (
                    isinstance(iterated, ast.Call)
                    and isinstance(iterated.func, ast.Name)
                    and iterated.func.id == "enumerate"
                )
                if (
                    isinstance(iterated, ast.Call)
                    and isinstance(iterated.func, ast.Name)
                    and iterated.func.id in ("enumerate", "zip", "reversed")
                ):
                    sources = iterated.args
                else:
                    sources = [iterated]
                if not any(
                    isinstance(source, (ast.Attribute, ast.Subscript, ast.Name))
                    and base_name(source) in ROOT_PARAMS | aliases
                    for source in sources
                ):
                    continue
                targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
                if is_enumerate:
                    # The index of `enumerate` is a plain integer.
                    targets = targets[1:]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            aliases.add(name.id)
    return aliases


def check_function(fn, block_lines, block_start, file_path):
    violations = []
    params = {arg.arg for arg in fn.args.args}
    aliases = find_aliases(fn)
    mutable_params = ROOT_PARAMS | (params if fn.name in MUTATING_HELPERS else set())

    def report(node, message):
        line = block_lines[node.lineno - 1] if node.lineno <= len(block_lines) else ""
        if ALLOW_MARKER in line:
            return
        violations.append((file_path, block_start + node.lineno, fn.name, message))

    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not fn:
            if isinstance(node, ast.Lambda):
                continue
            report(node, f"nested function `{node.name}`")
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if not isinstance(target, (ast.Attribute, ast.Subscript)):
                continue
            name = base_name(target)
            if name in aliases:
                report(node, f"write through local alias `{name}`; write through the full path")
            elif name in params and name not in mutable_params:
                report(node, f"write to parameter `{name}`")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in MUTATING_METHODS
        ):
            name = base_name(node.func.value)
            if name in aliases:
                report(node, f"`.{node.func.attr}()` on local alias `{name}`")
            elif name in params and name not in mutable_params:
                report(node, f"`.{node.func.attr}()` on parameter `{name}`")
    return violations


def check_file(file_path):
    violations = []
    text = Path(file_path).read_text(encoding="utf-8")
    for match in re.finditer(r"```python\n(.*?)```", text, re.S):
        block = match.group(1)
        block_start = text[: match.start()].count("\n") + 1
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        block_lines = block.split("\n")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                violations.extend(check_function(node, block_lines, block_start, file_path))
    return violations


def main():
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "specs").rglob("*.md"))
    violations = []
    for file_path in files:
        violations.extend(check_file(file_path))
    for file_path, line, function, message in violations:
        print(f"{Path(file_path).relative_to(root)}:{line}: {function}: {message}")
    if violations:
        print(f"\nFound {len(violations)} write(s) outside the safe Python subset.")
        print(f"See scripts/check_safe_python.py. Mark a reviewed exception with `{ALLOW_MARKER}`.")
        sys.exit(1)


if __name__ == "__main__":
    main()
