#!/usr/bin/env python3
"""
Check that spec functions only change state in ways a translator can follow.

A translator that turns the spec into a language without shared mutable references (for example
a proof assistant) has to find the state path behind every write. These patterns are supported:

- A local name bound to a path from a root object, then written through:
  ``validator = state.validators[index]`` then ``validator.exit_epoch = epoch``.
- A local name bound to one of several root paths in different branches:
  ``epoch_participation = state.current_epoch_participation`` or the previous one.
- A loop variable over a root list: ``for index, validator in enumerate(state.validators)``.
- The ``for_ops`` helper: a nested function that only loops over a list and calls a function on
  each element.

These patterns are reported:

- A write through a name bound to the result of a spec function call, a comprehension, a
  conditional expression or a tuple unpack. The translator cannot tell which path it refers to.
- A path alias whose index expression changes after the alias is bound.
- Any other nested function.
- A change to a parameter that is not a root object.

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


def index_names(node):
    """Return the names that the index expressions of a path read."""
    names = set()
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Subscript):
            names |= {n.id for n in ast.walk(node.slice) if isinstance(n, ast.Name)}
        node = node.value
    return names


# Builtins that return a new object.
FRESH_BUILTINS = {"list", "sorted", "set", "dict", "tuple", "range", "reversed", "zip", "enumerate"}


def is_fresh_call(node):
    """A call that builds a new object: a type constructor, ``Type.empty()``, ``.copy()`` or a
    builtin such as ``list``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id[:1].isupper() or func.id in FRESH_BUILTINS
    if isinstance(func, ast.Attribute):
        if func.attr == "copy":
            return True
        return isinstance(func.value, ast.Name) and func.value.id[:1].isupper()
    return False


def depth(target):
    """The number of attribute or subscript steps in a write target."""
    steps = 0
    while isinstance(target, (ast.Attribute, ast.Subscript)):
        steps += 1
        target = target.value
    return steps


def is_for_ops(fn):
    """The ``for_ops`` helper: loop over the first parameter, call the second on each element."""
    params = [arg.arg for arg in fn.args.args]
    if len(params) < 2 or len(fn.body) != 1 or not isinstance(fn.body[0], ast.For):
        return False
    loop = fn.body[0]
    if not (isinstance(loop.iter, ast.Name) and loop.iter.id == params[0]):
        return False
    if len(loop.body) != 1 or not isinstance(loop.body[0], ast.Expr):
        return False
    call = loop.body[0].value
    return (
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == params[1]
    )


def classify_names(fn):
    """Split local names into path aliases and names the translator cannot resolve.

    Returns ``(aliases, unresolved, alias_indices)``. ``alias_indices`` maps each alias to the
    names that its path index expressions read.
    """
    roots = set(ROOT_PARAMS)
    aliases, unresolved, alias_indices = set(), set(), {}
    for _ in range(3):
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                value = node.value
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if (
                            isinstance(value, (ast.Attribute, ast.Subscript))
                            and base_name(value) in roots | aliases
                        ):
                            aliases.add(target.id)
                            alias_indices.setdefault(target.id, set()).update(index_names(value))
                        elif isinstance(value, ast.Name) and value.id in aliases:
                            aliases.add(target.id)
                        elif (
                            isinstance(value, ast.Call) and not is_fresh_call(value)
                        ) or isinstance(value, ast.IfExp):
                            unresolved.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for element in target.elts:
                            if isinstance(element, ast.Name):
                                unresolved.add(element.id)
            if isinstance(node, ast.For):
                iterated = node.iter
                is_enumerate = (
                    isinstance(iterated, ast.Call)
                    and isinstance(iterated.func, ast.Name)
                    and iterated.func.id == "enumerate"
                )
                source = iterated.args[0] if is_enumerate and iterated.args else iterated
                targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
                if is_enumerate:
                    targets = targets[1:]
                is_root_list = isinstance(source, (ast.Attribute, ast.Subscript, ast.Name)) and (
                    base_name(source) in roots | aliases
                )
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            if is_root_list:
                                aliases.add(name.id)
                            elif isinstance(source, ast.Call) and not is_fresh_call(source):
                                unresolved.add(name.id)
    return aliases, unresolved - aliases, alias_indices


def collections_of_root_parts(fn):
    """Names bound to a new list whose elements can be parts of a root object."""
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(
            node.value, (ast.ListComp, ast.GeneratorExp, ast.Call)
        ):
            reads_root = any(
                isinstance(n, ast.Name) and n.id in ROOT_PARAMS for n in ast.walk(node.value)
            )
            if reads_root:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
    return names


def check_function(fn, block_lines, block_start, file_path):
    violations = []
    params = {arg.arg for arg in fn.args.args}
    aliases, unresolved, alias_indices = classify_names(fn)
    mutable_params = ROOT_PARAMS | (params if fn.name in MUTATING_HELPERS else set())
    loop_targets = {
        name.id
        for node in ast.walk(fn)
        if isinstance(node, ast.For)
        for name in ast.walk(node.target)
        if isinstance(name, ast.Name)
    }
    collections = collections_of_root_parts(fn)
    # Line numbers of every assignment to each local name.
    assigned_lines = {}
    alias_lines = {}
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned_lines.setdefault(target.id, []).append(node.lineno)
                    if target.id in aliases:
                        alias_lines.setdefault(target.id, []).append(node.lineno)

    def report(node, message):
        line = block_lines[node.lineno - 1] if node.lineno <= len(block_lines) else ""
        if ALLOW_MARKER in line:
            return
        violations.append((file_path, block_start + node.lineno, fn.name, message))

    def index_changed(name, write_line):
        """An index name of the alias path is reassigned between the binding and the write."""
        bound = [line for line in alias_lines.get(name, []) if line <= write_line]
        if not bound:
            return set()
        start = max(bound)
        return {
            index
            for index in alias_indices.get(name, set()) - loop_targets
            if any(start < line < write_line for line in assigned_lines.get(index, []))
        }

    def check_write(node, target, name, what):
        if name in unresolved:
            report(node, f"{what} through `{name}`, which a translator cannot map to a state path")
        elif name in aliases:
            changed = index_changed(name, node.lineno)
            if changed:
                report(node, f"{what} through `{name}`, whose path index {sorted(changed)} changes")
        elif name in collections and target is not None and depth(target) >= 2:
            report(node, f"{what} to an element of `{name}`, which can be part of a root object")
        elif name in params and name not in mutable_params:
            report(node, f"{what} to parameter `{name}`")

    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            if not is_for_ops(node):
                report(node, f"nested function `{node.name}`")
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                check_write(node, target, base_name(target), "write")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in MUTATING_METHODS
        ):
            check_write(node, node.func.value, base_name(node.func.value), f"`.{node.func.attr}()`")
    return violations


def check_file(file_path):
    violations = []
    text = Path(file_path).read_text(encoding="utf-8")
    for match in re.finditer(r"```python\n(.*?)```", text, re.DOTALL):
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
        print(f"\nFound {len(violations)} state write(s) that a translator cannot follow.")
        print(f"See scripts/check_safe_python.py. Mark a reviewed exception with `{ALLOW_MARKER}`.")
        sys.exit(1)


if __name__ == "__main__":
    main()
