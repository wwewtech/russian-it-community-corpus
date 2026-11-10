"""Find code quality issues in ricc/ package."""
from __future__ import annotations

import ast
from pathlib import Path

# ast.Print doesn't exist in Python 3.8+ (removed), check for print() calls instead
PRINT_CALL = "print"

for py_file in Path('ricc').rglob('*.py'):
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f'{py_file}: Syntax error: {e}')
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            print(f'{py_file}:{node.lineno}: Bare except clause - catches all exceptions')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == PRINT_CALL:
            print(f'{py_file}:{node.lineno}: print() call - use logging instead')
