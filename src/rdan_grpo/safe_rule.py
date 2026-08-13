"""Restricted execution for pinned HIR type4 rule functions."""

from __future__ import annotations

import ast
import hashlib
import multiprocessing
import os
import re
from collections import Counter
from dataclasses import dataclass
from queue import Empty
from typing import Any, Iterable

ALLOWED_CALLS = {
    "Counter",
    "all",
    "bool",
    "len",
    "re.escape",
    "re.findall",
    "re.search",
    "re.split",
    "paragraph.split",
    "paragraph.strip",
    "response.split",
    "response.strip",
    "sentence.split",
    "'\\\\b{}\\\\b'.format",
}
ALLOWED_NODES = {
    ast.Add,
    ast.And,
    ast.Assign,
    ast.Attribute,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Eq,
    ast.For,
    ast.FormattedValue,
    ast.FunctionDef,
    ast.GeneratorExp,
    ast.Gt,
    ast.GtE,
    ast.If,
    ast.Import,
    ast.ImportFrom,
    ast.In,
    ast.JoinedStr,
    ast.List,
    ast.Load,
    ast.LtE,
    ast.Module,
    ast.Name,
    ast.Not,
    ast.NotEq,
    ast.NotIn,
    ast.Or,
    ast.Return,
    ast.Store,
    ast.Sub,
    ast.Subscript,
    ast.UnaryOp,
    ast.alias,
    ast.arg,
    ast.arguments,
    ast.comprehension,
}
ALLOWED_NAMES = {
    "Counter",
    "all",
    "bool",
    "function",
    "highlight_main_points",
    "index",
    "instruction",
    "key",
    "keyword",
    "keyword1",
    "keyword2",
    "keyword3",
    "keyword_counts",
    "keyword_pattern",
    "keywords",
    "len",
    "paragraph",
    "paragraphs",
    "re",
    "response",
    "response_counts",
    "sentence",
    "sentences",
    "solution_to_achieve",
    "word",
    "word_count",
    "words",
}
SAFE_BUILTINS = {"all": all, "bool": bool, "len": len}


class UnsafeRuleError(ValueError):
    """Raised when code or execution violates the frozen type4 contract."""


@dataclass(frozen=True)
class RuleResult:
    """One isolated rule outcome."""

    valid: bool
    value: bool | None
    error: str | None


def function_sha256(code: str) -> str:
    """Hash exact UTF-8 function source."""

    return hashlib.sha256(code.encode()).hexdigest()


def validate_rule(code: str) -> ast.Module:
    """Parse and allowlist the exact grammar observed in pinned HIR type4 rules."""

    if not isinstance(code, str) or not code or "__" in code:
        raise UnsafeRuleError("rule source is empty or contains a dunder token")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise UnsafeRuleError(f"rule syntax is invalid: {error.msg}") from error
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "check_following":
        raise UnsafeRuleError("rule must define exactly one check_following function")
    if any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in tree.body):
        raise UnsafeRuleError("rule module contains an unapproved statement")
    if any(not _approved_import(node) for node in imports):
        raise UnsafeRuleError("rule contains an unapproved import")
    arguments = functions[0].args
    if (
        [argument.arg for argument in arguments.args] != ["instruction", "response"]
        or arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg
        or arguments.kwarg
        or arguments.defaults
    ):
        raise UnsafeRuleError("check_following must accept exactly instruction and response")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise UnsafeRuleError(f"unapproved syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise UnsafeRuleError("private attributes are forbidden")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise UnsafeRuleError(f"unapproved name: {node.id}")
        if isinstance(node, ast.Call) and ast.unparse(node.func) not in ALLOWED_CALLS:
            raise UnsafeRuleError(f"unapproved call: {ast.unparse(node.func)}")
    return tree


def evaluate_rule(
    code: str,
    instruction: str,
    response: str,
    *,
    allowed_hashes: Iterable[str],
    timeout_seconds: float = 1.0,
) -> RuleResult:
    """Run one certified rule in a resource-limited child and fail closed."""

    digest = function_sha256(code)
    if digest not in set(allowed_hashes):
        return RuleResult(False, None, "uncertified_rule_hash")
    try:
        validate_rule(code)
    except UnsafeRuleError as error:
        return RuleResult(False, None, str(error))
    context = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_child, args=(code, instruction, response, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.kill()
        process.join()
        return RuleResult(False, None, "timeout")
    try:
        value = queue.get_nowait()
    except Empty:
        return RuleResult(False, None, "isolated_rule_failed")
    if not isinstance(value, bool):
        return RuleResult(False, None, str(value))
    return RuleResult(True, value, None)


def mutate_returns(code: str) -> str:
    """Negate every explicit return expression for certification mutation probes."""

    tree = validate_rule(code)

    class NegateReturns(ast.NodeTransformer):
        def visit_Return(self, node: ast.Return) -> ast.Return:  # noqa: N802
            if node.value is None:
                return node
            return ast.copy_location(ast.Return(ast.UnaryOp(ast.Not(), node.value)), node)

    mutated = ast.fix_missing_locations(NegateReturns().visit(tree))
    return ast.unparse(mutated)


def _approved_import(node: ast.Import | ast.ImportFrom) -> bool:
    if isinstance(node, ast.Import):
        return len(node.names) == 1 and node.names[0].name == "re" and node.names[0].asname is None
    return (
        node.module == "collections"
        and node.level == 0
        and len(node.names) == 1
        and node.names[0].name == "Counter"
        and node.names[0].asname is None
    )


def _child(code: str, instruction: str, response: str, queue: Any) -> None:
    try:
        os.environ.clear()
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
        except (ImportError, OSError, ValueError):
            pass
        tree = validate_rule(code)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        scope = {"__builtins__": SAFE_BUILTINS, "Counter": Counter, "re": re}
        exec(compile(tree, "<hir-type4-rule>", "exec"), scope)
        value = scope["check_following"](instruction, response)
        queue.put(value if isinstance(value, bool) else "non_boolean_result")
    except BaseException as error:
        queue.put(f"{type(error).__name__}: {error}")
