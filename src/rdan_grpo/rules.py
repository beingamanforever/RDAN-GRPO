"""Deterministic hard-rubric checkers: RubricHub routes and sandboxed HIR type4 rules."""

from __future__ import annotations

import ast
import math
import multiprocessing
import numbers
import os
import re
from collections import Counter
from dataclasses import dataclass
from queue import Empty
from typing import Any, Callable, Mapping

RUBRICHUB_FUNCTIONS = frozenset({"CommaChecker", "LetterFrequencyChecker"})
RELATIONS = frozenset({"at least", "less than"})

# Type4 rules are Python supplied by the dataset, so they run in a killable child with no
# builtins beyond these and only these imports. Names are deliberately not allowlisted: an
# allowlist fitted to the rules seen so far rejects valid unseen ones during training.
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "zip": zip,
}
SAFE_IMPORTS = {"re": re, "collections": Counter}


class RuleError(ValueError):
    """Raised when a rule request violates the checker contract."""


@dataclass(frozen=True)
class RuleResult:
    """One deterministic rule outcome. ``valid`` is false when the checker itself failed."""

    valid: bool
    passed: bool
    error: str | None


def evaluate_rubrichub_rule(function: Any, response: Any, parameters: Any) -> RuleResult:
    """Evaluate one RubricHub checker route."""

    try:
        if not isinstance(function, str) or function not in RUBRICHUB_FUNCTIONS:
            raise RuleError("unsupported_rule_route")
        if not isinstance(response, str):
            raise RuleError("response_must_be_string")
        return RuleResult(True, _CHECKERS[function](response, parameters), None)
    except Exception as error:
        return RuleResult(False, False, str(error) if isinstance(error, RuleError) else "rule_evaluation_failed")


def evaluate_python_rule(code: str, instruction: str, response: str, timeout_seconds: float = 2.0) -> RuleResult:
    """Run a dataset-supplied ``check_following`` rule in an isolated child process."""

    try:
        validate_python_rule(code)
    except RuleError as error:
        return RuleResult(False, False, str(error))
    context = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_run_rule, args=(code, instruction, response, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.kill()
        process.join()
        return RuleResult(False, False, "timeout")
    try:
        value = queue.get_nowait()
    except Empty:
        return RuleResult(False, False, "rule_process_died")
    if not isinstance(value, bool):
        return RuleResult(False, False, str(value))
    return RuleResult(True, value, None)


def validate_python_rule(code: str) -> ast.Module:
    """Parse a type4 rule and reject dunder access and unapproved imports."""

    if not isinstance(code, str) or not code.strip():
        raise RuleError("rule_source_empty")
    if "__" in code:
        raise RuleError("rule_source_uses_dunder")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise RuleError(f"rule_syntax_invalid:{error.msg}") from error
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "check_following":
        raise RuleError("rule_must_define_one_check_following")
    if any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in tree.body):
        raise RuleError("rule_module_has_top_level_statement")
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name not in SAFE_IMPORTS for alias in node.names):
            raise RuleError("rule_imports_unapproved_module")
        if isinstance(node, ast.ImportFrom) and node.module not in SAFE_IMPORTS:
            raise RuleError("rule_imports_unapproved_module")
    if [argument.arg for argument in functions[0].args.args] != ["instruction", "response"]:
        raise RuleError("check_following_signature_invalid")
    return tree


def comma_checker(response: str, parameters: Any) -> bool:
    """Return whether the response contains no ASCII comma."""

    if not isinstance(parameters, Mapping) or parameters:
        raise RuleError("comma_parameters_must_be_empty")
    return "," not in response


def letter_frequency_checker(response: str, parameters: Any) -> bool:
    """Check one ASCII letter count against the requested relation."""

    normalized = normalize_letter_parameters(parameters)
    count = Counter(response.lower())[normalized["letter"]]
    if normalized["let_relation"] == "less than":
        return count < normalized["let_frequency"]
    return count >= normalized["let_frequency"]


def normalize_letter_parameters(parameters: Any) -> dict[str, str | int]:
    """Validate and normalize LetterFrequencyChecker parameters."""

    if not isinstance(parameters, Mapping) or set(parameters) != {"letter", "let_frequency", "let_relation"}:
        raise RuleError("letter_parameters_schema_invalid")
    letter = parameters["letter"]
    if not isinstance(letter, str) or len(letter) != 1 or not letter.isascii() or not letter.isalpha():
        raise RuleError("letter_must_be_one_ascii_alphabetic_character")
    relation = parameters["let_relation"]
    if not isinstance(relation, str) or relation not in RELATIONS:
        raise RuleError("let_relation_invalid")
    return {
        "letter": letter.lower(),
        "let_frequency": _frequency(parameters["let_frequency"]),
        "let_relation": relation,
    }


def _frequency(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RuleError("let_frequency_must_be_numeric_integer")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or not 1 <= numeric <= 20:
        raise RuleError("let_frequency_out_of_range")
    return int(numeric)


def _run_rule(code: str, instruction: str, response: str, queue: Any) -> None:
    """Child entry point: apply resource limits, execute the rule, and report one value."""

    try:
        os.environ.clear()
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
        except (ImportError, OSError, ValueError):
            pass
        tree = validate_python_rule(code)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        scope: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "re": re, "Counter": Counter}
        exec(compile(tree, "<type4-rule>", "exec"), scope)
        value = scope["check_following"](instruction, response)
        queue.put(value if isinstance(value, bool) else "rule_returned_non_boolean")
    except BaseException as error:
        queue.put(f"{type(error).__name__}: {error}")


_CHECKERS: dict[str, Callable[[str, Any], bool]] = {
    "CommaChecker": comma_checker,
    "LetterFrequencyChecker": letter_frequency_checker,
}
