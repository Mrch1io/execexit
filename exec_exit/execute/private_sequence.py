"""LC private-label completion, preserving the frozen instance and call order.

The frozen executor remains unchanged. Only its three loop exits and result
parser are adapted; candidate code still runs exclusively in the same sandbox.
"""
from __future__ import annotations

import ast
from collections import Counter
import copy
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path

from exec_exit.execute import leetcode_call as lc

FROZEN_ADAPTER_SHA = "501594d934fb63f82e1bc2dc3cbb570655da67f4109afea6f4bcb7b095455b43"
FAILURES = {"WRONG_ANSWER", "TIMEOUT_CPU", "TIMEOUT_WALL", "RUNTIME_ERROR", "OUTPUT_LIMIT"}


def check_source():
    if hashlib.sha256(Path(lc.__file__).read_bytes()).hexdigest() != FROZEN_ADAPTER_SHA:
        raise ValueError("Frozen LC adapter changed")


def continued_source(retained):
    check_source()
    tree = ast.parse(lc.official_subset())
    grade = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "grade_call_based")
    loop = next(n for n in grade.body if isinstance(n, ast.For))
    changes = []

    class ContinueIgnored(ast.NodeTransformer):
        def visit_Return(self, node):
            if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 2:
                raise ValueError("Unexpected official loop return")
            results, metadata = node.value.elts
            if not isinstance(results, ast.Name) or results.id != "all_results" or not isinstance(metadata, ast.Dict):
                raise ValueError("Official failure return changed")
            changes.append(next(ast.literal_eval(v) for k, v in zip(metadata.keys, metadata.values)
                                if isinstance(k, ast.Constant) and k.value == "error_code"))
            statements = ast.parse(
                "case_metadata[idx] = None\n"
                "if idx in retained_indices:\n"
                "    return all_results, {'by_case': case_metadata}\n"
                "continue"
            ).body
            statements[0].value = copy.deepcopy(metadata)
            return statements

    ContinueIgnored().visit(loop)
    if sorted(changes) != [-4, -3, -2]:
        raise ValueError("Expected exactly the three frozen failure exits")
    grade.body.insert(grade.body.index(loop), ast.parse("case_metadata = {}").body[0])
    final = grade.body[-1]
    if not isinstance(final, ast.Return) or not isinstance(final.value, ast.Tuple):
        raise ValueError("Official completion return changed")
    final.value.elts[1].keys.append(ast.Constant("by_case"))
    final.value.elts[1].values.append(ast.Name("case_metadata", ast.Load()))
    tree.body.append(ast.parse("retained_indices = " + repr(set(retained))).body[0])
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


@lru_cache(maxsize=1)
def host_code():
    check_source()
    tree = ast.parse(inspect.getsource(lc.evaluate_suite))
    changes = {"metadata": 0, "stop": 0}

    class ParseContinued(ast.NodeTransformer):
        def visit_For(self, node):
            if isinstance(node.target, ast.Tuple) and [getattr(n, "id", None) for n in node.target.elts] == ["index", "value"]:
                node = self.generic_visit(node)
                node.body.insert(0, ast.parse("case_metadata = metadata.get('by_case', {}).get(str(index), metadata)").body[0])
                return node
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "metadata" and node.func.attr == "get":
                node.func.value.id = "case_metadata"
                changes["metadata"] += 1
            return self.generic_visit(node)

        def visit_If(self, node):
            if (isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                    and node.test.func.id == "any" and "cases[:-1]" in ast.unparse(node.test)):
                node.test = ast.parse("any(i in retained_indices and row['status'] != 'PASS' for i, row in enumerate(cases[:-1]))", mode="eval").body
                changes["stop"] += 1
            return self.generic_visit(node)

    ParseContinued().visit(tree)
    if changes != {"metadata": 3, "stop": 1}:
        raise ValueError(f"Unexpected frozen parsing structure: {changes}")
    return compile(ast.fix_missing_locations(tree), "<frozen_lc_retained_parser>", "exec")


def evaluate_suite(sandbox, code, tests, limits, visibility, fn_name, retained_indices):
    retained = list(retained_indices)
    if retained != sorted(set(retained)) or any(type(i) is not int or not 0 <= i < len(tests) for i in retained):
        raise ValueError("Retained indices must be unique, ordered, and in range")
    if not retained:
        raise ValueError("No retained cases; caller must preserve UNKNOWN without executing")
    source = continued_source(retained)
    namespace = dict(lc.__dict__, retained_indices=set(retained), official_subset=lambda: source)
    exec(host_code(), namespace)
    result = namespace["evaluate_suite"](sandbox, code, tests, limits, visibility, fn_name)
    measured = sum(m["phase"] == "case" for m in result.get("method_measurements", []))
    if result["status"] != "UNKNOWN" and measured != result["evaluated"]:
        # Official run_test also returns [-4] for compilation/method failures.
        # Without an actual method call that value cannot identify case zero.
        result.update(status="UNKNOWN", all_passed=None, evaluated=0,
                      not_evaluated_after_failure=0, cases=[],
                      error="Official result cardinality differs from actual method calls")
    metadata = result.get("official_metadata", {}).get("by_case", {})
    if not result["cases"]:
        result["cases"] = [dict(test_index=i, status="UNKNOWN") for i in range(len(tests))]
    for case in result["cases"]:
        case["retained"] = case["test_index"] in retained
        case["official_metadata"] = metadata.get(str(case["test_index"]), {})
    states = [result["cases"][i]["status"] for i in retained]
    passed = False if any(s in FAILURES for s in states) else True if all(s == "PASS" for s in states) else None
    result.update(all_passed=passed, status="PASS" if passed is True else "FAIL" if passed is False else "UNKNOWN",
                  retained_indices=retained, retained_case_status_counts=dict(Counter(states)),
                  stop_on_failure="retained_only",
                  label_scope="Only retained cases; complete original order and same Solution instance",
                  private_sequence_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result
