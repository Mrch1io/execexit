"""复用现有chroot/uid/seccomp执行器和固定版本的LCB比较规则。

宿主机只编译可信评分器的AST。候选源码原样传入受限子进程，
不得在宿主机中对其调用exec/eval。
"""
from __future__ import annotations

import ast
import copy
from decimal import Decimal
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

from exec_exit.data.load_lcb import official_evaluator_path
from exec_exit.execute.machine_identity import current_machine_identity

ROOT = Path(__file__).resolve().parents[2]
LEGACY_EXECUTOR = ROOT / "research/early_exit_component/run/scripts/code_executor.py"
LIMITS_PATH = ROOT / "research/early_exit_component/run/configs/execution_limits.json"
_legacy, _compare = None, None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_compare(actual: str, expected: str, mode="lcb_stdio_exact_decimal") -> bool:
    """仅执行固定版本grade_stdio中的输出比较逻辑，不运行其执行器。"""
    global _compare
    if mode != "lcb_stdio_exact_decimal":
        raise ValueError("Only the frozen official stdio comparison is supported")
    if _compare is None:
        tree = ast.parse(official_evaluator_path().read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        loop = next(node for node in functions["grade_stdio"].body if isinstance(node, ast.For))
        index = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "prediction" for target in node.targets))
        wrapper = ast.parse("def compare(actual, expected):\n prediction=actual\n gt_out=expected\n gt_inp=''\n all_results=[]\n").body[0]
        wrapper.body += copy.deepcopy(loop.body[index+1:])
        wrapper.body += ast.parse("return all_results[-1] is True").body
        module = ast.Module(body=[copy.deepcopy(functions[name]) for name in
                                ("truncatefn", "convert_line_to_decimals", "get_stripped_lines")]+[wrapper], type_ignores=[])
        namespace = {"Decimal": Decimal}
        exec(compile(ast.fix_missing_locations(module), str(official_evaluator_path()), "exec"), namespace)
        _compare = namespace["compare"]
    return _compare(actual, expected) is True


def legacy_module():
    global _legacy
    if _legacy is None:
        spec = importlib.util.spec_from_file_location("exec_exit_pinned_legacy_executor", LEGACY_EXECUTOR)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.compare_output = official_compare
        _legacy = module
    return _legacy


def frozen_limits():
    values = json.loads(LIMITS_PATH.read_text())
    expected = dict(wall_seconds=72, cpu_seconds=60, memory_bytes=1073741824,
                    output_bytes=8388608, code_bytes=131072, input_bytes=8388608)
    if values != expected:
        raise ValueError("Existing frozen W0 limits changed")
    return values


def clone_template(template_dir, runtime_dir):
    """仅硬链接新运行的只读模板，各工作进程独立持有program.py。"""
    template, destination = Path(template_dir), Path(runtime_dir)
    if destination.exists():
        raise FileExistsError("Worker runtime already exists; no implicit reuse")
    def ignore(directory, names):
        return [name for name in names if name in {"program.py", "execution.lock", "runtime_manifest.json"}]
    shutil.copytree(template, destination, copy_function=os.link, ignore=ignore)
    manifest = json.loads((template/"runtime_manifest.json").read_text())
    manifest["rootfs"] = str((destination/"rootfs").resolve())
    (destination/"runtime_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def make_sandbox(runtime_dir, *, template_dir=None, python_executable=None):
    #隔离执行器必须显式选择解释器，不能自动使用宿主环境
    if python_executable is None:
        python_executable = os.environ["EXECEXIT_SANDBOX_PYTHON"]
    if template_dir is not None:
        clone_template(template_dir, runtime_dir)
    return legacy_module().PublicCodeExecutor(runtime_dir, python_executable=python_executable)


def execution_identity(sandbox):
    manifest = sandbox.runtime_manifest
    return dict(machine=current_machine_identity(), interpreter={key: manifest[key] for key in
                            ("python", "python_version", "python_sha256", "libpython_sha256")},
                comparator=dict(name="official_LCB_grade_stdio_comparison_block", source_sha256=sha(official_evaluator_path()),
                                adapter_sha256=sha(__file__)),
                executor=dict(legacy_source_sha256=sha(LEGACY_EXECUTOR), adapter_sha256=sha(__file__),
                              runtime_fingerprint=manifest["fingerprint"]))


def evaluate_suite(sandbox, code, tests, limits, visibility):
    """公开测试和开发集隐藏测试分别在首次失败时停止。

    旧evaluate()不允许公开测试提前停止，因此直接复用底层隔离执行接口，
    保持公开输入的身份，不将其伪装成隐藏测试。
    """
    if not isinstance(code, bytes) or len(code) > limits["code_bytes"] or not tests:
        raise ValueError("A bounded original program and nonempty test suite are required")
    for row in tests:
        if set(row) != {"input", "output"} or not all(isinstance(value, str) for value in row.values()):
            raise ValueError("Tests must contain exactly input/output strings")
        if len(row["input"].encode()) > limits["input_bytes"]:
            raise ValueError("Input exceeds the frozen limit")
    started, host_cpu = time.perf_counter(), time.thread_time()
    cases = []
    #1.同时持有执行器锁和文件锁，串行访问同一运行目录
    with sandbox.lock, (sandbox.runtime_dir/"execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, test in enumerate(tests):
            result = sandbox._run_case(code.decode("utf-8"), test["input"], legacy_module().ExecutionLimits(**limits))
            if result["status"] == "SANDBOX_SETUP_ERROR":
                raise RuntimeError("Isolation setup failed; no unsafe retry: "+result["stderr"])
            if result["status"] == "EXECUTED":
                result["status"] = "PASS" if official_compare(result["stdout"], test["output"]) else "WRONG_ANSWER"
            cases.append(result | dict(test_index=index))
            if result["status"] != "PASS":
                break
    #2.失败后未执行的用例单独记录，不能算作已观察到的失败
    observed = len(cases)
    all_passed = observed == len(tests) and all(row["status"] == "PASS" for row in cases)
    child_cpu = sum(row["cpu_seconds"] for row in cases)
    host_seconds = time.thread_time()-host_cpu
    return dict(status="PASS" if all_passed else "FAIL", all_passed=all_passed,
                total=len(tests), evaluated=observed, not_evaluated_after_failure=len(tests)-observed,
                cases=cases + [dict(test_index=i, status="NOT_EVALUATED_AFTER_FAILURE") for i in range(observed, len(tests))],
                wall_seconds=time.perf_counter()-started, child_wall_seconds=sum(row["wall_seconds"] for row in cases),
                cpu_seconds=child_cpu+host_seconds, child_cpu_seconds=child_cpu, host_thread_cpu_seconds=host_seconds,
                visibility=visibility, stop_on_failure=True, limits=limits,
                isolation=sandbox.runtime_manifest, code_sha256=hashlib.sha256(code).hexdigest())
