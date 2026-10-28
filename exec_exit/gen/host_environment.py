"""Read-only machine metadata without importing torch or creating a GPU context."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys


def collect_host_environment():
    errors = {}

    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError as error:
            errors[path] = repr(error)
            return "UNKNOWN"

    cpuinfo = read("/proc/cpuinfo")
    models = sorted({line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                     if line.startswith("model name") and ":" in line})
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        errors["sched_getaffinity"] = repr(error)
        affinity = None
    quota = read("/sys/fs/cgroup/cpu.max")
    quota_cores = "UNKNOWN"
    if quota != "UNKNOWN":
        fields = quota.split()
        if len(fields) == 2:
            quota_cores = "UNLIMITED" if fields[0] == "max" else int(fields[0]) / int(fields[1])
    versions = {}
    for name in ("vllm", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED_IN_PREPARE_INTERPRETER"
    try:
        gpu_inventory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        errors["nvidia-smi"] = repr(error)
        gpu_inventory = "UNKNOWN"
    return dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
        observation_scope="Prepare-time read-only host metadata; no GPU context or model initialized",
        cpu_model=models[0] if len(models) == 1 else models or "UNKNOWN",
        affinity_logical_cpus=len(affinity) if affinity is not None else "UNKNOWN",
        affinity_cpu_ids=affinity if affinity is not None else "UNKNOWN",
        cpu_quota=quota, cpu_quota_cores=quota_cores,
        hostname=platform.node(), boot_id=read("/proc/sys/kernel/random/boot_id"),
        gpu_inventory=gpu_inventory, software_versions=versions,
        python_executable=sys.executable, python_version=platform.python_version(),
        collection_errors=errors)
