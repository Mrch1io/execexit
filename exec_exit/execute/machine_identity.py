"""Explicit local execution host identity; no cross-host cache reuse."""
from __future__ import annotations
import hashlib,json,os,platform
from pathlib import Path

def current_machine_identity():
    def read(path):
        p=Path(path)
        return p.read_text().strip() if p.exists() else "UNKNOWN"
    info=read("/proc/cpuinfo")
    models=sorted({line.split(":",1)[1].strip() for line in info.splitlines() if line.startswith("model name")})
    boot=read("/proc/sys/kernel/random/boot_id")
    if boot=="UNKNOWN" or not models:
        raise RuntimeError("Host CPU/boot identity unavailable; refuse reusable execution cache")
    identity=dict(schema="exec_exit.host.v1",hostname=platform.node(),boot_id=boot,cpu_models=models,
       cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_max=read("/sys/fs/cgroup/cpu.max"),
       cpuset_effective=read("/sys/fs/cgroup/cpuset.cpus.effective"),memory_max=read("/sys/fs/cgroup/memory.max"),
       kernel=platform.release(),machine_architecture=platform.machine())
    # Boot and container host together deliberately over-isolate cache reuse after migration/reboot.
    identity["fingerprint"]=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return identity
