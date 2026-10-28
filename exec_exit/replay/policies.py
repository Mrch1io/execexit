"""按冻结的v0规则回放停止策略，输入仅包含公开信息。

生成token成本按已记录的访问路径核算，不代表在线实测时延。
隐藏测试标签仅供独立的oracle入口使用。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

IMPLEMENTED = True


@dataclass(frozen=True)
class PublicProbe:
    subject_id: str
    position: int
    generated_tokens: int
    control_tokens: int
    generation_valid: bool | None
    parse_ok: bool | None
    deliverable: bool | None
    public_status: str
    confidence_50: float | None


@dataclass(frozen=True)
class NativeCost:
    subject_id: str
    generated_tokens: int
    control_tokens: int = 0


def _validate(native: NativeCost, probes: Sequence[PublicProbe]):
    if not isinstance(native, NativeCost):
        raise TypeError("native must contain cost-only NativeCost fields")
    if native.generated_tokens < 0 or native.control_tokens < 0:
        raise ValueError("Native costs cannot be negative")
    previous, ids = 0, set()
    for probe in probes:
        if not isinstance(probe, PublicProbe):
            raise TypeError("Ordinary selection accepts PublicProbe, not evaluation dictionaries")
        if (probe.position <= previous or probe.position > native.generated_tokens
                or probe.subject_id in ids):
            raise ValueError("Candidates must have unique ascending real positions and identities")
        if probe.generated_tokens < 0 or probe.control_tokens < 0:
            raise ValueError("Probe costs cannot be negative")
        if probe.confidence_50 is not None and (
                not math.isfinite(probe.confidence_50) or not 0 <= probe.confidence_50 <= 1):
            raise ValueError("Known confidence must be finite and within [0,1]")
        previous = probe.position
        ids.add(probe.subject_id)


def _valid(probe: PublicProbe):
    return (probe.generation_valid is True and probe.parse_ok is True
            and probe.deliverable is True)


def _finish(policy, native, visits, selected):
    #被拒草稿也是停止策略产生的成本；若只计算最终交付答案，就会高估节省。
    probe_tokens = sum(row["charged_probe_generated_tokens"] for row in visits)
    control_tokens = sum(row["charged_probe_control_tokens"] for row in visits)

    #接受草稿只算主轨迹前缀；回退则算完整原生生成
    if selected is None:
        selected_id = native.subject_id
        position = None
        main_tokens = native.generated_tokens
        control_tokens += native.control_tokens
    else:
        selected_id = selected.subject_id
        position = selected.position
        main_tokens = position

    return dict(
        policy=policy,
        selected_subject_id=selected_id,
        accepted_probe=selected is not None,
        accepted_position=position,
        visited=visits,
        main_generated_tokens=main_tokens,
        probe_generated_tokens=probe_tokens,
        total_generated_tokens=main_tokens + probe_tokens,
        total_control_tokens=control_tokens,
        public_check_demands=sum(row["public_check_demand"] for row in visits),
        private_check_demands=sum(row["private_check_demand"] for row in visits),
        accepted_draft_charged_once=True, online_latency_seconds="UNKNOWN_NOT_MEASURED",
        cost_scope="Fixed-record policy path; input/control and physical corpus generation reported separately",
    )


def choose_stop(policy : str, native : NativeCost, probes : Sequence[PublicProbe], *, threshold=.95)->dict:
    """只用公开信号决定是否停止，隐藏测试标签留给离线分析。"""
    if policy not in {"N", "FIRST", "CONF-50", "EG"}:
        raise ValueError("Oracle requires choose_oracle; unsupported policy: " + policy)
    _validate(native, probes)
    if policy == "N":
        return _finish(policy, native, [], None)

    #只有CONF-50在置信度不足时提前停稿，最多支付前50个token
    visits = []
    for probe in probes:
        valid = _valid(probe)
        high = probe.confidence_50 is not None and probe.confidence_50 > threshold
        partial = policy == "CONF-50" and not high
        charged = min(50, probe.generated_tokens) if partial else probe.generated_tokens
        public_demand = policy == "EG" and valid
        if policy == "FIRST":
            accept = valid
        elif policy == "CONF-50":
            accept = valid and high
        else:
            accept = valid and probe.public_status == "PASS"

        if accept:
            reason = "ACCEPT"
        elif partial:
            reason = "CONFIDENCE_NOT_STRICTLY_ABOVE_THRESHOLD"
        elif not valid:
            reason = "INVALID_DELIVERY"
        else:
            reason = "PUBLIC_NOT_PASS"
        visits.append(dict(subject_id=probe.subject_id, position=probe.position,
            charged_probe_generated_tokens=charged, charged_probe_control_tokens=probe.control_tokens,
            full_record_generated_tokens=probe.generated_tokens, decision=reason,
            public_check_demand=public_demand, private_check_demand=False))
        if accept:
            return _finish(policy, native, visits, probe)
        if policy == "FIRST":
            break

    return _finish(policy, native, visits, None)


def choose_oracle(native: NativeCost, probes: Sequence[PublicProbe],
                  dev_labels: Mapping[str, str]):
    """离线机会参照：选中草稿之前访问过的草稿也全部计费。"""
    _validate(native, probes)
    visits = []
    for probe in probes:
        valid = _valid(probe)
        label = dev_labels.get(probe.subject_id, "UNKNOWN")
        accept = valid and probe.public_status == "PASS" and label == "PASS"
        visits.append(dict(subject_id=probe.subject_id, position=probe.position,
            charged_probe_generated_tokens=probe.generated_tokens,
            charged_probe_control_tokens=probe.control_tokens,
            full_record_generated_tokens=probe.generated_tokens,
            decision="ACCEPT" if accept else "NO_OBSERVED_COMPLETE_PASS",
            public_check_demand=valid, private_check_demand=valid and probe.public_status == "PASS",
            offline_development_label=label))
        if accept:
            return _finish("ORACLE", native, visits, probe)
    return _finish("ORACLE", native, visits, None)
