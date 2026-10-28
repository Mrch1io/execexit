"""在v1之外提供STRESS/AGREE策略，只接收公开信息。

停止决策仅使用首次公开测试和生成输入的执行观测，不读取评分标签。
UNKNOWN不能作为接受条件。
"""
from dataclasses import dataclass
from collections.abc import Mapping

from exec_exit.replay.policies import _finish, _validate
from exec_exit.replay.policies_v1 import _visit


@dataclass(frozen=True)
class ExecutionSignals:
    stress: Mapping[str, str]
    agreement: Mapping[tuple[str, str], str]


def choose_execution(policy, native, probes, signals):
    if policy not in {"EG-S", "EG-A", "EG-SA"}:
        raise ValueError("Only the three execution variants")
    if not isinstance(signals, ExecutionSignals):
        raise TypeError("Explicit public-only execution observations required")
    _validate(native, [p.public for p in probes])
    use_stress = policy in {"EG-S", "EG-SA"}
    use_agree = policy in {"EG-A", "EG-SA"}
    visits = []
    previous_pub = None

    #检查顺序固定为公开测试、压力测试、行为一致性
    for probe in probes:
        p = probe.public
        valid = probe.valid
        public_ok = valid and p.public_status == "PASS"
        stress_demand = public_ok and use_stress
        stress = signals.stress.get(p.subject_id, "UNKNOWN") if stress_demand else "NOT_REQUESTED"
        stress_ok = not use_stress or stress == "PASS"
        agree_demand = public_ok and use_agree and stress_ok
        agree = "NOT_REQUESTED"
        if agree_demand:
            if previous_pub is None:
                agree = "NO_PREVIOUS_PUB"
            else:
                agree = signals.agreement.get((previous_pub, p.subject_id), "UNKNOWN")

        accept = public_ok and stress_ok and (not use_agree or agree == "PASS")
        visit = _visit(probe, public=valid, decision="ACCEPT" if accept else "EXECUTION_CONDITION_NOT_PASS")
        visit.update(stress_status=stress, agreement_status=agree,
                     stress_check_demand=stress_demand,
                     agreement_check_demand=agree_demand and previous_pub is not None,
                     previous_pub_subject_id=previous_pub if agree_demand else None)
        visits.append(visit)
        if accept:
            return _finish(policy, native, visits, p)
        #记录上一份公开测试通过的草稿；压力测试或一致性失败不会清除这条观测
        if public_ok:
            previous_pub = p.subject_id

    return _finish(policy, native, visits, None)


def generator_failed(audit, rule):
    """按指定规则判断生成器失败，不使用候选程序的评分结果。"""
    if rule not in {"execution_only", "include_validator_rejection"}:
        raise ValueError("Explicit generator-failure rule required")
    failed = not audit["generator_all_planned_calls_returned"]
    if rule == "include_validator_rejection":
        failed |= audit["input_validation_counts"].get("MODEL_VALIDATOR_REJECTED", 0) > 0
    return failed


def tool_token_charge(total_generated_tokens, sample_count, mode):
    if mode == "per_sample_full":
        return total_generated_tokens
    if mode == "amortized_across_samples":
        if sample_count < 1:
            raise ValueError("Positive sample count required")
        return total_generated_tokens / sample_count
    raise ValueError("Primary tool-cost policy must be explicitly chosen")
