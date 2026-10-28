"""基于已观测的草稿前缀回放v1策略，只使用公开信号。"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from exec_exit.replay.policies import NativeCost, PublicProbe, _finish, _validate, _valid


@dataclass(frozen=True)
class ProbeV1:
    public: PublicProbe
    abort_at_token: int | None
    ast_signature: str | None

    @property
    def effective_tokens(self):
        return self.public.generated_tokens if self.abort_at_token is None else self.abort_at_token

    @property
    def valid(self):
        return self.abort_at_token is None and _valid(self.public)


def select_candidates(probes: Sequence[ProbeV1], *, k=1, minimum=0):
    if k not in (1, 2, 4) or minimum not in (0, 2048):
        raise ValueError("Exploration grid is k=1/2/4 and m=0/2048")
    ordered = sorted(probes, key=lambda p: p.public.position)
    #先按间隔抽取，再过滤最小位置；交换顺序会改变被访问的草稿
    return [p for p in ordered[::k] if p.public.position >= minimum]


def _visit(probe, *, charged=None, public=False, private=False, decision):
    p = probe.public
    return dict(subject_id=p.subject_id, position=p.position,
        charged_probe_generated_tokens=probe.effective_tokens if charged is None else charged,
        charged_probe_control_tokens=p.control_tokens,
        full_record_generated_tokens=p.generated_tokens,
        abort_at_token=probe.abort_at_token, decision=decision,
        public_check_demand=public, private_check_demand=private)


def choose_v1(policy : str, native : NativeCost, probes : Sequence[ProbeV1], *, threshold=.95)->dict:
    """按公开信号选择草稿，不读取开发集或隐藏测试的正确性标签。"""
    names = {"N", "FIRST", "CONF-50", "CONV-TEXT", "EG", "EG-2"}
    fixed = policy.startswith("BF@")
    if policy not in names and not fixed:
        raise ValueError("Unknown deployable policy")
    if any(not isinstance(p, ProbeV1) for p in probes):
        raise TypeError("Ordinary rules only accept public ProbeV1 objects")
    _validate(native, [p.public for p in probes])
    if policy == "N":
        return _finish(policy, native, [], None)
    #1.固定预算策略只访问对应位置，之前的候选不产生探测成本
    if fixed:
        budget = int(policy.split("@", 1)[1])
        if budget not in (2048, 4096, 8192, 16384):
            raise ValueError("Unknown fixed budget")
        probes = [p for p in probes if p.public.position == budget]

    #2.逐个判断，连续通过或一致性策略需要记住上一份草稿
    visits = []
    previous_pub = False
    previous_ast = None
    for probe in probes:
        p = probe.public
        high = p.confidence_50 is not None and p.confidence_50 > threshold
        #中止和低置信度门限谁先到，就只支付到谁的位置
        charged = probe.effective_tokens
        if policy == "CONF-50" and not high:
            charged = min(50, charged)
        valid = probe.valid
        public_demand = policy in {"EG", "EG-2"} and valid
        pub_ok = valid and p.public_status == "PASS"
        text_ok = (valid and probe.ast_signature is not None
                   and probe.ast_signature == previous_ast)
        if not valid:
            accept = False
        elif fixed or policy == "FIRST":
            accept = True
        elif policy == "CONF-50":
            accept = high
        elif policy == "CONV-TEXT":
            accept = text_ok
        elif policy == "EG":
            accept = pub_ok
        else:
            accept = pub_ok and previous_pub

        if accept:
            reason = "ACCEPT"
        elif policy == "CONF-50" and not high and charged == 50:
            reason = "CONFIDENCE_REJECT_AT_50"
        elif probe.abort_at_token is not None:
            reason = "COMMENT_ABORT"
        elif not valid:
            reason = "INVALID_DELIVERY"
        else:
            reason = "ACCEPTANCE_CONDITION_FALSE"
        visits.append(_visit(probe, charged=charged, public=public_demand, decision=reason))
        if accept:
            return _finish(policy, native, visits, p)
        if policy == "FIRST" or fixed:
            break
        #连续通过不能跳过失败草稿；文本一致也不能跨过无效或缺少表示的草稿
        previous_pub = pub_ok
        previous_ast = probe.ast_signature if valid else None

    return _finish(policy, native, visits, None)


def choose_oracle_v1(native: NativeCost, probes: Sequence[ProbeV1], labels: Mapping[str, str]):
    _validate(native, [p.public for p in probes])
    visits = []
    for probe in probes:
        p = probe.public
        accept = probe.valid and p.public_status == "PASS" and labels.get(p.subject_id) == "PASS"
        visits.append(_visit(probe, public=probe.valid,
                             private=probe.valid and p.public_status == "PASS",
                             decision="ACCEPT" if accept else "NO_OBSERVED_COMPLETE_PASS"))
        if accept:
            return _finish("ORACLE", native, visits, p)
    return _finish("ORACLE", native, visits, None)


def free_perfect_bound(native: NativeCost, probes: Sequence[ProbeV1], labels: Mapping[str, str], *, include_delivery=True):
    """在已记录的可通过答案中取最低成本，仅供离线参照。

    没有可通过草稿时保留原生答案；原生答案通过且更便宜时也可选它。
    此处不计探测查询和检查成本，include_delivery=False还免去交付草稿成本，
    因而不能作为实际可部署策略的收益。
    """
    options = [(native.generated_tokens, native.subject_id)] if labels.get(native.subject_id) == "PASS" else []
    for probe in probes:
        p = probe.public
        if probe.valid and labels.get(p.subject_id) == "PASS":
            cost = p.position + (p.generated_tokens if include_delivery else 0)
            options.append((cost, p.subject_id))
    if options:
        cost, selected = min(options)
    else:
        cost, selected = native.generated_tokens, native.subject_id
    return dict(selected_subject_id=selected, total_generated_tokens=cost,
                outcome=labels.get(selected, "UNKNOWN"), bound_only=True,
                include_delivery_draft_cost=include_delivery,
                probe_query_and_check_cost=0)
