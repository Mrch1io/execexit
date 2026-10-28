"""在同一份公开探测记录上回放HYB-A和HYB-B。"""
from __future__ import annotations

from exec_exit.replay.policies import NativeCost, _finish, _validate
from exec_exit.replay.policies_v1 import ProbeV1, _visit, select_candidates


def choose_hybrid(policy : str, native : NativeCost, probes : list[ProbeV1])->dict:
    """每隔4份访问一次，4096位置单独检查；决策只使用公开信号。"""
    if policy not in {"HYB-A", "HYB-B"}:
        raise ValueError("Only the two prespecified hybrids are supported")
    if any(not isinstance(p, ProbeV1) for p in probes):
        raise TypeError("Only public ProbeV1 inputs are allowed")
    _validate(native, [p.public for p in probes])

    #1.4096边界不参与抽稀，也不能因为抽稀选中过就检查两次
    before = [p for p in select_candidates(probes, k=4, minimum=0)
              if p.public.position < 4096]
    boundary = [p for p in probes if p.public.position == 4096]

    #2.边界处A接受有效草稿，B还要求公开测试通过
    visits = []
    for probe in before + boundary:
        at_boundary = probe.public.position == 4096
        requires_pub = not at_boundary or policy == "HYB-B"
        valid = probe.valid
        #AUROC提高，说明整体上更能分清对错，但通过公开样例的草稿仍可能是错的。
        #HYB-B拿到第一份通过的草稿就停，不会等后面的答案。
        #如果继续生成本来能答对，这次提前停止就把一道能做对的题做错了。
        #所以还得比较两种做法最终交付的正确率。
        accept = valid and (not requires_pub or probe.public.public_status == "PASS")
        if accept:
            reason = "ACCEPT"
        elif probe.abort_at_token is not None:
            reason = "COMMENT_ABORT"
        elif not valid:
            reason = "INVALID_DELIVERY"
        else:
            reason = "ACCEPTANCE_CONDITION_FALSE"
        visits.append(_visit(probe, public=requires_pub and valid, decision=reason))
        if accept:
            return _finish(policy, native, visits, probe.public)

    return _finish(policy, native, visits, None)
