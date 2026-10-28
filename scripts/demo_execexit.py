"""用两条合成轨迹演示现有HYB-B策略。

在仓库根目录运行python -m scripts.demo_execexit，仅依赖标准库。
公开测试结果是人为设定的，只用于说明决策和token计费，不是实验证据。
"""

from exec_exit.replay.hybrid_v1 import choose_hybrid
from exec_exit.replay.policies import NativeCost, PublicProbe
from exec_exit.replay.policies_v1 import ProbeV1


def candidates(boundary_status: str) -> list[ProbeV1]:
    return [
        ProbeV1(
            public=PublicProbe(
                subject_id=f"draft@{position}",
                position=position,
                generated_tokens=tokens,
                control_tokens=6,
                generation_valid=True,
                parse_ok=True,
                deliverable=True,
                public_status=boundary_status if position == 4096 else "FAIL",
                confidence_50=None,
            ),
            abort_at_token=None,
            ast_signature=None,
        )
        for position, tokens in [(512, 80), (1024, 90), (2048, 100), (3072, 110), (4096, 120)]
    ]


def main() -> None:
    print("ExecExit: synthetic HYB-B replay (no model or candidate execution)")
    print("HYB-B visits candidates 1, 5, ... before token 4096, plus the 4096 boundary.")
    print("PUB outcomes below are invented inputs; passing samples does not prove correctness.")
    #1.对照边界通过与全部被拒两种情况，观察交付答案和费用的变化
    native = NativeCost(subject_id="native", generated_tokens=6000, control_tokens=4)
    for title, status in [("Reject, then accept", "PASS"), ("All rejected: native fallback", "FAIL")]:
        probes = candidates(status)
        result = choose_hybrid("HYB-B", native, probes)
        #2.只打印策略实际访问过的草稿，不能把未访问的候选算进费用
        public_status = {probe.public.subject_id: probe.public.public_status for probe in probes}
        print(f"\n{title}")
        for visit in result["visited"]:
            subject = visit["subject_id"]
            decision = "accept" if visit["decision"] == "ACCEPT" else "reject"
            print(
                f"  {subject}: PUB={public_status[subject]} -> {decision}; "
                f"draft cost={visit['charged_probe_generated_tokens']} generated tokens"
            )
        print(f"  Delivered: {result['selected_subject_id']}")
        print(
            f"  Generated tokens: {result['main_generated_tokens']} main + "
            f"{result['probe_generated_tokens']} visited drafts = {result['total_generated_tokens']} "
            f"(native alone: {native.generated_tokens})"
        )
        print(f"  Control tokens, separately: {result['total_control_tokens']}")
    print("\nRejected drafts remain chargeable; an accepted draft is charged once.")
    print("These are replay token costs. GPU work and service latency are not measured.")


if __name__ == "__main__":
    main()
