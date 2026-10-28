"""仅使用公开信息的异步接受协调器。

后端提供真实生成流、取消操作和仅执行公开测试的进程池检查器。
本模块不读取隐藏标签、未来轨迹或隐藏测试。
"""
from __future__ import annotations

import asyncio
import time

IMPLEMENTED = True
GPU_VALIDATED = False


class TokenAccount:
    def __init__(self):
        self.requests = {}

    def observe(self, request_id, ids):
        if not isinstance(ids, list) or any(type(t) is not int for t in ids):
            raise ValueError('Actual native token IDs required')
        row = self.requests.setdefault(request_id, dict(observed_tokens=0,
            observed_after_abort=0, abort_ns=None, unobserved_cancelled_tokens='UNKNOWN',
            gpu_kernel_seconds='UNKNOWN'))
        row['observed_tokens'] += len(ids)
        if row['abort_ns'] is not None:
            row['observed_after_abort'] += len(ids)

    def abort_requested(self, request_id):
        self.observe(request_id, [])
        if self.requests[request_id]['abort_ns'] is None:
            self.requests[request_id]['abort_ns'] = time.perf_counter_ns()

    def finished(self, request_id, native_finished):
        self.observe(request_id, [])
        row = self.requests[request_id]
        row['ended_ns'] = time.perf_counter_ns()
        if native_finished and row['abort_ns'] is None:
            row['unobserved_cancelled_tokens'] = 0


async def serve(main_stream, *, positions, draft, check_public,
                abort_main, abort_draft, journal, max_pending=8, on_delivery=None):
    """主生成持续运行，已知前缀按有限队列先进先出地探测。

    positions提供已可见的合法HYB-B位置，draft返回真实生成并提取的草稿，
    check_public通过CPU进程池只运行公开测试。原生答案与合格草稿竞争唯一交付，
    迟到的检查不能覆盖已交付答案；已开始的CPU工作必须等待结束并计费。
    取消回调必须排空模型流，AsyncLLM.abort返回不代表GPU已停止生成。
    """
    if max_pending < 1:
        raise ValueError('Positive finite queue required')
    delivered = asyncio.get_running_loop().create_future()
    pending = asyncio.Queue(maxsize=max_pending)
    ids, seen, visits, errors = [], set(), [], []
    draft_active = False
    started = time.perf_counter_ns()

    def deliver(kind, value):
        if delivered.done():
            return False
        now = time.perf_counter_ns()
        #回调可立即发起非阻塞发送，不必等待生成和检查排空；客户端接收时间另测
        if on_delivery is not None:
            on_delivery(kind, value, now)
        delivered.set_result(dict(kind=kind, value=value, delivered_ns=now))
        journal('delivery', kind=kind, visible_ns=now)
        return True

    async def produce():
        try:
            async for event in main_stream:
                delta = event['token_ids']
                if not isinstance(delta, list) or any(type(x) is not int for x in delta):
                    raise ValueError('Actual native delta IDs required')
                ids.extend(delta)
                journal('main_receive', tokens=len(delta), visible_ns=time.perf_counter_ns())
                if delivered.done():
                    continue
                if event.get('finished'):
                    deliver('N', event['result'])
                    if draft_active:
                        await abort_draft()
                    return
                for position in positions(ids, False):
                    if type(position) is not int or not 0 < position <= min(len(ids), 4096):
                        raise ValueError('Cannot probe future or post-4096 tokens')
                    if position in seen:
                        continue
                    if seen and position < max(seen):
                        raise ValueError('Cannot revise past opportunity order')
                    seen.add(position)
                    #队列满时显式报错，不能阻塞主生成或静默丢弃合法探测点
                    pending.put_nowait((position, tuple(ids[:position])))
            if not delivered.done():
                raise RuntimeError('Main ended without native completion or accepted draft')
        except BaseException as exc:
            errors.append(dict(stage='main', error=repr(exc)))
            if not delivered.done():
                delivered.set_exception(exc)

    async def consume():
        nonlocal draft_active
        try:
            while True:
                opportunity = await pending.get()
                if opportunity is None:
                    return
                position, prefix = opportunity
                row = dict(position=position, started_ns=time.perf_counter_ns())
                visits.append(row)
                if delivered.done():
                    row['status'] = 'NOT_STARTED_AFTER_DELIVERY'
                    continue
                draft_active = True
                output = await draft(position, prefix)
                draft_active = False
                row.update(draft=output, draft_complete_ns=time.perf_counter_ns())
                if delivered.done():
                    row['status'] = 'NATIVE_COMPLETED_BEFORE_CHECK'
                    continue
                if output.get('deliverable') is not True or output.get('comment_aborted'):
                    row['status'] = 'DRAFT_REJECTED'
                    continue
                row['public_started_ns'] = time.perf_counter_ns()
                check = await check_public(output)
                row.update(public=check, public_complete_ns=time.perf_counter_ns())
                if check.get('status') != 'PASS':
                    row['status'] = 'PUBLIC_REJECTED_OR_UNKNOWN'
                    continue
                if deliver('PROBE', output):
                    row['status'] = 'ACCEPTED'
                    row['abort_requested_ns'] = time.perf_counter_ns()
                    await abort_main()
                    row['abort_returned_ns'] = time.perf_counter_ns()
                else:
                    row['status'] = 'LATE_CHECK_AFTER_NATIVE_DELIVERY'
        except BaseException as exc:
            errors.append(dict(stage='probe_or_public', error=repr(exc)))
            if not delivered.done():
                delivered.set_exception(exc)
        finally:
            draft_active = False

    producer = asyncio.create_task(produce())
    consumer = asyncio.create_task(consume())
    failure = None
    try:
        result = await delivered
    except BaseException as exc:
        failure = exc
        result = None
        await abort_main()
        await abort_draft()
    finally:
        await producer
        #交付后不再生成排队中的草稿，但仍记录这些机会并等待已开始的工作结束
        while not pending.empty():
            item = pending.get_nowait()
            if item is not None:
                visits.append(dict(position=item[0], status='NOT_STARTED_AFTER_DELIVERY'))
        pending.put_nowait(None)
        await consumer
    if failure is not None:
        raise failure
    return dict(result, started_ns=started, drained_ns=time.perf_counter_ns(),
                main_observed_tokens=len(ids), visits=visits, errors=errors,
                gpu_kernel_seconds='UNKNOWN_NOT_ATTRIBUTED_BY_REQUEST')
