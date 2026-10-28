"""CONF-50 streaming adapter: frozen rank-one score and v1 comment rejection.

No operating point is selected here. Every returned ID is retained; same-message
and unexposed cancellation tails are not falsely charged as exactly 50 tokens.
"""
import asyncio,time
from exec_exit.gen.engine_preflight import validate_prompt_token_ids,serialize_logprobs
from exec_exit.gen.probes import rank_one_values,native_metrics
from exec_exit.replay.early_abort import StreamAbortDetector
from exec_exit.replay.replay import confidence_50

async def generate_conf50_probe(engine, tokenizer, writer, spec, parameter_factory, observe=lambda *args: None,
                         *, codec, special_ids=()):
    """Detect using actual token bytes; retain/charge the entire received chunk."""
    prompt = spec["prompt_token_ids"]
    validate_prompt_token_ids(prompt)
    request_id, started = spec["probe_id"], time.perf_counter_ns()
    call = dict(request_id=request_id, prompt_token_ids=prompt, prompt_tokens=len(prompt),
        sampling_parameters=spec["sampling_parameters"], seed=spec["sampling_parameters"]["seed"],
        started_ns=started, token_ids=[], raw_text="", logprobs=[], rank_one_logprobs=[],
        returned_rank_one_entry_counts=[], token_observation_ns=[], native_finished=False,
        finish_reason=None, stop_reason=None, num_cached_tokens=None, first_output_ns=None,
        last_output_ns=None, error=None, abort_requested=False, protocol_rejected=False, confidence_rejected=False, confidence_decision=None,
        preemptions="RUN_TOTAL_NATIVE_COUNTER; PER_REQUEST_UNKNOWN")
    writer.add("probe_start", probe_id=request_id, case_id=spec["case_id"],
        candidate_index=spec["candidate_index"], position_generated_tokens=spec["position_generated_tokens"],
        prompt_token_ids=prompt, control_token_ids=spec["control_token_ids"],
        sampling_parameters=spec["sampling_parameters"])
    texts, detector, specials = [], StreamAbortDetector(), set(special_ids)
    stream = None
    try:
        parameters = parameter_factory(spec["sampling_parameters"])
        call["effective_sampling_parameters"] = {
            name: parameters.get(name) if isinstance(parameters, dict) else getattr(parameters, name)
            for name in spec["sampling_parameters"]}
        stream = engine.generate({"prompt_token_ids": prompt}, parameters, request_id)
        async for output in stream:
            received = time.perf_counter_ns()
            if len(output.outputs) != 1:
                raise RuntimeError("Expected exactly one completion per probe")
            completion = output.outputs[0]
            ids, offset = list(completion.token_ids), len(call["token_ids"])
            call["token_ids"].extend(ids)
            call["token_observation_ns"].extend([received] * len(ids))
            texts.append(completion.text)
            call.update(native_finished=bool(output.finished), finish_reason=completion.finish_reason,
                stop_reason=completion.stop_reason, last_output_ns=received)
            cached = getattr(output, "num_cached_tokens", None)
            if cached is not None:
                call["num_cached_tokens"] = int(cached)
            if ids and call["first_output_ns"] is None:
                call["first_output_ns"] = received
            probabilities = serialize_logprobs(completion.logprobs)
            ranks = rank_one_values(ids, probabilities)
            counts = [sum(item["rank"] == 1 for item in row.values()) for row in (probabilities or [])]
            call["logprobs"].extend(probabilities or [])
            call["rank_one_logprobs"].extend(ranks)
            call["returned_rank_one_entry_counts"].extend(counts)
            observe(request_id, len(call["token_ids"]))
            writer.add("probe_output", probe_id=request_id, token_offset=offset, token_ids=ids,
                text=completion.text, logprobs=probabilities, rank_one_logprobs=ranks,
                returned_rank_one_entry_counts=counts, native_frontend_receive_ns=received,
                num_cached_tokens=cached, finished=output.finished,
                finish_reason=completion.finish_reason, stop_reason=completion.stop_reason)
            match = None
            for token_id in ids:
                match = detector.feed(b"" if token_id in specials else codec.vocab[token_id])
                if match is not None:
                    break
            if match is None and output.finished:
                match = detector.finish()
            if output.finished:
                call["native_metrics"] = native_metrics(output)
            if call['confidence_decision'] is None and (len(call['token_ids'])>=50 or output.finished):
                score=confidence_50(call)
                call['confidence_decision']=dict(score=score,threshold_strictly_greater_than=.95,
                    high=score is not None and score>.95,visible_ns=received,prefix_tokens=min(50,len(call['token_ids'])))
                call['confidence_rejected']=not call['confidence_decision']['high']
            if match is not None or call['confidence_rejected']:
                call['protocol_rejected']=match is not None
                gate_cutoff=min(50,len(call['token_ids'])) if call['confidence_rejected'] else None
                comment_cutoff=match.token_count if match is not None else None
                cutoff=min(x for x in (gate_cutoff,comment_cutoff) if x is not None)
                reason='CONFIDENCE_REJECT_AT_50_OR_FINAL' if gate_cutoff is not None and (comment_cutoff is None or gate_cutoff<=comment_cutoff) else 'COMMENT_REFLECTION_V1'
                call.update(reject_reason=reason,abort_match=match.as_dict() if match is not None else None,
                    detection_visible_ns=received,detection_token_count=cutoff,
                    received_tokens_when_detected=len(call['token_ids']),extra_received_tokens_same_message=len(call['token_ids'])-cutoff)
                if not output.finished:
                    call.update(abort_requested=True,abort_requested_ns=time.perf_counter_ns())
                    await asyncio.wait_for(engine.abort(request_id),15);call['abort_returned_ns']=time.perf_counter_ns()
                    call['abort_return_semantics']='Frontend cancellation submitted; unexposed GPU work UNKNOWN'
                    call['unobserved_inflight_tokens']='UNKNOWN';await stream.aclose()
                else:call['unobserved_inflight_tokens']=0
                writer.add('probe_policy_reject',probe_id=request_id,reason=reason,first_visible_cutoff=cutoff,
                    received_tokens=len(call['token_ids']),confidence_decision=call['confidence_decision'],
                    abort_requested=call['abort_requested'],unobserved_inflight_tokens=call['unobserved_inflight_tokens'])
                break
        if not call["native_finished"] and not call["protocol_rejected"] and not call["confidence_rejected"]:
            raise RuntimeError("Probe ended without a native finished event or protocol rejection")
        count = len(call["token_ids"])
        if count == 0 or count > 2048 or len(call["rank_one_logprobs"]) != count:
            raise RuntimeError("Probe native token/logprob alignment failed")
    except BaseException as error:
        call["error"] = repr(error)
        try:
            await asyncio.wait_for(engine.abort(request_id), timeout=15)
        except BaseException as abort_error:
            call["abort_error"] = repr(abort_error)
        raise
    finally:
        call.update(ended_ns=time.perf_counter_ns(), raw_text="".join(texts),
            token_count=len(call["token_ids"]),
            canonical_text=tokenizer.decode(call["token_ids"], skip_special_tokens=True,
                clean_up_tokenization_spaces=False),
            canonical_text_with_special_tokens=tokenizer.decode(call["token_ids"], skip_special_tokens=False,
                clean_up_tokenization_spaces=False))
        call["first_50_visible_ns"] = call["token_observation_ns"][49] if len(call["token_ids"]) >= 50 else "NOT_REACHED"
        call["returned_rank_one_tie_positions_zero_based"] = [
            i for i, count in enumerate(call["returned_rank_one_entry_counts"]) if count > 1]
        call["received_generated_token_cost"] = call["token_count"]
        call["physical_generated_token_cost"] = "UNKNOWN" if call["abort_requested"] else call["token_count"]
        writer.add("probe_end", probe_id=request_id, case_id=spec["case_id"], call=call)
    return {key: value for key, value in spec.items() if key not in ("prompt_token_ids", "sampling_parameters")} | dict(
        schema="exec_exit.probe.v2_online_abort", status="COMPLETE", call=call,
        protocol_rejected=call["protocol_rejected"], confidence_rejected=call["confidence_rejected"], confidence_decision=call["confidence_decision"], correctness="NOT_EVALUATED")

