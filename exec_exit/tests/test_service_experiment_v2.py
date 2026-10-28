import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from exec_exit.serve import service_experiment_v2 as service


class ServiceDispatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.task = {
            "case_id": "synthetic:a",
            "task_id": "synthetic:a",
            "question_content": "Print OK.",
            "public_test_cases": [{"input": "", "output": "OK\n"}],
        }
        self.tasks = {self.task["case_id"]: self.task}
        self.spec = {"sample_index": 0, "serial": {"H": {"schedule": "MATCHED_SYNC"}}}
        self.entry = {"kind": "serial", "case_id": self.task["case_id"], "mode": "H"}

    async def test_matched_control_waits_for_complete_sync(self):
        backend = AsyncMock()
        for prior in (None, {"status": "FAILED_SERVICE"}):
            with self.subTest(prior=prior):
                previous = {} if prior is None else {(self.task["case_id"], "HYB-B-SYNC"): prior}
                result = await service.dispatch(
                    self.entry, self.spec, self.tasks, None, None, {}, previous, backend=backend
                )
                self.assertEqual(result["status"], "NOT_RUN_DEPENDENCY_FAILED")
                self.assertNotIn((self.task["case_id"], "H"), previous)
        backend.assert_not_awaited()

    async def test_matched_control_uses_only_executed_positions(self):
        previous = {
            (self.task["case_id"], "HYB-B-SYNC"): {
                "status": "COMPLETE",
                "pauses": [
                    {"nominal_positions": [100, 2048], "executed_nominal_positions": [100]},
                    {"nominal_positions": [4096], "executed_nominal_positions": [4096]},
                ],
            }
        }
        record = {"status": "COMPLETE", "client": {"client_e2e_seconds": 2.0}}
        backend = AsyncMock(return_value=record)
        result = await service.dispatch(self.entry, self.spec, self.tasks, None, None, {}, previous, backend=backend)
        self.assertIs(result, record)
        self.assertIs(previous[self.task["case_id"], "H"], record)
        arguments = backend.await_args.kwargs
        self.assertEqual(arguments["matched_positions"], [100, 4096])
        self.assertEqual(arguments["h_schedule"], "MATCHED_SYNC")
        self.assertEqual(set(arguments["case"]), {"case_id", "task_id", "question_content"})
        self.assertIs(arguments["task"], self.task)

    async def test_backend_failure_does_not_record_success(self):
        previous = {}
        backend = AsyncMock(side_effect=RuntimeError("native request failed"))
        with self.assertRaisesRegex(RuntimeError, "native request failed"):
            await service.dispatch(
                dict(self.entry, mode="N"), self.spec, self.tasks, None, None, {}, previous, backend=backend
            )
        self.assertEqual(previous, {})

    async def test_poisson_dispatch_preserves_scheduled_requests(self):
        arrivals = [
            {"request_index": 0, "case_id": self.task["case_id"], "intended_arrival_seconds": 0.1},
            {"request_index": 1, "case_id": self.task["case_id"], "intended_arrival_seconds": 0.2},
        ]
        engine = object()
        record = {"status": "COMPLETE"}
        backend = AsyncMock(return_value=record)

        async def poisson(actual_engine, **kwargs):
            self.assertIs(actual_engine, engine)
            self.assertIs(kwargs["arrivals"], arrivals)
            self.assertEqual(kwargs["run_identity"], "c4-poisson-7")
            return await asyncio.gather(
                *(kwargs["backend"](actual_engine, request_index=row["request_index"]) for row in arrivals)
            )

        result = await service.dispatch(
            {"kind": "poisson", "mode": "N", "arrivals": arrivals, "experiment_index": 7},
            self.spec,
            self.tasks,
            engine,
            None,
            {},
            {},
            backend=backend,
            poisson=poisson,
        )
        self.assertEqual(result, [record, record])
        self.assertEqual([call.kwargs["request_index"] for call in backend.await_args_list], [0, 1])
        self.assertTrue(all(call.kwargs["engine"] is engine for call in backend.await_args_list))


class ServiceFreezeTests(unittest.TestCase):
    def test_changed_source_is_rejected_before_loading_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "FREEZE.md"
            document.write_text("Frozen synthetic service protocol.\n")
            source = root / "implementation.py"
            source.write_text("VERSION = 1\n")
            spec = {
                "freeze_record": str(root / "freeze.json"),
                "implementation_sources": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
            }
            frozen = {
                "state": "FROZEN",
                "freeze_document": {
                    "path": str(document),
                    "sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
                },
                "included_c4": {"spec_method_sha256": service.method_sha(spec)},
            }
            Path(spec["freeze_record"]).write_text(json.dumps(frozen))
            spec["freeze_record_sha256"] = hashlib.sha256(Path(spec["freeze_record"]).read_bytes()).hexdigest()
            with (
                patch.object(service, "ROOT", root),
                patch.object(service, "validate_spec"),
                patch.object(service, "load_tasks_after_freeze_impl", return_value={}) as loader,
            ):
                tasks, identity = service.load_tasks_after_freeze(spec)
                self.assertEqual(tasks, {})
                self.assertEqual(identity["spec_method_sha256"], service.method_sha(spec))
                loader.assert_called_once_with(spec)
                loader.reset_mock()
                source.write_text("VERSION = 2\n")
                with self.assertRaisesRegex(ValueError, "Exact artifact SHA mismatch"):
                    service.load_tasks_after_freeze(spec)
                loader.assert_not_called()

    def test_planned_denominator_keeps_unfinished_arms(self):
        entries = [
            {"kind": "serial", "status": "FAILED_SERVICE"},
            {"kind": "serial", "status": "PENDING"},
            {"kind": "poisson", "status": "PENDING", "scheduled_requests": 40},
        ]
        self.assertEqual(service.planned_requests(entries), 42)


if __name__ == "__main__":
    unittest.main()
