from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from tensor_network_editor.app.server import EditorServer
from tensor_network_editor.app.session import EditorSession
from tensor_network_editor.models import EngineName
from tests.app_support import request_json, request_json_with_status
from tests.test_api import build_sample_spec


class AppRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = EditorSession(
            initial_spec=build_sample_spec(),
            default_engine=EngineName.EINSUM,
        )
        self.server = EditorServer(self.session)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_bootstrap_returns_initial_spec_and_engine(self) -> None:
        payload = request_json(f"{self.server.base_url}/api/bootstrap")

        self.assertEqual(payload["default_engine"], EngineName.EINSUM.value)
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["spec"]["network"]["id"], "network_demo")
        self.assertEqual(payload["spec"]["schema_version"], 3)
        self.assertIn(EngineName.QUIMB.value, payload["engines"])
        self.assertEqual(
            payload["templates"], ["mps", "mpo", "peps_2x2", "mera", "binary_tree"]
        )

    def test_validate_route_reports_issues_for_invalid_spec(self) -> None:
        invalid_spec = build_sample_spec()
        invalid_spec.edges.append(invalid_spec.edges[0])

        payload = request_json(
            f"{self.server.base_url}/api/validate",
            method="POST",
            payload={"spec": {"schema_version": 3, "network": invalid_spec.to_dict()}},
        )

        self.assertFalse(payload["ok"])
        self.assertIn(
            "index-already-connected", [issue["code"] for issue in payload["issues"]]
        )

    def test_validate_route_rejects_invalid_json_with_400(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/validate",
            method="POST",
            raw_body=b"{not-json}",
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("invalid JSON", payload["message"])

    def test_validate_route_rejects_non_object_json_payload_with_400(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/validate",
            method="POST",
            raw_body=json.dumps(["not", "an", "object"]).encode("utf-8"),
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("JSON object", payload["message"])

    def test_validate_route_rejects_legacy_schema_versions(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/validate",
            method="POST",
            payload={
                "spec": {"schema_version": 2, "network": build_sample_spec().to_dict()}
            },
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("Unsupported schema version", payload["message"])

    def test_generate_route_returns_code(self) -> None:
        payload = request_json(
            f"{self.server.base_url}/api/generate",
            method="POST",
            payload={
                "engine": EngineName.TENSORNETWORK.value,
                "spec": {"schema_version": 3, "network": build_sample_spec().to_dict()},
            },
        )

        self.assertEqual(payload["engine"], EngineName.TENSORNETWORK.value)
        self.assertIn("import tensornetwork as tn", payload["code"])

    def test_generate_route_rejects_missing_spec_with_400(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/generate",
            method="POST",
            payload={"engine": EngineName.TENSORNETWORK.value},
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("Missing 'spec' payload.", payload["message"])

    def test_generate_route_rejects_unsupported_engine_with_400(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/generate",
            method="POST",
            payload={
                "engine": "unknown-engine",
                "spec": {"schema_version": 3, "network": build_sample_spec().to_dict()},
            },
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("Unsupported engine", payload["message"])

    def test_generate_route_returns_validation_issues_for_invalid_spec(self) -> None:
        invalid_spec = build_sample_spec()
        invalid_spec.edges.append(invalid_spec.edges[0])

        payload = request_json(
            f"{self.server.base_url}/api/generate",
            method="POST",
            payload={
                "engine": EngineName.TENSORNETWORK.value,
                "spec": {"schema_version": 3, "network": invalid_spec.to_dict()},
            },
        )

        self.assertFalse(payload["ok"])
        self.assertIn(
            "index-already-connected", [issue["code"] for issue in payload["issues"]]
        )

    def test_complete_route_stores_result_in_session(self) -> None:
        payload = request_json(
            f"{self.server.base_url}/api/complete",
            method="POST",
            payload={
                "engine": EngineName.QUIMB.value,
                "spec": {"schema_version": 3, "network": build_sample_spec().to_dict()},
            },
        )

        self.assertTrue(payload["ok"])
        result = self.session.wait_for_result(timeout=0.1)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.engine, EngineName.QUIMB)
        self.assertIsNotNone(result.codegen)

    def test_cancel_route_ends_session_without_result(self) -> None:
        payload = request_json(
            f"{self.server.base_url}/api/cancel",
            method="POST",
            payload={},
        )

        self.assertTrue(payload["ok"])
        self.assertIsNone(self.session.wait_for_result(timeout=0.1))

    def test_autolayout_route_is_not_available(self) -> None:
        status, payload = request_json_with_status(
            f"{self.server.base_url}/api/autolayout",
            method="POST",
            payload={
                "spec": {"schema_version": 3, "network": build_sample_spec().to_dict()}
            },
        )

        self.assertEqual(status, 404)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["message"], "Not found.")

    def test_template_route_returns_valid_serialized_spec(self) -> None:
        payload = request_json(
            f"{self.server.base_url}/api/template",
            method="POST",
            payload={"template": "mps"},
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["spec"]["schema_version"], 3)
        self.assertEqual(payload["spec"]["network"]["name"], "MPS")
        self.assertGreater(len(payload["spec"]["network"]["tensors"]), 0)

    def test_analyze_contraction_route_returns_manual_summary(self) -> None:
        payload = request_json(
            f"{self.server.base_url}/api/analyze-contraction",
            method="POST",
            payload={
                "spec": {"schema_version": 3, "network": build_sample_spec().to_dict()}
            },
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["automatic_strategy"], "greedy")
        self.assertEqual(payload["manual"]["status"], "complete")
        self.assertEqual(payload["manual"]["summary"]["total_estimated_flops"], 24)
        self.assertEqual(payload["manual"]["summary"]["final_shape"], [2, 4])
        self.assertIn(payload["automatic"]["status"], {"complete", "unavailable"})
        if payload["automatic"]["status"] == "complete":
            self.assertEqual(payload["automatic"]["summary"]["final_shape"], [2, 4])

    def test_unexpected_server_errors_return_generic_500_payload(self) -> None:
        with patch(
            "tensor_network_editor.app.server.routes.handle_generate",
            side_effect=RuntimeError("boom"),
        ):
            status, payload = request_json_with_status(
                f"{self.server.base_url}/api/generate",
                method="POST",
                payload={
                    "engine": EngineName.TENSORNETWORK.value,
                    "spec": {
                        "schema_version": 3,
                        "network": build_sample_spec().to_dict(),
                    },
                },
            )

        self.assertEqual(status, 500)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["message"], "Internal server error.")


if __name__ == "__main__":
    unittest.main()
