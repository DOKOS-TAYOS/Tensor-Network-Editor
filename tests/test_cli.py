from __future__ import annotations

import unittest
from unittest.mock import patch

from tensor_network_editor.models import EngineName


class CliTests(unittest.TestCase):
    @patch("tensor_network_editor.cli.launch_tensor_network_editor")
    def test_main_passes_engine_and_browser_flags(self, launch_mock) -> None:
        from tensor_network_editor.cli import main

        exit_code = main(["--engine", "einsum_numpy", "--no-browser"])

        self.assertEqual(exit_code, 0)
        launch_mock.assert_called_once()
        _, kwargs = launch_mock.call_args
        self.assertEqual(kwargs["default_engine"], EngineName.EINSUM_NUMPY)
        self.assertFalse(kwargs["open_browser"])

    @patch(
        "tensor_network_editor.cli.launch_tensor_network_editor",
        side_effect=KeyboardInterrupt,
    )
    def test_main_returns_130_on_keyboard_interrupt(self, launch_mock) -> None:
        from tensor_network_editor.cli import main

        exit_code = main([])

        self.assertEqual(exit_code, 130)
        launch_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
