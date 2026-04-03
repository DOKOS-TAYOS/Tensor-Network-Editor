from __future__ import annotations

import argparse
from collections.abc import Sequence

from .api import launch_tensor_network_editor, load_spec
from .models import EngineName


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensor-network-editor",
        description="Launch the local tensor network editor in your browser.",
    )
    parser.add_argument(
        "--engine",
        choices=[engine.value for engine in EngineName],
        default=EngineName.TENSORNETWORK.value,
        help="Default target engine shown in the editor.",
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Optional path to a saved JSON design to preload.",
    )
    parser.add_argument(
        "--save-code",
        type=str,
        help="Optional output path for the generated Python code when the editor is confirmed.",
    )
    parser.add_argument(
        "--print-code",
        action="store_true",
        help="Print generated code to stdout when the editor is confirmed.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the local server without opening the browser automatically.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    initial_spec = load_spec(args.load) if args.load else None
    try:
        launch_tensor_network_editor(
            initial_spec=initial_spec,
            default_engine=EngineName(args.engine),
            open_browser=not args.no_browser,
            print_code=args.print_code,
            code_path=args.save_code,
        )
    except KeyboardInterrupt:
        return 130
    return 0
