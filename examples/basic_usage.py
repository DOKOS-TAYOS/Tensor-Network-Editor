"""Launch the local editor and print the confirmed result."""

from __future__ import annotations

from tensor_network_editor import EngineName
from tensor_network_editor.editor import EditorLaunchOptions, open_editor


def main() -> None:
    """Run a small example session against the NumPy einsum backend."""
    result = open_editor(
        options=EditorLaunchOptions(default_engine=EngineName.EINSUM_NUMPY)
    )
    if result is None:
        print("Editor cancelled.")
        return

    print(f"Design name: {result.spec.name}")
    if result.codegen is not None:
        print("Generated code:")
        print(result.codegen.code)


if __name__ == "__main__":
    main()
