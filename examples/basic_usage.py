from __future__ import annotations

from tensor_network_editor import EngineName, launch_tensor_network_editor


def main() -> None:
    result = launch_tensor_network_editor(default_engine=EngineName.EINSUM_NUMPY)
    if result is None:
        print("Editor cancelled.")
        return

    print(f"Design name: {result.spec.name}")
    if result.codegen is not None:
        print("Generated code:")
        print(result.codegen.code)


if __name__ == "__main__":
    main()
