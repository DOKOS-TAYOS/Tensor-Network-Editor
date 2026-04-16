from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_runtime_script(tmp_path: Path, filename: str, body: str) -> Path:
    script_path = tmp_path / filename
    script_path.write_text(textwrap.dedent(body), encoding="utf-8")
    return script_path


def _run_runtime_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_editor_store_and_selectors_track_template_catalog_state(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "editor_store_selectors.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const stateUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state.js")!r}).href;
        const storeUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state" / "editorStore.js")!r}).href;
        const selectorsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state" / "editorSelectors.js")!r}).href;

        const [stateModule, storeModule, selectorsModule] = await Promise.all([
          import(stateUrl),
          import(storeUrl),
          import(selectorsUrl),
        ]);

        const store = storeModule.createEditorStore(stateModule.createInitialState());
        store.setSchemaVersion(4);
        store.setAvailableCollectionFormats(["list", "dict"]);
        store.setSelectedEngine("quimb");
        store.setSelectedCollectionFormat("dict");
        store.setTemplateCatalogData({{
          templateNames: ["project_pair", "mps"],
          templateDefinitions: {{
            project_pair: {{ display_name: "Project Pair", source: "project" }},
            mps: {{ display_name: "MPS", source: "global" }},
          }},
          templateCatalogWarnings: ["warning one"],
        }});

        const selectors = selectorsModule.createEditorSelectors({{ store }});
        if (store.getState().schemaVersion !== 4) {{
          throw new Error(`Expected schema version 4, received ${{store.getState().schemaVersion}}.`);
        }}
        if (selectors.getTemplateDefinition("project_pair").display_name !== "Project Pair") {{
          throw new Error("Project template definition was not preserved.");
        }}
        if (!selectors.isProjectTemplate("project_pair")) {{
          throw new Error("Project template should be recognized as project-local.");
        }}
        if (selectors.isProjectTemplate("mps")) {{
          throw new Error("Global template should not be recognized as project-local.");
        }}
        if (!selectors.hasTemplateCatalogWarnings()) {{
          throw new Error("Expected warning state to be tracked.");
        }}
        if (selectors.getSelectedEngine() !== "quimb") {{
          throw new Error(`Expected selected engine quimb, received ${{selectors.getSelectedEngine()}}.`);
        }}
        if (selectors.getSelectedCollectionFormat() !== "dict") {{
          throw new Error(`Expected selected collection format dict, received ${{selectors.getSelectedCollectionFormat()}}.`);
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The editor store/selectors runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_editor_services_route_session_requests_through_explicit_dependencies(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "editor_services.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const sessionServiceUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "services" / "editorSessionService.js")!r}).href;
        const templateServiceUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "services" / "templateCatalogService.js")!r}).href;

        const [sessionModule, templateModule] = await Promise.all([
          import(sessionServiceUrl),
          import(templateServiceUrl),
        ]);

        const calls = [];
        const apiGet = async (path) => {{
          calls.push({{ method: "GET", path }});
          return {{ ok: true }};
        }};
        const apiPost = async (path, payload) => {{
          calls.push({{ method: "POST", path, payload }});
          return {{ ok: true }};
        }};

        const sessionService = sessionModule.createEditorSessionService({{ apiGet, apiPost }});
        const templateService = templateModule.createTemplateCatalogService({{ apiPost }});

        await sessionService.loadBootstrap();
        await sessionService.generateCode({{
          engine: "quimb",
          collectionFormat: "dict",
          spec: {{ schema_version: 4, network: {{ id: "network_demo" }} }},
        }});
        await templateService.renameTemplate({{
          templateName: "project_pair",
          newTemplateName: "renamed_pair",
          overwrite: true,
        }});

        if (calls[0].method !== "GET" || calls[0].path !== "/api/bootstrap") {{
          throw new Error(`Unexpected bootstrap call: ${{JSON.stringify(calls[0])}}`);
        }}
        if (calls[1].path !== "/api/generate") {{
          throw new Error(`Unexpected generate path: ${{calls[1].path}}`);
        }}
        if (calls[1].payload.collection_format !== "dict") {{
          throw new Error(`Expected collection_format=dict, received ${{calls[1].payload.collection_format}}.`);
        }}
        if (calls[2].path !== "/api/template/rename") {{
          throw new Error(`Unexpected template rename path: ${{calls[2].path}}`);
        }}
        if (calls[2].payload.new_template_name !== "renamed_pair" || calls[2].payload.overwrite !== true) {{
          throw new Error(`Unexpected rename payload: ${{JSON.stringify(calls[2].payload)}}`);
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The editor services runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )
