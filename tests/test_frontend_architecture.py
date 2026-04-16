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


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_graph_view_modules_build_and_apply_descriptor_diffs(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "graph_view_modules.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const descriptorsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "views" / "graphDescriptors.js")!r}).href;
        const diffUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "views" / "graphModelDiff.js")!r}).href;
        const adapterUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "views" / "cytoscapeGraphAdapter.js")!r}).href;

        const [descriptorsModule, diffModule, adapterModule] = await Promise.all([
          import(descriptorsUrl),
          import(diffUrl),
          import(adapterUrl),
        ]);

        const firstModel = {{
          orderedIds: ["tensor_a"],
          visibleSignature: "tensor_a",
          ephemeralSignature: "editable",
          descriptorsById: {{
            tensor_a: {{
              group: "nodes",
              classes: "",
              data: {{
                id: "tensor_a",
                label: "A",
                kind: "tensor",
              }},
              position: {{ x: 10, y: 20 }},
              grabbable: true,
              selectable: true,
            }},
          }},
          elements: [],
        }};
        firstModel.elements = firstModel.orderedIds.map((elementId) => firstModel.descriptorsById[elementId]);

        const secondModel = {{
          orderedIds: ["tensor_a", "tensor_b"],
          visibleSignature: "tensor_a|tensor_b",
          ephemeralSignature: "editable",
          descriptorsById: {{
            tensor_a: {{
              group: "nodes",
              classes: "planner-pending-tensor",
              data: {{
                id: "tensor_a",
                label: "Tensor A",
                kind: "tensor",
              }},
              position: {{ x: 30, y: 40 }},
              grabbable: true,
              selectable: true,
            }},
            tensor_b: {{
              group: "nodes",
              classes: "",
              data: {{
                id: "tensor_b",
                label: "B",
                kind: "tensor",
              }},
              position: {{ x: 80, y: 90 }},
              grabbable: false,
              selectable: false,
            }},
          }},
          elements: [],
        }};
        secondModel.elements = secondModel.orderedIds.map((elementId) => secondModel.descriptorsById[elementId]);

        if (descriptorsModule.graphElementDescriptorsEqual(firstModel.descriptorsById.tensor_a, secondModel.descriptorsById.tensor_a)) {{
          throw new Error("Updated tensor descriptor should not compare equal.");
        }}
        const updatePlan = diffModule.buildGraphElementUpdatePlan({{
          previousDescriptorsById: firstModel.descriptorsById,
          nextModel: secondModel,
        }});
        if (updatePlan.removedIds.length !== 0) {{
          throw new Error(`Expected no removals, received ${{updatePlan.removedIds.length}}.`);
        }}
        if (updatePlan.addedDescriptors.length !== 1 || updatePlan.addedDescriptors[0].data.id !== "tensor_b") {{
          throw new Error(`Expected tensor_b as the only added descriptor, received ${{JSON.stringify(updatePlan.addedDescriptors)}}.`);
        }}
        if (updatePlan.updatedDescriptors.length !== 1 || updatePlan.updatedDescriptors[0].id !== "tensor_a") {{
          throw new Error(`Expected tensor_a as the only updated descriptor, received ${{JSON.stringify(updatePlan.updatedDescriptors)}}.`);
        }}

        const stats = {{
          addCalls: 0,
          removeCalls: 0,
          dataUpdates: 0,
          positionUpdates: 0,
          classUpdates: 0,
          selectableUpdates: 0,
          grabbableUpdates: 0,
        }};
        const elements = new Map();
        const createElement = (descriptor) => {{
          const state = {{
            descriptor: descriptorsModule.cloneGraphElementDescriptor(descriptor),
          }};
          return {{
            length: 1,
            remove() {{
              stats.removeCalls += 1;
              elements.delete(state.descriptor.data.id);
            }},
            data(nextData) {{
              stats.dataUpdates += 1;
              state.descriptor.data = {{ ...nextData }};
            }},
            position(nextPosition) {{
              stats.positionUpdates += 1;
              state.descriptor.position = nextPosition ? {{ ...nextPosition }} : null;
            }},
            classes(nextClasses) {{
              stats.classUpdates += 1;
              state.descriptor.classes = nextClasses;
            }},
            selectable(nextSelectable) {{
              stats.selectableUpdates += 1;
              state.descriptor.selectable = nextSelectable;
            }},
            grabbable(nextGrabbable) {{
              stats.grabbableUpdates += 1;
              state.descriptor.grabbable = nextGrabbable;
            }},
          }};
        }};
        const cy = {{
          add(descriptors) {{
            stats.addCalls += 1;
            const normalizedDescriptors = Array.isArray(descriptors) ? descriptors : [descriptors];
            normalizedDescriptors.forEach((descriptor) => {{
              elements.set(descriptor.data.id, createElement(descriptor));
            }});
          }},
          getElementById(elementId) {{
            return elements.get(elementId) || {{ length: 0 }};
          }},
        }};
        const state = {{
          cy,
          graphRenderCyRef: null,
          graphRenderDescriptorById: {{}},
          graphRenderDescriptorOrder: [],
          graphRenderVisibleSignature: null,
          graphRenderEphemeralSignature: null,
          graphRenderDescriptorRevision: -1,
          cySelectionSyncedIds: [],
          pendingInteractionRenderedPlannerSelectionId: null,
          pendingInteractionRenderedIndexId: null,
          specRevision: 5,
        }};
        const adapter = adapterModule.createCytoscapeGraphAdapter({{
          state,
          getCy: () => cy,
        }});

        adapter.ensureForCurrentCy();
        adapter.applyModel(firstModel);
        adapter.applyModel(secondModel);

        if (stats.addCalls !== 2) {{
          throw new Error(`Expected two add batches, received ${{stats.addCalls}}.`);
        }}
        if (stats.removeCalls !== 0) {{
          throw new Error(`Expected no removals, received ${{stats.removeCalls}}.`);
        }}
        if (stats.dataUpdates !== 1 || stats.positionUpdates !== 1 || stats.classUpdates !== 1) {{
          throw new Error(`Expected in-place updates for tensor_a, received ${{JSON.stringify(stats)}}.`);
        }}
        if (state.graphRenderVisibleSignature !== secondModel.visibleSignature) {{
          throw new Error("The graph render cache did not track the visible signature.");
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The graph-view module runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_contraction_scene_modules_expose_progression_and_snapshot_helpers(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "contraction_scene_modules.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const progressionUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state" / "contractionSceneProgression.js")!r}).href;
        const snapshotsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state" / "contractionSceneSnapshots.js")!r}).href;

        const [progressionModule, snapshotsModule] = await Promise.all([
          import(progressionUrl),
          import(snapshotsUrl),
        ]);

        const initialOperands = [
          {{
            id: "tensor_a",
            name: "A",
            isDerived: false,
            sourceTensorIds: ["tensor_a"],
            tokens: [{{ key: "edge:ab", name: "ab", dimension: 3, textColorSeed: "ab", sourceEdgeId: "edge_ab", sourceIndexId: null }}],
          }},
          {{
            id: "tensor_b",
            name: "B",
            isDerived: false,
            sourceTensorIds: ["tensor_b"],
            tokens: [{{ key: "edge:ab", name: "ab", dimension: 3, textColorSeed: "ab", sourceEdgeId: "edge_ab", sourceIndexId: null }}],
          }},
          {{
            id: "tensor_c",
            name: "C",
            isDerived: false,
            sourceTensorIds: ["tensor_c"],
            tokens: [{{ key: "open:c", name: "c", dimension: 5, textColorSeed: "c", sourceEdgeId: null, sourceIndexId: "idx_c" }}],
          }},
        ];
        const progression = progressionModule.buildContractionOperandProgression({{
          initialOperands,
          planSteps: [
            {{ id: "step_ab", left_operand_id: "tensor_a", right_operand_id: "tensor_b", metadata: {{}} }},
            {{ id: "step_root", left_operand_id: "step_ab", right_operand_id: "tensor_c", metadata: {{}} }},
          ],
          previousOperandId: "__linear_previous__",
          nextOperandId: "__linear_next__",
        }});

        if (progression.validSteps.length !== 2) {{
          throw new Error(`Expected 2 valid steps, received ${{progression.validSteps.length}}.`);
        }}
        const finalState = progressionModule.buildContractionStateFromProgression(progression, 2);
        if (finalState.activeOperands.length !== 1 || finalState.activeOperands[0].id !== "step_root") {{
          throw new Error(`Expected the root operand to remain active, received ${{JSON.stringify(finalState.activeOperands)}}.`);
        }}

        const snapshot = {{
          applied_step_count: 1,
          operand_layouts: [
            {{
              operand_id: "step_ab",
              position: {{ x: 120, y: 80 }},
              size: {{ width: 160, height: 90 }},
            }},
          ],
        }};
        const layoutMap = snapshotsModule.buildSnapshotLayoutMap(snapshot);
        if (layoutMap.step_ab.position.x !== 120 || layoutMap.step_ab.size.width !== 160) {{
          throw new Error(`Unexpected snapshot layout map: ${{JSON.stringify(layoutMap)}}.`);
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The contraction-scene module runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_planner_and_property_modules_use_explicit_internal_contracts(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "planner_property_modules.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const plannerSelectorsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state" / "plannerSelectors.js")!r}).href;
        const plannerCommandsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "actions" / "plannerCommands.js")!r}).href;
        const plannerServiceUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "services" / "plannerAnalysisService.js")!r}).href;
        const propertySummariesUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "properties" / "propertySummaries.js")!r}).href;
        const propertyMetadataUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "properties" / "metadataEditors.js")!r}).href;
        const propertyCommandsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "actions" / "propertyCommands.js")!r}).href;

        const [
          plannerSelectorsModule,
          plannerCommandsModule,
          plannerServiceModule,
          propertySummariesModule,
          propertyMetadataModule,
          propertyCommandsModule,
        ] = await Promise.all([
          import(plannerSelectorsUrl),
          import(plannerCommandsUrl),
          import(plannerServiceUrl),
          import(propertySummariesUrl),
          import(propertyMetadataUrl),
          import(propertyCommandsUrl),
        ]);

        const plannerOperandState = plannerSelectorsModule.buildPlannerOperandState({{
          tensors: [{{ id: "tensor_a" }}, {{ id: "tensor_b" }}],
          steps: [{{ id: "step_ab", left_operand_id: "tensor_a", right_operand_id: "tensor_b" }}],
          seedOperands: [
            {{ id: "tensor_a", sourceTensorIds: ["tensor_a"], selectionIds: ["tensor_a"] }},
            {{ id: "tensor_b", sourceTensorIds: ["tensor_b"], selectionIds: ["tensor_b"] }},
          ],
          previousOperandId: "__linear_previous__",
          nextOperandId: "__linear_next__",
        }});
        if (plannerOperandState.validSteps.length !== 1 || plannerOperandState.activeOperandIds[0] !== "step_ab") {{
          throw new Error(`Unexpected planner operand state: ${{JSON.stringify(plannerOperandState)}}.`);
        }}

        const plannerState = {{
          plannerMode: true,
          pendingPlannerOperandId: null,
          pendingPlannerSelectionId: null,
        }};
        const plannerCalls = [];
        const plannerCommands = plannerCommandsModule.createPlannerCommands({{
          state: plannerState,
          resolvePlannerOperandId: (operandId) => operandId,
          getPlannerOperandLabel: (operandId) => operandId,
          setStatus: (message, level = "info") => plannerCalls.push({{ message, level }}),
          setActiveSidebarTab: (tabId) => plannerCalls.push({{ tabId }}),
          renderPlanner: () => plannerCalls.push("renderPlanner"),
          renderOverlayDecorations: () => plannerCalls.push("renderOverlayDecorations"),
          syncPendingInteractionClasses: () => plannerCalls.push("syncPendingInteractionClasses"),
          isInspectingPastStage: () => false,
          applyManualContractionStep: (leftOperandId, rightOperandId) =>
            plannerCalls.push({{ leftOperandId, rightOperandId }}),
        }});
        plannerCommands.handlePlannerOperandClick("tensor_a");
        if (plannerState.pendingPlannerOperandId !== "tensor_a" || plannerState.pendingPlannerSelectionId !== "tensor_a") {{
          throw new Error(`Unexpected pending planner selection state: ${{JSON.stringify(plannerState)}}.`);
        }}

        const analysisCalls = [];
        let queuedCallback = null;
        const analysisService = plannerServiceModule.createPlannerAnalysisService({{
          analysisRefreshDelayMs: 25,
          schedule: (callback) => {{
            queuedCallback = callback;
            return 99;
          }},
          cancel: (timerId) => analysisCalls.push({{ cancelled: timerId }}),
          analyze: async (payload) => {{
            analysisCalls.push(payload);
            return {{ status: "ready", payload: {{ automatic_future: {{ steps: [] }} }} }};
          }},
          serializeCurrentSpec: () => ({{ network: {{ id: "demo" }} }}),
          onAnalysisResult: (result) => analysisCalls.push({{ resultStatus: result.status }}),
          onRenderRequested: () => analysisCalls.push("render"),
        }});
        analysisService.requestRefresh({{ reason: "first" }});
        analysisService.requestRefresh({{ reason: "latest" }});
        await queuedCallback();
        if (analysisCalls.filter((entry) => entry && entry.reason).length !== 1) {{
          throw new Error(`Expected one queued analysis call, received ${{JSON.stringify(analysisCalls)}}.`);
        }}
        if (!analysisCalls.some((entry) => entry && entry.reason === "latest")) {{
          throw new Error(`Expected the latest analysis request to win, received ${{JSON.stringify(analysisCalls)}}.`);
        }}

        const tensor = {{
          id: "tensor_a",
          indices: [
            {{ id: "index_left", dimension: 2 }},
            {{ id: "index_right", dimension: 3 }},
          ],
        }};
        const totalElementCount = propertySummariesModule.getTensorTotalElementCount(
          tensor,
          (value, fallbackValue) => Number.isFinite(Number(value)) ? Number(value) : fallbackValue
        );
        if (totalElementCount.toString() !== "6") {{
          throw new Error(`Expected 6 tensor elements, received ${{totalElementCount}}.`);
        }}

        const metadataSupport = propertyMetadataModule.createMetadataEditorSupport({{
          escapeHtml: (value) => String(value),
          isObject: (value) => Boolean(value) && typeof value === "object" && !Array.isArray(value),
          annotationDefinitionsByScope: {{
            tensor: [
              {{
                key: "role",
                label: "Role",
                placeholder: "physical",
                suggestions: ["physical"],
              }},
            ],
          }},
        }});
        const parsedMetadata = metadataSupport.parseCustomMetadataValue(
          '{{"color": "#fff", "role": "physical", "kept": 7}}',
          "tensor"
        );
        if (parsedMetadata.kept !== 7 || Object.prototype.hasOwnProperty.call(parsedMetadata, "color") || Object.prototype.hasOwnProperty.call(parsedMetadata, "role")) {{
          throw new Error(`Unexpected custom metadata parsing result: ${{JSON.stringify(parsedMetadata)}}.`);
        }}

        const propertyEvents = [];
        const propertyCommands = propertyCommandsModule.createPropertyCommands({{
          applyDesignChange: (mutate, options = {{}}) => {{
            mutate();
            propertyEvents.push(options.statusMessage || null);
          }},
          setStatus: (message, level = "info") => propertyEvents.push(`${{level}}:${{message}}`),
          findIndexOwner: (indexId) =>
            indexId === "index_left"
              ? {{ tensor, index: tensor.indices[0] }}
              : null,
          syncConnectedIndexDimension: (indexId, nextDimension) =>
            propertyEvents.push(`sync:${{indexId}}:${{nextDimension}}`),
          tensorIndexNameExists: () => false,
        }});
        propertyCommands.updateIndexDimension({{
          indexId: "index_left",
          rawValue: "5",
          invalidate: {{ graph: true, analysis: true }},
          statusMessage: "Updated index index_left.",
        }});
        if (tensor.indices[0].dimension !== 5) {{
          throw new Error(`Expected index dimension 5, received ${{tensor.indices[0].dimension}}.`);
        }}
        if (!propertyEvents.includes("sync:index_left:5")) {{
          throw new Error(`Expected connected-dimension sync, received ${{JSON.stringify(propertyEvents)}}.`);
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The planner/property module runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )
