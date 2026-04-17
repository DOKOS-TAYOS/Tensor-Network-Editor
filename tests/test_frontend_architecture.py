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
          isMetadataDisclosureOpen: (disclosureKey) => disclosureKey === "network:metadata",
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
        if (
          parsedMetadata.kept !== 7 ||
          parsedMetadata.role !== "physical" ||
          Object.prototype.hasOwnProperty.call(parsedMetadata, "color")
        ) {{
          throw new Error(`Unexpected custom metadata parsing result: ${{JSON.stringify(parsedMetadata)}}.`);
        }}

        const propertyEvents = [];
        const spec = {{
          name: "Demo network",
          groups: [{{ id: "group_a", name: "Group A", tensor_ids: ["tensor_a"], metadata: {{ color: "#123456" }} }}],
          edges: [{{ id: "edge_a", name: "edge_a", metadata: {{ color: "#654321" }} }}],
          notes: [{{ id: "note_a", text: "Original note", metadata: {{ color: "#abcdef" }} }}],
        }};
        const tensorById = {{
          tensor_a: tensor,
          tensor_b: {{
            id: "tensor_b",
            indices: [],
            metadata: {{}},
          }},
        }};
        const applyColorToSelection = (nextColor) => {{
          Object.values(tensorById).forEach((candidate) => {{
            candidate.metadata = candidate.metadata || {{}};
            candidate.metadata.color = nextColor;
          }});
        }};
        const propertyCommands = propertyCommandsModule.createPropertyCommands({{
          applyDesignChange: (mutate, options = {{}}) => {{
            mutate();
            propertyEvents.push(options.statusMessage || null);
          }},
          applyColorToSelection,
          createIndex: (targetTensor, indexPosition) => ({{
            id: `${{targetTensor.id}}_index_${{indexPosition}}`,
            name: `i${{indexPosition + 1}}`,
            dimension: 2,
            metadata: {{}},
          }}),
          findTensorById: (tensorId) => tensorById[tensorId] || null,
          setStatus: (message, level = "info") => propertyEvents.push(`${{level}}:${{message}}`),
          findIndexOwner: (indexId) =>
            indexId === "index_left"
              ? {{ tensor, index: tensor.indices[0] }}
              : null,
          getSelectedTensorIds: () => ["tensor_a", "tensor_b"],
          removeGroup: (groupId) => {{
            spec.groups = spec.groups.filter((candidate) => candidate.id !== groupId);
          }},
          removeEdge: (edgeId) => {{
            spec.edges = spec.edges.filter((candidate) => candidate.id !== edgeId);
          }},
          removeNote: (noteId) => {{
            spec.notes = spec.notes.filter((candidate) => candidate.id !== noteId);
          }},
          syncConnectedIndexDimension: (indexId, nextDimension) =>
            propertyEvents.push(`sync:${{indexId}}:${{nextDimension}}`),
          tensorIndexNameExists: () => false,
        }});
        propertyCommands.renameNetwork({{
          spec,
          proposedName: "Refined network",
          invalidate: {{ properties: true }},
          statusMessage: "Updated design name.",
        }});
        propertyCommands.applySelectionColor({{
          nextColor: "#ff8800",
          invalidate: {{ graph: true }},
          statusMessage: "Updated the selection color.",
        }});
        propertyCommands.addIndexToSelectedTensors({{
          selectionIds: ["tensor_a", "tensor_b"],
          primaryId: "tensor_b",
          statusMessage: "Added one index to each selected tensor.",
        }});
        propertyCommands.renameGroup({{
          group: spec.groups[0],
          proposedName: "Cluster A",
          invalidate: {{ overlays: true }},
          statusMessage: "Updated group Cluster A.",
        }});
        propertyCommands.renameEdge({{
          edge: spec.edges[0],
          proposedName: "bond_main",
          invalidate: {{ graph: true }},
          statusMessage: "Updated connection bond_main.",
        }});
        propertyCommands.updateNoteText({{
          note: spec.notes[0],
          proposedText: "Updated note",
          invalidate: {{ overlays: true }},
          statusMessage: "Updated the note.",
        }});
        propertyCommands.updateIndexDimension({{
          indexId: "index_left",
          rawValue: "5",
          invalidate: {{ graph: true, analysis: true }},
          statusMessage: "Updated index index_left.",
        }});
        propertyCommands.deleteGroup({{
          groupId: "group_a",
          selectionIds: [],
          invalidate: {{ overlays: true, lookups: true }},
          statusMessage: "Deleted group Cluster A.",
        }});
        propertyCommands.deleteEdge({{
          edgeId: "edge_a",
          selectionIds: [],
          statusMessage: "Deleted connection bond_main.",
        }});
        propertyCommands.deleteNote({{
          noteId: "note_a",
          selectionIds: [],
          invalidate: {{ overlays: true, lookups: true }},
          statusMessage: "Deleted the note.",
        }});
        if (spec.name !== "Refined network") {{
          throw new Error(`Expected renamed network, received ${{spec.name}}.`);
        }}
        if (tensorById.tensor_a.metadata.color !== "#ff8800" || tensorById.tensor_b.metadata.color !== "#ff8800") {{
          throw new Error(`Expected batch color to update all selected tensors, received ${{JSON.stringify(tensorById)}}.`);
        }}
        if (tensor.indices.length !== 3 || tensorById.tensor_b.indices.length !== 1) {{
          throw new Error(`Expected addIndexToSelectedTensors to append indices, received ${{JSON.stringify(tensorById)}}.`);
        }}
        if (spec.groups.length !== 0) {{
          throw new Error(`Expected deleteGroup to remove the group, received ${{JSON.stringify(spec.groups)}}.`);
        }}
        if (spec.edges.length !== 0) {{
          throw new Error(`Expected deleteEdge to remove the edge, received ${{JSON.stringify(spec.edges)}}.`);
        }}
        if (spec.notes.length !== 0) {{
          throw new Error(`Expected deleteNote to remove the note, received ${{JSON.stringify(spec.notes)}}.`);
        }}
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


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_metadata_autocomplete_and_canvas_context_menu_modules_support_new_ui(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "metadata_autocomplete_and_context_menu.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const propertyMetadataUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "properties" / "metadataEditors.js")!r}).href;
        const canvasContextMenuUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "canvasContextMenu.js")!r}).href;

        const [propertyMetadataModule, canvasContextMenuModule] = await Promise.all([
          import(propertyMetadataUrl),
          import(canvasContextMenuUrl),
        ]);

        const metadataSupport = propertyMetadataModule.createMetadataEditorSupport({{
          escapeHtml: (value) => String(value),
          isObject: (value) => Boolean(value) && typeof value === "object" && !Array.isArray(value),
          isMetadataDisclosureOpen: (disclosureKey) => disclosureKey === "network:metadata",
          annotationDefinitionsByScope: {{
            tensor: [
              {{
                key: "role",
                label: "Role",
                placeholder: "observable",
                suggestions: ["operator", "observable"],
              }},
            ],
          }},
          tagSuggestionsByScope: {{
            tensor: ["observable", "quimb", "tensornetwork"],
          }},
        }});

        const autocompleteSuggestions =
          metadataSupport.buildTagAutocompleteSuggestions("tensor", "ob");
        if (JSON.stringify(autocompleteSuggestions) !== JSON.stringify(["observable"])) {{
          throw new Error(`Expected existing tags to win over guided duplicates, received ${{JSON.stringify(autocompleteSuggestions)}}.`);
        }}
        const autocompleteSuggestionsWithOrdering =
          metadataSupport.buildTagAutocompleteSuggestions("tensor", "o");
        if (JSON.stringify(autocompleteSuggestionsWithOrdering) !== JSON.stringify(["observable", "operator"])) {{
          throw new Error(`Expected existing tags first and guided suggestions after them, received ${{JSON.stringify(autocompleteSuggestionsWithOrdering)}}.`);
        }}
        const replacedTagValue =
          metadataSupport.replaceActiveTagToken("alpha, ob", "observable");
        if (replacedTagValue !== "alpha, observable") {{
          throw new Error(`Expected active tag token replacement, received ${{replacedTagValue}}.`);
        }}
        const metadataMarkup = metadataSupport.buildMetadataEditorMarkup({{
          tagsInputId: "network-tags-input",
          tagsFocusKey: "network:tags",
          customMetadataInputId: "network-custom-metadata-input",
          customMetadataFocusKey: "network:custom-metadata",
          target: {{ metadata: {{}} }},
          collapsible: true,
        }});
        if (!metadataMarkup.includes('id="network-tags-input-disclosure"')) {{
          throw new Error("Expected collapsible metadata editors to expose a stable disclosure id.");
        }}
        if (!metadataMarkup.includes("open")) {{
          throw new Error("Expected the metadata disclosure helper to respect persisted open state.");
        }}

        function createClassList() {{
          return {{
            add() {{}},
            remove() {{}},
            toggle() {{}},
          }};
        }}

        function createFakeElement(id = null, tagName = "div") {{
          let innerHtml = "";
          const queryResults = new Map();
          return {{
            id,
            tagName: String(tagName || "div").toUpperCase(),
            value: "",
            textContent: "",
            dataset: {{}},
            checked: false,
            disabled: false,
            classList: createClassList(),
            style: {{}},
            listeners: {{}},
            ownerDocument: null,
            addEventListener(eventName, listener) {{
              if (!this.listeners[eventName]) {{
                this.listeners[eventName] = [];
              }}
              this.listeners[eventName].push(listener);
            }},
            dispatchEvent(eventName, event = {{}}) {{
              (this.listeners[eventName] || []).forEach((listener) => {{
                listener({{
                  preventDefault() {{}},
                  stopPropagation() {{}},
                  target: this,
                  ...event,
                }});
              }});
            }},
            click() {{
              this.dispatchEvent("click");
            }},
            focus() {{}},
            setAttribute() {{}},
            removeAttribute() {{}},
            appendChild() {{}},
            get innerHTML() {{
              return innerHtml;
            }},
            set innerHTML(value) {{
              innerHtml = value;
              queryResults.clear();
              const suggestionButtons = [];
              const suggestionPattern =
                /<button[\\s\\S]*?data-tag-suggestion="([^"]+)"[\\s\\S]*?>/g;
              let suggestionMatch = suggestionPattern.exec(value);
              while (suggestionMatch) {{
                const button = createFakeElement(null, "button");
                button.dataset.tagSuggestion = suggestionMatch[1];
                button.ownerDocument = this.ownerDocument;
                suggestionButtons.push(button);
                suggestionMatch = suggestionPattern.exec(value);
              }}
              queryResults.set("[data-tag-suggestion]", suggestionButtons);
            }},
            querySelectorAll(selector) {{
              return queryResults.get(selector) || [];
            }},
          }};
        }}

        function createFakeDocument() {{
          const elements = new Map();
          const listeners = {{}};
          return {{
            registerHtml(html) {{
              elements.clear();
              const tagPattern = /<(input|button|label|div|textarea)[^>]*id="([^"]+)"[^>]*>/g;
              let tagMatch = tagPattern.exec(html);
              while (tagMatch) {{
                const element = createFakeElement(tagMatch[2], tagMatch[1]);
                element.ownerDocument = this;
                elements.set(tagMatch[2], element);
                tagMatch = tagPattern.exec(html);
              }}
            }},
            getElementById(id) {{
              return elements.get(id) || null;
            }},
            addEventListener(eventName, listener) {{
              listeners[eventName] = listener;
            }},
            dispatchEvent(eventName, event = {{}}) {{
              if (listeners[eventName]) {{
                listeners[eventName](event);
              }}
            }},
            createElement(tagName) {{
              return createFakeElement(null, tagName);
            }},
          }};
        }}

        function createRoot(document) {{
          let html = "";
          return {{
            getBoundingClientRect() {{
              return {{ left: 100, top: 200, width: 800, height: 600 }};
            }},
            get innerHTML() {{
              return html;
            }},
            set innerHTML(value) {{
              html = value;
              document.registerHtml(value);
            }},
          }};
        }}

        const document = createFakeDocument();
        const contextMenuRoot = createRoot(document);
        const contextMenuEvents = [];
        const tensor = {{
          id: "tensor_a",
          name: "Tensor A",
          size: {{ width: 140, height: 84 }},
          indices: [
            {{ id: "index_left", name: "left", dimension: 2, metadata: {{ color: "#123456" }} }},
            {{ id: "index_right", name: "right", dimension: 3, metadata: {{}} }},
          ],
          metadata: {{ color: "#345678" }},
        }};
        const edge = {{
          id: "edge_ab",
          name: "bond_ab",
          metadata: {{ color: "#778899" }},
        }};
        const group = {{
          id: "group_a",
          name: "Group A",
          tensor_ids: ["tensor_a"],
          metadata: {{}},
        }};
        const ctx = {{
          state: {{
            spec: {{
              tensors: [tensor],
              edges: [edge],
              groups: [group],
            }},
            canvasContextMenu: null,
          }},
          document,
          dom: {{
            canvasContextMenuRoot: contextMenuRoot,
          }},
          window: {{
            addEventListener() {{}},
          }},
          escapeHtml: (value) => String(value),
          asFiniteNumber: (value, fallback = 1) => {{
            const candidate = Number(value);
            return Number.isFinite(candidate) ? candidate : fallback;
          }},
          buildMetadataEditorMarkup: (options) =>
            metadataSupport.buildMetadataEditorMarkup(options),
          bindMetadataEditors: (options) =>
            metadataSupport.bindMetadataEditors({{
              ...options,
              bindDebouncedAutosave: (element, fieldKey, commit, bindOptions = {{}}) => {{
                if (!element) {{
                  return;
                }}
                element.dataset.focusKey = fieldKey;
                element.addEventListener("blur", () => {{
                  commit();
                }});
                if (bindOptions.commitOnEnter !== false) {{
                  element.addEventListener("keydown", (event) => {{
                    if (event.key !== "Enter" || event.shiftKey) {{
                      return;
                    }}
                    event.preventDefault();
                    commit();
                  }});
                }}
              }},
              applyDesignChange: (mutate) => mutate(),
              setStatus: (message, level = "info") => {{
                contextMenuEvents.push(`status:${{level}}:${{message}}`);
              }},
            }}),
          getMetadataColor: (metadata, fallbackColor) =>
            metadata && metadata.color ? metadata.color : fallbackColor,
          render: () => contextMenuEvents.push("render"),
          setSelection: (selectionIds, options = {{}}) =>
            contextMenuEvents.push({{ selectionIds, primaryId: options.primaryId || null }}),
          propertyCommands: {{
            renameTensor: (payload) => {{
              contextMenuEvents.push(`renameTensor:${{payload.proposedName}}`);
              return true;
            }},
            addTensorIndex: (payload) => {{
              contextMenuEvents.push(`addTensorIndex:${{payload.tensor.id}}`);
            }},
            renameIndex: (payload) => {{
              contextMenuEvents.push(`renameIndex:${{payload.proposedName}}`);
              return true;
            }},
            updateIndexDimension: (payload) => {{
              contextMenuEvents.push(`updateIndexDimension:${{payload.rawValue}}`);
              return true;
            }},
            moveTensorIndex: (payload) => {{
              contextMenuEvents.push(`moveTensorIndex:${{payload.direction}}`);
            }},
            updateTargetColor: (payload) => {{
              contextMenuEvents.push(`updateTargetColor:${{payload.target.id}}:${{payload.nextColor}}`);
            }},
            deleteTensor: (payload) => {{
              contextMenuEvents.push(`deleteTensor:${{payload.tensorId}}`);
            }},
            deleteTensorIndex: (payload) => {{
              contextMenuEvents.push(`deleteTensorIndex:${{payload.indexId}}`);
            }},
            renameEdge: (payload) => {{
              contextMenuEvents.push(`renameEdge:${{payload.proposedName}}`);
              return true;
            }},
            deleteEdge: (payload) => {{
              contextMenuEvents.push(`deleteEdge:${{payload.edgeId}}`);
            }},
            renameGroup: (payload) => {{
              contextMenuEvents.push(`renameGroup:${{payload.proposedName}}`);
              return true;
            }},
          }},
          propertyInvalidation: () => ({{ graph: true }}),
          findTensorById: (tensorId) => (tensorId === tensor.id ? tensor : null),
          findEdgeById: (edgeId) => (edgeId === edge.id ? edge : null),
          findIndexOwner: (indexId) => {{
            const index = tensor.indices.find((candidate) => candidate.id === indexId) || null;
            return index ? {{ tensor, index }} : null;
          }},
          findGroupById: (groupId) => (groupId === group.id ? group : null),
          toggleGroupCollapse: (groupId) => {{
            contextMenuEvents.push(`toggleGroupCollapse:${{groupId}}`);
          }},
        }};

        canvasContextMenuModule.registerCanvasContextMenu(ctx);

        ctx.openCanvasContextMenu({{ kind: "tensor", id: "tensor_a", clientX: 110, clientY: 220 }});
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-add-index-button"')) {{
          throw new Error("Expected the tensor context menu to expose the add-index action.");
        }}
        if (!contextMenuRoot.innerHTML.includes('style="left: 10px; top: 20px;"')) {{
          throw new Error(`Expected the context menu to anchor to the cursor inside the canvas overlay, received HTML:\\n${{contextMenuRoot.innerHTML}}`);
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-tensor-color-input"')) {{
          throw new Error("Expected the tensor context menu to expose the color picker.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-delete-tensor-button"')) {{
          throw new Error("Expected the tensor context menu to expose deletion.");
        }}
        if (!contextMenuRoot.innerHTML.includes(">Indices</span>")) {{
          throw new Error("Expected the tensor context menu to expose the index count chip.");
        }}
        if (!contextMenuRoot.innerHTML.includes(">Total elements</span>")) {{
          throw new Error("Expected the tensor context menu to expose the total elements chip.");
        }}
        if (contextMenuRoot.innerHTML.includes("canvas-context-menu-title")) {{
          throw new Error("The context menu should no longer render a title bar.");
        }}
        if (contextMenuRoot.innerHTML.includes(">Name<")) {{
          throw new Error("The context menu should no longer render the explicit Name label.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-tensor-tags-input"')) {{
          throw new Error("Expected the tensor context menu to expose inline metadata tags.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-tensor-custom-metadata-input"')) {{
          throw new Error("Expected the tensor context menu to expose inline custom metadata.");
        }}
        document.getElementById("context-menu-add-index-button").click();
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-delete-tensor-button"')) {{
          throw new Error("Adding an index should keep the tensor context menu open.");
        }}
        const tensorNameInput = document.getElementById("context-menu-name-input");
        tensorNameInput.value = "Tensor renamed";
        tensorNameInput.dispatchEvent("keydown", {{
          key: "Enter",
          preventDefault() {{
            contextMenuEvents.push("tensorEnterPrevented");
          }},
        }});
        if (ctx.state.canvasContextMenu !== null || contextMenuRoot.innerHTML !== "") {{
          throw new Error("Pressing Enter in the tensor name field should close the context menu.");
        }}

        ctx.openCanvasContextMenu({{ kind: "tensor", id: "tensor_a", clientX: 110, clientY: 220 }});
        const tensorTagsInput = document.getElementById("context-menu-tensor-tags-input");
        tensorTagsInput.value = "alpha";
        tensorTagsInput.dispatchEvent("keydown", {{
          key: "Enter",
          preventDefault() {{
            contextMenuEvents.push("tensorTagsEnterPrevented");
          }},
        }});
        if (ctx.state.canvasContextMenu !== null || contextMenuRoot.innerHTML !== "") {{
          throw new Error("Pressing Enter in the tensor tags field should close the context menu.");
        }}
        ctx.openCanvasContextMenu({{ kind: "tensor", id: "tensor_a", clientX: 110, clientY: 220 }});
        const tensorTagsInputForSuggestions = document.getElementById(
          "context-menu-tensor-tags-input"
        );
        tensorTagsInputForSuggestions.value = "ob";
        tensorTagsInputForSuggestions.selectionStart = 2;
        tensorTagsInputForSuggestions.selectionEnd = 2;
        tensorTagsInputForSuggestions.dispatchEvent("input");
        const tensorSuggestions = document.getElementById(
          "context-menu-tensor-tags-input-suggestions"
        );
        const tensorSuggestionButtons =
          tensorSuggestions.querySelectorAll("[data-tag-suggestion]");
        if (!tensorSuggestionButtons.length) {{
          throw new Error("Expected tag suggestions to appear while typing in the context menu.");
        }}
        tensorSuggestionButtons[0].click();
        if (ctx.state.canvasContextMenu === null || contextMenuRoot.innerHTML === "") {{
          throw new Error("Clicking a tag suggestion should keep the context menu open.");
        }}
        if (tensorTagsInputForSuggestions.value !== "observable, ") {{
          throw new Error(`Expected tag suggestions to keep the field ready for another tag, received ${{tensorTagsInputForSuggestions.value}}.`);
        }}
        if (
          !tensorSuggestions.querySelectorAll("[data-tag-suggestion]").length
        ) {{
          throw new Error("Clicking a tag suggestion should keep the suggestions list available.");
        }}
        document.getElementById("context-menu-delete-tensor-button").click();

        ctx.openCanvasContextMenu({{ kind: "index", id: "index_left", clientX: 10, clientY: 20 }});
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-dimension-input"')) {{
          throw new Error("Expected the index context menu to expose the dimension editor.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-index-color-input"')) {{
          throw new Error("Expected the index context menu to expose the color picker.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-delete-index-button"')) {{
          throw new Error("Expected the index context menu to expose deletion.");
        }}
        if (contextMenuRoot.innerHTML.includes(">Dimension<")) {{
          throw new Error("The index context menu should no longer render the explicit Dimension label.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-index-tags-input"')) {{
          throw new Error("Expected the index context menu to expose inline metadata tags.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-index-custom-metadata-input"')) {{
          throw new Error("Expected the index context menu to expose inline custom metadata.");
        }}
        document.getElementById("context-menu-move-up-button").click();
        ctx.openCanvasContextMenu({{ kind: "index", id: "index_left", clientX: 10, clientY: 20 }});
        document.getElementById("context-menu-delete-index-button").click();

        ctx.openCanvasContextMenu({{ kind: "edge", id: "edge_ab", clientX: 130, clientY: 240 }});
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-edge-color-input"')) {{
          throw new Error("Expected the bond context menu to expose the color picker.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-delete-edge-button"')) {{
          throw new Error("Expected the bond context menu to expose deletion.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-edge-tags-input"')) {{
          throw new Error("Expected the bond context menu to expose inline metadata tags.");
        }}
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-edge-custom-metadata-input"')) {{
          throw new Error("Expected the bond context menu to expose inline custom metadata.");
        }}
        document.getElementById("context-menu-delete-edge-button").click();

        ctx.openCanvasContextMenu({{ kind: "group", id: "group_a", clientX: 10, clientY: 20 }});
        if (!contextMenuRoot.innerHTML.includes('id="context-menu-toggle-group-button"')) {{
          throw new Error("Expected the group context menu to expose the collapse toggle.");
        }}
        document.getElementById("context-menu-toggle-group-button").click();

        if (
          !contextMenuEvents.includes("addTensorIndex:tensor_a") ||
          !contextMenuEvents.includes("moveTensorIndex:-1") ||
          !contextMenuEvents.includes("deleteTensor:tensor_a") ||
          !contextMenuEvents.includes("deleteTensorIndex:index_left") ||
          !contextMenuEvents.includes("deleteEdge:edge_ab") ||
          !contextMenuEvents.includes("toggleGroupCollapse:group_a")
        ) {{
          throw new Error(`Expected the context menu to reuse injected actions, received ${{JSON.stringify(contextMenuEvents)}}.`);
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The metadata autocomplete/context-menu runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_runtime_history_and_spec_kernel_modules_preserve_explicit_contracts(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "history_spec_kernel_modules.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const historySnapshotsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state" / "historySnapshots.js")!r}).href;
        const selectionEntriesUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "state" / "selectionEntries.js")!r}).href;
        const mutationPipelineUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "actions" / "designMutationPipeline.js")!r}).href;
        const specNormalizationUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "spec" / "specNormalization.js")!r}).href;
        const specLookupsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "spec" / "specLookups.js")!r}).href;
        const specMutationsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "spec" / "specMutations.js")!r}).href;

        const [
          historySnapshotsModule,
          selectionEntriesModule,
          mutationPipelineModule,
          specNormalizationModule,
          specLookupsModule,
          specMutationsModule,
        ] = await Promise.all([
          import(historySnapshotsUrl),
          import(selectionEntriesUrl),
          import(mutationPipelineUrl),
          import(specNormalizationUrl),
          import(specLookupsUrl),
          import(specMutationsUrl),
        ]);

        const requiredFactories = [
          historySnapshotsModule.createHistorySnapshotSupport,
          selectionEntriesModule.createSelectionEntrySupport,
          mutationPipelineModule.createDesignMutationPipeline,
          specNormalizationModule.createSpecNormalizationBindings,
          specLookupsModule.createSpecLookupBindings,
          specMutationsModule.createSpecMutationBindings,
        ];
        if (requiredFactories.some((candidate) => typeof candidate !== "function")) {{
          throw new Error("One or more runtime kernel helper factories were not exported.");
        }}

        const constants = {{
          TENSOR_WIDTH: 140,
          TENSOR_HEIGHT: 84,
          MIN_TENSOR_WIDTH: 96,
          MIN_TENSOR_HEIGHT: 60,
          NOTE_WIDTH: 220,
          NOTE_HEIGHT: 120,
          NOTE_MIN_WIDTH: 120,
          NOTE_MIN_HEIGHT: 90,
        }};
        const runtime = {{
          deepClone: (value) => structuredClone(value),
          isObject: (value) =>
            Boolean(value) && typeof value === "object" && !Array.isArray(value),
          asFiniteNumber: (value, fallbackValue) =>
            Number.isFinite(Number(value)) ? Number(value) : fallbackValue,
          makeId: (prefix) => `${{prefix}}_generated`,
          nextName: (prefix, existingNames = []) => {{
            let counter = 1;
            while (existingNames.includes(`${{prefix}}${{counter}}`)) {{
              counter += 1;
            }}
            return `${{prefix}}${{counter}}`;
          }},
          ensureTensorIndexOffsets(tensor) {{
            tensor.indices.forEach((index, indexPosition) => {{
              if (!index.offset) {{
                index.offset = {{ x: indexPosition * 8, y: 0 }};
              }}
            }});
          }},
          defaultIndexOffsetForOrder(indexPosition) {{
            return {{ x: indexPosition * 16, y: 0 }};
          }},
          getMetadataColor: (metadata, fallbackValue) =>
            metadata && typeof metadata.color === "string" ? metadata.color : fallbackValue,
          getIndexColor: () => "#224466",
          isLinearPeriodicMode: (spec) => Boolean(spec && spec.linear_periodic_chain),
          normalizeLinearPeriodicChainInPlace: (chain) => chain,
          hydrateActiveLinearPeriodicCell() {{}},
          syncCurrentGraphIntoLinearPeriodicChain(spec) {{
            spec.synced = true;
          }},
          isLinearPeriodicBoundaryTensor: (tensor) =>
            Boolean(tensor && tensor.linear_periodic_role),
        }};

        const normalizedState = {{
          spec: {{
            id: "network_demo",
            name: "Network Demo",
            tensors: [
              {{
                id: "tensor_a",
                name: "A",
                position: {{ x: 10, y: 20 }},
                size: {{ width: 140, height: 84 }},
                indices: [
                  {{
                    id: "index_a",
                    name: "a",
                    dimension: 2,
                    offset: {{ x: 0, y: 0 }},
                    metadata: {{}},
                  }},
                ],
                metadata: {{}},
              }},
            ],
            groups: [
              {{
                id: "group_a",
                name: "Group A",
                tensor_ids: ["tensor_a"],
                metadata: {{}},
              }},
            ],
            edges: [],
            notes: [
              {{
                id: "note_a",
                text: "Note",
                position: {{ x: 60, y: 80 }},
                size: {{ width: 220, height: 120 }},
                metadata: {{}},
              }},
            ],
            contraction_plan: null,
            metadata: {{}},
          }},
          schemaVersion: "1.0",
          specRevision: 4,
          lookupRevision: -1,
          tensorById: {{}},
          edgeById: {{}},
          edgeByIndexId: {{}},
          groupById: {{}},
          indexOwnerById: {{}},
          groupsByTensorId: {{}},
          noteById: {{}},
        }};

        const specNormalization = specNormalizationModule.createSpecNormalizationBindings({{
          state: normalizedState,
          constants,
          runtime,
        }});
        const specLookups = specLookupsModule.createSpecLookupBindings({{
          ctx: {{
            findVisibleEdgeById: () => null,
            getVisibleEdges: () => [],
            getVisibleTensors: () => [],
          }},
          state: normalizedState,
          runtime,
        }});
        const specMutations = specMutationsModule.createSpecMutationBindings({{
          ctx: {{
            getSelectedEntries() {{
              return [
                {{
                  kind: "tensor",
                  id: "tensor_a",
                  tensor: normalizedState.spec.tensors[0],
                }},
              ];
            }},
          }},
          state: normalizedState,
          constants,
          runtime,
          findTensorById: (tensorId) => specLookups.findTensorById(tensorId),
          findGroupById: (groupId) => specLookups.findGroupById(groupId),
          findEdgeById: (edgeId) => specLookups.findEdgeById(edgeId),
          findEdgeByIndexId: (indexId) => specLookups.findEdgeByIndexId(indexId),
          findIndexOwner: (indexId) => specLookups.findIndexOwner(indexId),
          resolveBaseEdgeId: (edgeId) => specLookups.resolveBaseEdgeId(edgeId),
        }});

        specLookups.ensureSpecLookups();
        if (!specLookups.findTensorById("tensor_a")) {{
          throw new Error("Spec lookup helpers should find tensors by id.");
        }}
        if (!specLookups.findIndexOwner("index_a")) {{
          throw new Error("Spec lookup helpers should resolve index owners.");
        }}
        specMutations.applyColorToSelection("#ff8800");
        if (normalizedState.spec.tensors[0].metadata.color !== "#ff8800") {{
          throw new Error("Spec mutation helpers should update selection colors.");
        }}
        const createdTensor = specMutations.createTensor(120, 160);
        if (createdTensor.indices.length !== 2) {{
          throw new Error(`Expected created tensor to receive two default indices, received ${{createdTensor.indices.length}}.`);
        }}
        const normalizedGraphSection = specNormalization.normalizeGraphSectionInPlace({{
          tensors: [{{ position: {{}}, size: {{}}, indices: [{{ metadata: {{}} }}], metadata: {{}} }}],
          groups: [{{ tensor_ids: ["tensor_a"] }}],
          edges: [{{ left: {{}}, right: {{}}, metadata: {{}} }}],
          notes: [{{ position: {{}}, size: {{}}, metadata: {{}} }}],
          contraction_plan: null,
          metadata: {{}},
        }});
        if (!normalizedGraphSection.tensors[0].id || !normalizedGraphSection.edges[0].id) {{
          throw new Error("Spec normalization helpers should seed missing entity ids.");
        }}

        const historyEvents = [];
        const historyState = {{
          spec: normalizedState.spec,
          tensorOrder: ["tensor_a"],
          undoStack: [],
          redoStack: [{{ spec: {{ id: "redo" }}, tensorOrder: [] }}],
          generatedCode: "print('demo')",
          lastMutationClearedCode: false,
          pendingIndexId: "missing_index",
          pendingPlannerOperandId: "missing_operand",
          pendingPlannerSelectionId: "selection_a",
          plannerInspectionStepCount: 2,
          plannerPreviewMode: "automaticFuture",
          plannerFutureBadgeDisclosure: {{ future: true }},
          activeNoteResize: {{ noteId: "note_a" }},
          activeSidebarTab: "planner",
          pendingPropertiesIndexFocusId: "index_a",
          autoExpandedTensorIndex: {{ tensorId: "tensor_a", indexId: "index_a", wasOpen: false }},
          tensorIndexDisclosureState: {{
            tensor_a: {{ index_a: true }},
          }},
          selectionIds: ["tensor_a", "missing_selection"],
          primarySelectionId: "missing_selection",
          selectedElement: null,
          lookupRevision: 4,
          specRevision: 4,
        }};
        const historySupport = historySnapshotsModule.createHistorySnapshotSupport({{
          state: historyState,
          historyLimit: 1,
          buildHistorySnapshotSpec: () => ({{ id: "snapshot_spec" }}),
          deepClone: (value) => structuredClone(value),
          updateToolbarState: () => historyEvents.push("toolbar"),
          normalizeSpec: (spec) => ({{ ...spec, normalized: true }}),
          bumpSpecRevision: () => historyEvents.push("bump"),
          reconcileTensorOrder: () => historyEvents.push("reconcile"),
          enforceLinearPeriodicEngineSupport: () => historyEvents.push("linear-periodic"),
          clearGeneratedCodePreview: () => {{
            historyEvents.push("clear-code");
            historyState.generatedCode = "";
            return true;
          }},
          pruneSelectionToExisting: () => historyEvents.push("prune"),
          render: () => historyEvents.push("render"),
          refreshContractionAnalysis: () => historyEvents.push("analysis"),
        }});
        const historySnapshot = historySupport.createHistorySnapshot();
        if (historySnapshot.spec.id !== "snapshot_spec") {{
          throw new Error("History snapshot support should use the injected spec builder.");
        }}
        historySupport.commitHistorySnapshot({{ spec: {{ id: "older" }}, tensorOrder: [] }});
        historySupport.restoreHistorySnapshot({{
          spec: {{ id: "restored_spec" }},
          tensorOrder: ["tensor_a"],
        }});
        if (historyState.undoStack.length !== 1 || historyState.redoStack.length !== 0) {{
          throw new Error("History snapshot support should prune undo/redo stacks explicitly.");
        }}
        if (!historyEvents.includes("render") || !historyEvents.includes("analysis")) {{
          throw new Error(`History restore should trigger render and analysis refresh, received ${{JSON.stringify(historyEvents)}}.`);
        }}

        const selectionState = {{
          selectionIds: ["group_a", "missing_selection"],
          primarySelectionId: "missing_selection",
          pendingIndexId: "missing_index",
          pendingPlannerOperandId: "missing_operand",
          pendingPlannerSelectionId: "selection_a",
          selectedElement: null,
          tensorIndexDisclosureState: {{
            tensor_a: {{ index_a: true }},
          }},
          autoExpandedTensorIndex: {{
            tensorId: "tensor_a",
            indexId: "index_a",
            wasOpen: false,
          }},
          pendingPropertiesIndexFocusId: null,
        }};
        const selectionSupport = selectionEntriesModule.createSelectionEntrySupport({{
          state: selectionState,
          findGroupById: (groupId) =>
            groupId === "group_a" ? {{ id: "group_a", name: "Group A" }} : null,
          findTensorById: (tensorId) =>
            tensorId === "tensor_a"
              ? {{ id: "tensor_a", indices: [{{ id: "index_a" }}] }}
              : null,
          findVisibleTensorById: () => null,
          findIndexOwner: (indexId) =>
            indexId === "index_a"
              ? {{
                  tensor: {{ id: "tensor_a" }},
                  index: {{ id: "index_a" }},
                  indexPosition: 0,
                }}
              : null,
          findEdgeById: (edgeId) =>
            edgeId === "edge_a" ? {{ id: "edge_a" }} : null,
          findNoteById: (noteId) =>
            noteId === "note_a" ? {{ id: "note_a" }} : null,
          isContractionSceneVisible: () => false,
          isInspectingPastStage: () => false,
          isPlannerOperandAvailable: (operandId) => operandId === "operand_ok",
          renderSelectionUi: () => {{}},
        }});
        const selectedEntries = selectionSupport.getSelectedEntries();
        if (selectedEntries.length !== 1 || selectedEntries[0].kind !== "group") {{
          throw new Error(`Selection support should resolve existing selection entries, received ${{JSON.stringify(selectedEntries)}}.`);
        }}
        selectionSupport.pruneSelectionToExisting();
        if (selectionState.selectionIds.length !== 1 || selectionState.selectionIds[0] !== "group_a") {{
          throw new Error(`Selection pruning should remove missing ids, received ${{JSON.stringify(selectionState.selectionIds)}}.`);
        }}

        const pipelineEvents = [];
        const mutationState = {{
          spec: {{ id: "network_demo" }},
          selectionIds: ["tensor_a"],
          primarySelectionId: "tensor_a",
          plannerPreviewMode: "automaticFuture",
          plannerFutureBadgeDisclosure: {{ badge: true }},
          lastMutationClearedCode: false,
        }};
        const mutationPipeline = mutationPipelineModule.createDesignMutationPipeline({{
          state: mutationState,
          isLinearPeriodicMode: () => false,
          captureEditableFocus: () => "focus-token",
          restoreEditableFocus: (focusToken) => pipelineEvents.push(`restore:${{focusToken}}`),
          resetDerivedStateCaches: () => pipelineEvents.push("reset-caches"),
          syncCurrentGraphIntoLinearPeriodicChain: () => pipelineEvents.push("sync-chain"),
          repairContractionPlan: () => pipelineEvents.push("repair-plan"),
          reconcileTensorOrder: () => pipelineEvents.push("reconcile-order"),
          bumpSpecRevision: () => pipelineEvents.push("bump"),
          createHistorySnapshot: () => ({{ id: "before" }}),
          commitHistorySnapshot: (snapshot) => {{
            if (snapshot.id !== "before") {{
              throw new Error("Mutation pipeline should commit the snapshot captured before the mutation.");
            }}
            pipelineEvents.push("commit");
            mutationState.lastMutationClearedCode = true;
            return true;
          }},
          buildDesignStatusMessage: (message, previewCleared) =>
            previewCleared ? `${{message}} preview-cleared` : message,
          pruneSelectionToExisting: () => pipelineEvents.push("prune-selection"),
          updatePendingPropertiesIndexFocus: () => pipelineEvents.push("sync-index-focus"),
          syncSelectedElementState: () => pipelineEvents.push("sync-selected"),
          renderMutationState: (invalidate) =>
            pipelineEvents.push(`render:${{invalidate.graph}}:${{invalidate.analysis}}`),
          refreshContractionAnalysis: () => pipelineEvents.push("refresh-analysis"),
          setStatus: (message, level) => pipelineEvents.push(`status:${{level}}:${{message}}`),
        }});
        mutationPipeline.applyDesignChange(
          () => {{
            mutationState.spec.updated = true;
          }},
          {{
            statusMessage: "Updated design.",
            invalidate: {{
              graph: true,
              analysis: true,
            }},
          }}
        );
        if (!mutationState.spec.updated) {{
          throw new Error("Mutation pipeline should execute the provided mutator.");
        }}
        if (
          !pipelineEvents.includes("commit") ||
          !pipelineEvents.includes("refresh-analysis") ||
          !pipelineEvents.includes("status:success:Updated design. preview-cleared")
        ) {{
          throw new Error(`Mutation pipeline should keep the explicit mutation flow, received ${{JSON.stringify(pipelineEvents)}}.`);
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The history/spec kernel runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_shell_modules_expose_explicit_bootstrap_flow_and_toolbar_bindings(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "shell_modules.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const bootstrapFlowUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "shell" / "editorBootstrapFlow.js")!r}).href;
        const shellBindingsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "shell" / "editorShellBindings.js")!r}).href;
        const shortcutTooltipUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "shell" / "shortcutTooltip.js")!r}).href;

        const [bootstrapFlowModule, shellBindingsModule, tooltipModule] = await Promise.all([
          import(bootstrapFlowUrl),
          import(shellBindingsUrl),
          import(shortcutTooltipUrl),
        ]);

        function createButton(id) {{
          return {{
            id,
            disabled: false,
            dataset: {{}},
            attributes: {{}},
            listeners: {{}},
            textContent: "",
            setAttribute(name, value) {{
              this.attributes[name] = value;
            }},
            removeAttribute(name) {{
              delete this.attributes[name];
            }},
            addEventListener(type, handler) {{
              this.listeners[type] = handler;
            }},
            click() {{
              if (this.listeners.click) {{
                this.listeners.click({{ target: this }});
              }}
            }},
          }};
        }}

        const state = {{
          availableCollectionFormats: [],
          templateCatalogWarnings: [],
          selectedEngine: "",
        }};
        const storeCalls = [];
        const store = {{
          setSpec(spec) {{
            state.spec = spec;
            storeCalls.push({{ step: "setSpec", spec }});
          }},
          setSchemaVersion(schemaVersion) {{
            state.schemaVersion = schemaVersion;
            storeCalls.push({{ step: "setSchemaVersion", schemaVersion }});
          }},
          setAvailableCollectionFormats(collectionFormats) {{
            state.availableCollectionFormats = [...collectionFormats];
            storeCalls.push({{ step: "setAvailableCollectionFormats", collectionFormats }});
          }},
          setAnnotationDefinitions(annotationDefinitions) {{
            state.annotationDefinitions = annotationDefinitions;
            storeCalls.push({{ step: "setAnnotationDefinitions", annotationDefinitions }});
          }},
          setSelectedEngine(engine) {{
            state.selectedEngine = engine;
            storeCalls.push({{ step: "setSelectedEngine", engine }});
          }},
          setSelectedCollectionFormat(collectionFormat) {{
            state.selectedCollectionFormat = collectionFormat;
            storeCalls.push({{ step: "setSelectedCollectionFormat", collectionFormat }});
          }},
        }};
        const flowEvents = [];
        const bootstrapFlow = bootstrapFlowModule.createEditorBootstrapFlow({{
          state,
          store,
          sessionService: {{
            async loadBootstrap() {{
              flowEvents.push("loadBootstrap");
              return {{
                spec: {{ network: {{ id: "network_demo" }} }},
                schema_version: 4,
                collection_formats: ["list", "dict"],
                templates: ["mps"],
                template_definitions: {{ mps: {{ display_name: "MPS" }} }},
                template_catalog_warnings: ["Template warning"],
                annotation_definitions: {{ tensor: [] }},
                default_engine: "quimb",
                default_collection_format: "dict",
                engines: ["quimb", "cotengra"],
              }};
            }},
          }},
          actions: {{
            normalizeSpec: (spec) => ({{ ...spec, normalized: true }}),
            applyTemplateCatalogPayload: (payload) => {{
              state.templateCatalogWarnings = payload.templateCatalogWarnings;
              flowEvents.push({{ templatePayload: payload }});
            }},
            reconcileTensorOrder: () => flowEvents.push("reconcileTensorOrder"),
            populateEngineOptions: (engines) => flowEvents.push({{ engines }}),
            enforceLinearPeriodicEngineSupport: () =>
              flowEvents.push("enforceLinearPeriodicEngineSupport"),
            populateCollectionFormatOptions: (formats) =>
              flowEvents.push({{ formats }}),
            initGraph: () => flowEvents.push("initGraph"),
            clearHistory: () => flowEvents.push("clearHistory"),
            render: () => flowEvents.push("render"),
            refreshContractionAnalysis: () =>
              flowEvents.push("refreshContractionAnalysis"),
            setStatus: (message, level) => flowEvents.push({{ message, level }}),
          }},
        }});
        await bootstrapFlow.bootstrap();
        if (!storeCalls.some((entry) => entry.step === "setSpec" && entry.spec.normalized === true)) {{
          throw new Error(`Expected bootstrap flow to normalize the incoming spec, received ${{JSON.stringify(storeCalls)}}.`);
        }}
        if (!flowEvents.some((entry) => entry.message === "Template warning" && entry.level === "error")) {{
          throw new Error(`Expected bootstrap flow to surface template warnings, received ${{JSON.stringify(flowEvents)}}.`);
        }}
        if (!flowEvents.includes("enforceLinearPeriodicEngineSupport") || !flowEvents.includes("refreshContractionAnalysis")) {{
          throw new Error(`Expected bootstrap flow to run post-load actions, received ${{JSON.stringify(flowEvents)}}.`);
        }}

        const buttonRegistry = new Map();
        const getButton = (id) => {{
          if (!buttonRegistry.has(id)) {{
            buttonRegistry.set(id, createButton(id));
          }}
          return buttonRegistry.get(id);
        }};
        const documentListeners = [];
        const windowListeners = [];
        const tooltipDocument = {{
          body: {{
            appendChild(node) {{
              flowEvents.push({{ tooltipAttached: node.className }});
            }},
          }},
          getElementById: (id) => getButton(id),
          createElement: () => ({{
            className: "",
            textContent: "",
            style: {{}},
            classList: {{
              add() {{}},
              remove() {{}},
            }},
            setAttribute() {{}},
            getBoundingClientRect() {{
              return {{ width: 0, height: 0 }};
            }},
          }}),
          addEventListener(type, handler) {{
            documentListeners.push(type);
          }},
        }};
        const shortcutTooltip = tooltipModule.createShortcutTooltip({{
          documentRef: tooltipDocument,
          windowRef: {{
            innerWidth: 800,
            innerHeight: 600,
            addEventListener(type, handler) {{
              windowListeners.push(type);
            }},
          }},
        }});
        shortcutTooltip.applyShortcutHint("generate-button", "Generate code", "Shift+G");
        if (getButton("generate-button").dataset.shortcut !== "Shift+G") {{
          throw new Error("Expected shortcut tooltip helper to set the shortcut dataset.");
        }}

        const dom = {{
          addNoteButton: getButton("add-note-button"),
          connectButton: getButton("connect-button"),
          loadInput: {{ click() {{ flowEvents.push("loadInput.click"); }}, addEventListener(type, handler) {{ this[type] = handler; }} }},
          subnetworkLoadInput: {{ addEventListener(type, handler) {{ this[type] = handler; }} }},
          undoButton: getButton("undo-button"),
          redoButton: getButton("redo-button"),
          exportButton: getButton("export-button"),
          exportFormatSelect: {{ value: "py", addEventListener(type, handler) {{ this[type] = handler; }} }},
          toggleLinearPeriodicButton: getButton("toggle-linear-periodic-button"),
          linearPeriodicPreviousCellButton: getButton("linear-periodic-previous-cell-button"),
          linearPeriodicCellLabel: {{ textContent: "" }},
          linearPeriodicNextCellButton: getButton("linear-periodic-next-cell-button"),
          templateSelect: {{ value: "mps", addEventListener(type, handler) {{ this[type] = handler; }} }},
          templateGraphSizeInput: {{ addEventListener(type, handler) {{ this[type] = handler; }} }},
          templateBondDimensionInput: {{ addEventListener(type, handler) {{ this[type] = handler; }} }},
          templatePhysicalDimensionInput: {{ addEventListener(type, handler) {{ this[type] = handler; }} }},
          insertTemplateButton: getButton("insert-template-button"),
          insertSubnetworkButton: getButton("insert-subnetwork-button"),
          renameTemplateButton: getButton("rename-template-button"),
          deleteTemplateButton: getButton("delete-template-button"),
          reflowImportedButton: getButton("reflow-imported-button"),
          createGroupButton: getButton("create-group-button"),
          helpButton: getButton("help-button"),
          helpModal: {{ classList: {{ add() {{}}, remove() {{}} }} }},
          helpBackdrop: getButton("help-backdrop"),
          helpCloseButton: getButton("help-close-button"),
          canvasShell: {{ addEventListener(type, handler) {{ this[type] = handler; }}, getBoundingClientRect() {{ return {{ left: 0, top: 0, width: 1000, height: 800 }}; }} }},
          minimapCanvas: {{ addEventListener(type, handler) {{ this[type] = handler; }} }},
          engineSelect: {{ value: "cotengra", addEventListener(type, handler) {{ this[type] = handler; }} }},
          collectionFormatSelect: {{ value: "dict", addEventListener(type, handler) {{ this[type] = handler; }} }},
        }};
        const shellActions = {{
          handleNewDesign: () => flowEvents.push("handleNewDesign"),
          addTensorAtCenter: () => flowEvents.push("addTensorAtCenter"),
          addNoteAtCenter: () => flowEvents.push("addNoteAtCenter"),
          toggleConnectMode: () => flowEvents.push("toggleConnectMode"),
          deleteSelection: () => flowEvents.push("deleteSelection"),
          saveDesign: () => flowEvents.push("saveDesign"),
          generateCode: () => flowEvents.push("generateCode"),
          completeEditor: () => flowEvents.push("completeEditor"),
          cancelEditor: () => flowEvents.push("cancelEditor"),
          copyGeneratedCode: () => flowEvents.push("copyGeneratedCode"),
          performUndo: () => flowEvents.push("performUndo"),
          performRedo: () => flowEvents.push("performRedo"),
          downloadSelectedExport: () => flowEvents.push("downloadSelectedExport"),
          updateToolbarState: () => flowEvents.push("updateToolbarState"),
          toggleLinearPeriodicMode: () => flowEvents.push("toggleLinearPeriodicMode"),
          switchLinearPeriodicCell: (direction) =>
            flowEvents.push(`switchLinearPeriodicCell:${{direction}}`),
          handleTemplateSelectionChange: () => flowEvents.push("handleTemplateSelectionChange"),
          handleTemplateParameterInput: () => flowEvents.push("handleTemplateParameterInput"),
          insertTemplate: () => flowEvents.push("insertTemplate"),
          openSubnetworkPicker: () => flowEvents.push("openSubnetworkPicker"),
          renameSelectedTemplate: () => flowEvents.push("renameSelectedTemplate"),
          deleteSelectedTemplate: () => flowEvents.push("deleteSelectedTemplate"),
          reflowLastImportedTensors: () => flowEvents.push("reflowLastImportedTensors"),
          createGroupFromSelection: () => flowEvents.push("createGroupFromSelection"),
          toggleHelpModal: (isOpen) => flowEvents.push(`toggleHelpModal:${{isOpen}}`),
          enforceLinearPeriodicEngineSupport: () =>
            flowEvents.push("binding.enforceLinearPeriodicEngineSupport"),
          renderPlanner: () => flowEvents.push("binding.renderPlanner"),
          formatEngineLabel: (engine) => engine.toUpperCase(),
          setStatus: (message, level) => flowEvents.push({{ bindingStatus: message, level }}),
          loadDesignFromFile: () => flowEvents.push("loadDesignFromFile"),
          loadSubnetworkFromFile: () => flowEvents.push("loadSubnetworkFromFile"),
          handleKeydown: () => flowEvents.push("handleKeydown"),
          sendCancelBeacon: () => flowEvents.push("sendCancelBeacon"),
          handleWindowResize: () => flowEvents.push("handleWindowResize"),
          handleGlobalMouseMove: () => flowEvents.push("handleGlobalMouseMove"),
          handleGlobalMouseUp: () => flowEvents.push("handleGlobalMouseUp"),
          handleCanvasContextMenu: () => flowEvents.push("handleCanvasContextMenu"),
          handleCanvasWheel: () => flowEvents.push("handleCanvasWheel"),
          handleCanvasMouseDown: () => flowEvents.push("handleCanvasMouseDown"),
          handleMinimapMouseDown: () => flowEvents.push("handleMinimapMouseDown"),
        }};
        const shellBindings = shellBindingsModule.createEditorShellBindings({{
          state,
          store,
          dom,
          documentRef: tooltipDocument,
          windowRef: {{
            addEventListener(type, handler) {{
              windowListeners.push(type);
            }},
          }},
          actions: shellActions,
          shortcutTooltip,
        }});
        shellBindings.attachToolbarHandlers();
        getButton("generate-button").click();
        dom.engineSelect.change({{ target: {{ value: "cotengra" }} }});
        dom.insertSubnetworkButton.click();
        if (!flowEvents.includes("generateCode")) {{
          throw new Error(`Expected toolbar generate binding to invoke the injected action, received ${{JSON.stringify(flowEvents)}}.`);
        }}
        if (!flowEvents.includes("openSubnetworkPicker")) {{
          throw new Error(`Expected subnetwork button binding to invoke the injected action, received ${{JSON.stringify(flowEvents)}}.`);
        }}
        if (!flowEvents.includes("binding.enforceLinearPeriodicEngineSupport") || !flowEvents.includes("binding.renderPlanner")) {{
          throw new Error(`Expected engine change binding to run its injected actions, received ${{JSON.stringify(flowEvents)}}.`);
        }}
        if (!flowEvents.some((entry) => entry.bindingStatus === "Engine set to COTENGRA.")) {{
          throw new Error(`Expected engine change binding to set status through injected actions, received ${{JSON.stringify(flowEvents)}}.`);
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The shell module runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required")
def test_editor_shell_helper_modules_expose_explicit_ui_and_invalidation_adapters(
    tmp_path: Path,
) -> None:
    script_path = _write_runtime_script(
        tmp_path,
        "editor_shell_helpers.mjs",
        f"""
        import {{ pathToFileURL }} from "node:url";

        const sessionUiUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "session" / "sessionUiAdapters.js")!r}).href;
        const plannerBindingsUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "planner" / "plannerPanelBindings.js")!r}).href;
        const propertyAutosaveUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "properties" / "propertyAutosave.js")!r}).href;
        const propertyInvalidationUrl = pathToFileURL({str(REPO_ROOT / "src" / "tensor_network_editor" / "app" / "static" / "js" / "properties" / "propertyInvalidation.js")!r}).href;

        const [
          sessionUiModule,
          plannerBindingsModule,
          propertyAutosaveModule,
          propertyInvalidationModule,
        ] = await Promise.all([
          import(sessionUiUrl),
          import(plannerBindingsUrl),
          import(propertyAutosaveUrl),
          import(propertyInvalidationUrl),
        ]);

        const uiEvents = [];
        const sessionUi = sessionUiModule.createSessionUiAdapters({{
          promptText: (message, defaultValue) => {{
            uiEvents.push({{ kind: "prompt", message, defaultValue }});
            return "project_fragment";
          }},
          confirmAction: (message) => {{
            uiEvents.push({{ kind: "confirm", message }});
            return true;
          }},
          copyText: async (text) => {{
            uiEvents.push({{ kind: "copy", text }});
          }},
          downloadText: (filename, text, contentType) => {{
            uiEvents.push({{ kind: "downloadText", filename, text, contentType }});
          }},
          downloadBlob: (filename, blobLike) => {{
            uiEvents.push({{ kind: "downloadBlob", filename, type: blobLike.type }});
          }},
          closeWindow: () => {{
            uiEvents.push({{ kind: "close" }});
          }},
        }});
        if (sessionUi.promptText("Name?", "seed") !== "project_fragment") {{
          throw new Error("Session UI prompt adapter did not return the injected value.");
        }}
        if (!sessionUi.confirmAction("Overwrite?")) {{
          throw new Error("Session UI confirm adapter should forward the injected result.");
        }}
        await sessionUi.copyText("result = 1");
        sessionUi.downloadText("demo.json", "{{}}", "application/json");
        sessionUi.downloadBlob("demo.py", {{ type: "text/x-python" }});
        sessionUi.closeWindow();
        if (!uiEvents.some((event) => event.kind === "copy" && event.text === "result = 1")) {{
          throw new Error(`Expected injected copy adapter to run, received ${{JSON.stringify(uiEvents)}}.`);
        }}
        if (!uiEvents.some((event) => event.kind === "downloadText" && event.filename === "demo.json")) {{
          throw new Error(`Expected injected text download adapter to run, received ${{JSON.stringify(uiEvents)}}.`);
        }}

        function createButton(dataset = {{}}) {{
          return {{
            dataset,
            listeners: {{}},
            addEventListener(type, handler) {{
              this.listeners[type] = handler;
            }},
          }};
        }}

        const plannerEvents = [];
        const toggleButton = createButton();
        const resetButton = createButton();
        const trimButton = createButton({{ trimStep: "2" }});
        const inspectButton = createButton({{ inspectStep: "1" }});
        const disclosureButton = createButton({{ disclosure: "automaticFuture" }});
        const previewButton = createButton({{ previewMode: "automaticFuture" }});
        const acceptButton = createButton({{ acceptMode: "automaticFuture" }});
        const plannerBindings = plannerBindingsModule.createPlannerPanelBindings({{
          plannerPanel: {{
            querySelectorAll(selector) {{
              if (selector === "[data-trim-step]") {{
                return [trimButton];
              }}
              if (selector === "[data-inspect-step]") {{
                return [inspectButton];
              }}
              if (selector === "[data-disclosure]") {{
                return [disclosureButton];
              }}
              if (selector === "[data-preview-mode]") {{
                return [previewButton];
              }}
              if (selector === "[data-accept-mode]") {{
                return [acceptButton];
              }}
              return [];
            }},
          }},
          plannerDocument: {{
            getElementById(elementId) {{
              if (elementId === "toggle-planner-mode-button") {{
                return toggleButton;
              }}
              if (elementId === "planner-reset-button") {{
                return resetButton;
              }}
              return null;
            }},
          }},
          actions: {{
            togglePlannerMode: () => plannerEvents.push("togglePlannerMode"),
            trimContractionPlan: (stepCount) =>
              plannerEvents.push(`trim:${{stepCount}}`),
            togglePastInspection: (stepIndex) =>
              plannerEvents.push(`inspect:${{stepIndex}}`),
            clearAutomaticPreview: (options) =>
              plannerEvents.push({{ clearAutomaticPreview: options }}),
            togglePlannerDisclosure: (disclosureKey) =>
              plannerEvents.push(`disclosure:${{disclosureKey}}`),
            startAutomaticPreview: (mode) =>
              plannerEvents.push(`preview:${{mode}}`),
            acceptAutomaticPlan: (mode) =>
              plannerEvents.push(`accept:${{mode}}`),
            renderPlanner: () => plannerEvents.push("renderPlanner"),
            renderEditor: () => plannerEvents.push("renderEditor"),
            renderOverlayDecorations: () =>
              plannerEvents.push("renderOverlayDecorations"),
          }},
        }});
        plannerBindings.bindPlannerPanelInteractions();
        toggleButton.listeners.click();
        resetButton.listeners.click();
        trimButton.listeners.click();
        inspectButton.listeners.click();
        disclosureButton.listeners.click();
        previewButton.listeners.click();
        acceptButton.listeners.click();
        if (!plannerEvents.includes("togglePlannerMode")) {{
          throw new Error(`Expected planner mode toggle binding, received ${{JSON.stringify(plannerEvents)}}.`);
        }}
        if (!plannerEvents.includes("trim:0") || !plannerEvents.includes("trim:2")) {{
          throw new Error(`Expected trim bindings for reset and step trim, received ${{JSON.stringify(plannerEvents)}}.`);
        }}
        if (!plannerEvents.includes("inspect:1") || !plannerEvents.includes("renderEditor")) {{
          throw new Error(`Expected injected inspect actions to run, received ${{JSON.stringify(plannerEvents)}}.`);
        }}

        function createInput() {{
          return {{
            dataset: {{}},
            listeners: {{}},
            addEventListener(type, handler) {{
              this.listeners[type] = handler;
            }},
          }};
        }}

        const timers = [];
        const clearedTimers = [];
        const autosave = propertyAutosaveModule.createPropertyAutosaveBindings({{
          windowRef: {{
            setTimeout(callback, delay) {{
              const timerId = timers.length + 1;
              timers.push({{ timerId, delay, callback }});
              return timerId;
            }},
            clearTimeout(timerId) {{
              clearedTimers.push(timerId);
            }},
          }},
          delayMs: 45,
        }});
        const input = createInput();
        let debouncedCommits = 0;
        autosave.bindDebouncedAutosave(input, "field:name", () => {{
          debouncedCommits += 1;
        }});
        input.listeners.input();
        if (timers.length !== 1 || timers[0].delay !== 45) {{
          throw new Error(`Expected one autosave timer with delay 45, received ${{JSON.stringify(timers)}}.`);
        }}
        timers[0].callback();
        input.listeners.blur();
        if (debouncedCommits !== 2) {{
          throw new Error(`Expected blur and timer to commit, received ${{debouncedCommits}} commits.`);
        }}
        if (clearedTimers.length !== 0) {{
          throw new Error(`Expected no extra timer clears after the scheduled commit completed, received ${{JSON.stringify(clearedTimers)}}.`);
        }}

        const invalidationSupport =
          propertyInvalidationModule.createPropertyInvalidationSupport({{
            isLinearPeriodicMode: () => true,
          }});
        const baseInvalidation = invalidationSupport.propertyInvalidation({{
          graph: true,
        }});
        if (!baseInvalidation.graph || !baseInvalidation.properties) {{
          throw new Error(`Expected explicit invalidation defaults, received ${{JSON.stringify(baseInvalidation)}}.`);
        }}
        const selectionInvalidation =
          invalidationSupport.selectionColorInvalidation([
            {{ kind: "tensor" }},
            {{ kind: "group" }},
          ]);
        if (!selectionInvalidation.graph || !selectionInvalidation.overlays) {{
          throw new Error(`Expected selection invalidation to track graph and overlays, received ${{JSON.stringify(selectionInvalidation)}}.`);
        }}
        """,
    )

    completed_process = _run_runtime_script(script_path)

    assert completed_process.returncode == 0, (
        "The editor-shell helper runtime script failed.\n"
        f"STDOUT:\n{completed_process.stdout}\n"
        f"STDERR:\n{completed_process.stderr}"
    )
