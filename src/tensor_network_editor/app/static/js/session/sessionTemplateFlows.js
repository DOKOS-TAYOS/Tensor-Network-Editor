import {
  buildExportTemplatePayload,
  createSessionTemplateImportSupport,
} from "./sessionTemplateImports.js";
import { createSessionTemplateDialogs } from "./sessionTemplateDialogs.js";
import { createSessionTemplateManager } from "./sessionTemplateManager.js";

export function createSessionTemplateFlows({
  dom,
  state,
  store,
  selectors,
  services,
  commands,
  sessionUi,
  actions,
}) {
  const {
    subnetworkLoadInput,
    templateLoadInput,
    templateManagerList,
    subnetworkLibraryList,
    subnetworkLibrarySearchInput,
    subnetworkLibraryTagFilter,
    subnetworkLibraryWarning,
  } = dom;
  const sessionService = services.session;
  const subnetworkService = services.subnetwork;
  const documentRef =
    (templateManagerList && templateManagerList.ownerDocument)
    || (subnetworkLibraryList && subnetworkLibraryList.ownerDocument)
    || globalThis.document;

  function getGroupById(groupId) {
    return actions.findGroupById(groupId);
  }

  function getSelectedTensorIds() {
    return actions.getSelectedTensorIds();
  }

  function getSubnetworkTargetCenter() {
    return actions.suggestTensorPosition(actions.viewportCenterPosition());
  }

  function getCurrentTemplateName() {
    return typeof dom.templateSelect.value === "string"
      ? dom.templateSelect.value.trim()
      : "";
  }

  function isForModeActive() {
    return (
      (typeof actions.isForMode === "function" && actions.isForMode()) ||
      (typeof actions.isLinearPeriodicMode === "function" &&
        actions.isLinearPeriodicMode())
    );
  }

  function isBenchmarkSchemeView() {
    return Boolean(
      state.benchmarkSession &&
        state.benchmarkSession.enabled &&
        Number.isInteger(state.benchmarkSession.activePosition) &&
        state.benchmarkSession.activePosition > 0
    );
  }

  function openSubnetworkPicker() {
    if (isForModeActive()) {
      actions.setStatus(
        "Subnetwork insertion is only available in normal graph mode.",
        "error"
      );
      return;
    }
    sessionUi.openFilePicker(subnetworkLoadInput);
  }

  const dialogs = createSessionTemplateDialogs({
    actions,
    sessionUi,
  });
  const {
    promptForTemplateDisplayName,
    promptForSubnetworkName,
    promptForSubnetworkTags,
  } = dialogs;
  const importSupport = createSessionTemplateImportSupport({
    templateLoadInput,
    subnetworkLoadInput,
    state,
    sessionService,
    subnetworkService,
    commands,
    sessionUi,
    actions,
    getSubnetworkTargetCenter,
    isForModeActive,
  });
  const {
    loadSessionTemplatesFromFile,
    loadSubnetworkFromFile,
  } = importSupport;
  const templateManager = createSessionTemplateManager({
    templateManagerList,
    documentRef,
    state,
    actions,
  });
  const {
    discardTemplateManagerChanges,
    saveTemplateManagerChanges,
    toggleTemplateManager,
  } = templateManager;

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function normalizeSubnetworkCatalogName(rawName, fallbackName = "subnetwork") {
    const normalizedFallback = String(fallbackName || "subnetwork")
      .toLowerCase()
      .replaceAll(/[^a-z0-9_]+/g, "_")
      .replaceAll(/^_+|_+$/g, "")
      .replaceAll(/_+/g, "_");
    let normalizedName = String(rawName || "")
      .normalize("NFKD")
      .replaceAll(/[^\w\s-]+/g, "")
      .toLowerCase()
      .replaceAll(/[\s-]+/g, "_")
      .replaceAll(/_+/g, "_")
      .replaceAll(/^_+|_+$/g, "");
    if (!normalizedName) {
      normalizedName = normalizedFallback || "subnetwork";
    }
    if (!/^[a-z]/.test(normalizedName)) {
      normalizedName = `subnetwork_${normalizedName}`.replaceAll(/_+/g, "_");
    }
    return normalizedName;
  }

  function getSubnetworkEntries() {
    return (Array.isArray(state.availableSubnetworks) ? state.availableSubnetworks : [])
      .map((subnetworkName) => {
        const definition =
          state.subnetworkDefinitions &&
          typeof state.subnetworkDefinitions === "object"
            ? state.subnetworkDefinitions[subnetworkName]
            : null;
        if (!definition || typeof definition !== "object") {
          return null;
        }
        return {
          subnetworkName,
          displayName:
            typeof definition.display_name === "string" && definition.display_name
              ? definition.display_name
              : subnetworkName.replaceAll("_", " "),
          source:
            typeof definition.source === "string" && definition.source
              ? definition.source
              : "project",
          tags: Array.isArray(definition.tags)
            ? definition.tags.filter((tag) => typeof tag === "string" && tag)
            : [],
          tensorCount: Number.isFinite(definition.tensor_count)
            ? definition.tensor_count
            : 0,
          edgeCount: Number.isFinite(definition.edge_count)
            ? definition.edge_count
            : 0,
          serializedSpec:
            definition.spec && typeof definition.spec === "object"
              ? definition.spec
              : null,
        };
      })
      .filter(Boolean);
  }

  function getSubnetworkEntryByName(subnetworkName) {
    return (
      getSubnetworkEntries().find(
        (entry) => entry.subnetworkName === subnetworkName
      ) || null
    );
  }

  function collectSubnetworkLibraryTags() {
    return [...new Set(
      getSubnetworkEntries().flatMap((entry) => entry.tags)
    )].sort((leftTag, rightTag) => leftTag.localeCompare(rightTag));
  }

  function getFilteredSubnetworkEntries() {
    const searchQuery =
      typeof state.subnetworkLibrarySearchQuery === "string"
        ? state.subnetworkLibrarySearchQuery.trim().toLowerCase()
        : "";
    const tagFilter =
      typeof state.subnetworkLibraryTagFilter === "string"
        ? state.subnetworkLibraryTagFilter.trim()
        : "";
    return getSubnetworkEntries().filter((entry) => {
      if (tagFilter && !entry.tags.includes(tagFilter)) {
        return false;
      }
      if (!searchQuery) {
        return true;
      }
      return [
        entry.subnetworkName,
        entry.displayName,
        entry.source,
        ...entry.tags,
      ].some(
        (value) =>
          typeof value === "string" && value.toLowerCase().includes(searchQuery)
      );
    });
  }

  function buildSubnetworkPreviewMarkup(entry) {
    const tensors =
      entry &&
      entry.serializedSpec &&
      entry.serializedSpec.network &&
      Array.isArray(entry.serializedSpec.network.tensors)
        ? entry.serializedSpec.network.tensors
        : [];
    const edges =
      entry &&
      entry.serializedSpec &&
      entry.serializedSpec.network &&
      Array.isArray(entry.serializedSpec.network.edges)
        ? entry.serializedSpec.network.edges
        : [];
    if (!tensors.length) {
      return '<div class="subnetwork-library-preview-empty">Empty</div>';
    }
    const previewWidth = 136;
    const previewHeight = 88;
    const padding = 8;
    const bounds = {
      left: Math.min(
        ...tensors.map((tensor) => (tensor.position?.x || 0) - (tensor.size?.width || 120) / 2)
      ),
      right: Math.max(
        ...tensors.map((tensor) => (tensor.position?.x || 0) + (tensor.size?.width || 120) / 2)
      ),
      top: Math.min(
        ...tensors.map((tensor) => (tensor.position?.y || 0) - (tensor.size?.height || 72) / 2)
      ),
      bottom: Math.max(
        ...tensors.map((tensor) => (tensor.position?.y || 0) + (tensor.size?.height || 72) / 2)
      ),
    };
    const spanX = Math.max(1, bounds.right - bounds.left);
    const spanY = Math.max(1, bounds.bottom - bounds.top);
    const scale = Math.min(
      (previewWidth - padding * 2) / spanX,
      (previewHeight - padding * 2) / spanY
    );
    const scalePositionX = (value) => (value - bounds.left) * scale + padding;
    const scalePositionY = (value) => (value - bounds.top) * scale + padding;
    const centerByTensorId = Object.fromEntries(
      tensors.map((tensor) => [
        tensor.id,
        {
          x: scalePositionX(tensor.position?.x || 0),
          y: scalePositionY(tensor.position?.y || 0),
        },
      ])
    );
    const edgeMarkup = edges
      .map((edge) => {
        const leftCenter = centerByTensorId[edge?.left?.tensor_id];
        const rightCenter = centerByTensorId[edge?.right?.tensor_id];
        if (!leftCenter || !rightCenter) {
          return "";
        }
        return `<line x1="${leftCenter.x}" y1="${leftCenter.y}" x2="${rightCenter.x}" y2="${rightCenter.y}" />`;
      })
      .join("");
    const tensorMarkup = tensors
      .map((tensor) => {
        const width = Math.max(12, (tensor.size?.width || 120) * scale);
        const height = Math.max(10, (tensor.size?.height || 72) * scale);
        const center = centerByTensorId[tensor.id];
        return `
          <rect
            x="${center.x - width / 2}"
            y="${center.y - height / 2}"
            width="${width}"
            height="${height}"
            rx="${Math.min(7, height / 3)}"
          />
        `;
      })
      .join("");
    return `
      <svg viewBox="0 0 ${previewWidth} ${previewHeight}" aria-hidden="true" focusable="false">
        <g class="subnetwork-library-preview-edges">${edgeMarkup}</g>
        <g class="subnetwork-library-preview-tensors">${tensorMarkup}</g>
      </svg>
    `;
  }

  function setSubnetworkLibraryWarning(message = "") {
    if (!subnetworkLibraryWarning) {
      return;
    }
    subnetworkLibraryWarning.textContent = message;
    subnetworkLibraryWarning.hidden = !message;
  }

  function populateSubnetworkTagFilter() {
    if (!subnetworkLibraryTagFilter) {
      return;
    }
    const availableTags = collectSubnetworkLibraryTags();
    const currentValue =
      typeof state.subnetworkLibraryTagFilter === "string"
        ? state.subnetworkLibraryTagFilter
        : "";
    subnetworkLibraryTagFilter.innerHTML = "";
    const allOption = documentRef.createElement("option");
    allOption.value = "";
    allOption.textContent = "All tags";
    subnetworkLibraryTagFilter.appendChild(allOption);
    availableTags.forEach((tag) => {
      const option = documentRef.createElement("option");
      option.value = tag;
      option.textContent = tag;
      subnetworkLibraryTagFilter.appendChild(option);
    });
    subnetworkLibraryTagFilter.value =
      currentValue && availableTags.includes(currentValue) ? currentValue : "";
    if (subnetworkLibraryTagFilter.value !== currentValue) {
      state.subnetworkLibraryTagFilter = subnetworkLibraryTagFilter.value;
    }
  }

  function renderSubnetworkLibrary() {
    if (!subnetworkLibraryList) {
      return;
    }
    if (subnetworkLibrarySearchInput) {
      subnetworkLibrarySearchInput.value =
        typeof state.subnetworkLibrarySearchQuery === "string"
          ? state.subnetworkLibrarySearchQuery
          : "";
    }
    populateSubnetworkTagFilter();
    const warningMessages = Array.isArray(state.subnetworkCatalogWarnings)
      ? state.subnetworkCatalogWarnings.filter(
          (warningMessage) => typeof warningMessage === "string" && warningMessage
        )
      : [];
    setSubnetworkLibraryWarning(warningMessages[0] || "");
    subnetworkLibraryList.innerHTML = "";
    const filteredEntries = getFilteredSubnetworkEntries();
    if (!filteredEntries.length) {
      const emptyState = documentRef.createElement("p");
      emptyState.className = "subnetwork-library-empty-state";
      emptyState.textContent = state.availableSubnetworks.length
        ? "No subnetworks match the current filters."
        : "No reusable subnetworks have been saved yet.";
      subnetworkLibraryList.appendChild(emptyState);
      return;
    }
    filteredEntries.forEach((entry) => {
      const row = documentRef.createElement("article");
      row.className = "subnetwork-library-row";
      if (entry.subnetworkName === state.selectedSubnetworkName) {
        row.classList.add("is-selected");
      }
      row.addEventListener("click", () => {
        state.selectedSubnetworkName = entry.subnetworkName;
        renderSubnetworkLibrary();
      });

      const preview = documentRef.createElement("div");
      preview.className = "subnetwork-library-preview";
      preview.innerHTML = buildSubnetworkPreviewMarkup(entry);
      row.appendChild(preview);

      const content = documentRef.createElement("div");
      content.className = "subnetwork-library-content";
      const titleRow = documentRef.createElement("div");
      titleRow.className = "subnetwork-library-title-row";
      const title = documentRef.createElement("strong");
      title.textContent = entry.displayName;
      const sourceBadge = documentRef.createElement("span");
      sourceBadge.className = `subnetwork-library-source source-${entry.source}`;
      sourceBadge.textContent = entry.source === "shared" ? "Shared" : "Project";
      titleRow.append(title, sourceBadge);
      content.appendChild(titleRow);

      const meta = documentRef.createElement("p");
      meta.className = "subnetwork-library-meta";
      meta.textContent = `${entry.tensorCount} tensor(s) | ${entry.edgeCount} connection(s)`;
      content.appendChild(meta);

      const idLine = documentRef.createElement("p");
      idLine.className = "subnetwork-library-id";
      idLine.innerHTML = `<code>${escapeHtml(entry.subnetworkName)}</code>`;
      content.appendChild(idLine);

      const tagsRow = documentRef.createElement("div");
      tagsRow.className = "subnetwork-library-tags";
      if (entry.tags.length) {
        entry.tags.forEach((tag) => {
          const tagChip = documentRef.createElement("span");
          tagChip.className = "subnetwork-library-tag";
          tagChip.textContent = tag;
          tagsRow.appendChild(tagChip);
        });
      } else {
        const emptyTag = documentRef.createElement("span");
        emptyTag.className = "subnetwork-library-tag is-empty";
        emptyTag.textContent = "No tags";
        tagsRow.appendChild(emptyTag);
      }
      content.appendChild(tagsRow);
      row.appendChild(content);

      const actionRow = documentRef.createElement("div");
      actionRow.className = "subnetwork-library-actions";
      const insertButton = documentRef.createElement("button");
      insertButton.type = "button";
      insertButton.className = "button-accent-insert";
      insertButton.textContent = "Insert";
      insertButton.addEventListener("click", async (event) => {
        event.stopPropagation();
        await insertSubnetworkFromLibrary(entry.subnetworkName);
      });
      actionRow.appendChild(insertButton);

      if (entry.source === "project") {
        const renameButton = documentRef.createElement("button");
        renameButton.type = "button";
        renameButton.textContent = "Rename";
        renameButton.addEventListener("click", async (event) => {
          event.stopPropagation();
          await renameLibrarySubnetwork(entry.subnetworkName);
        });
        actionRow.appendChild(renameButton);

        const deleteButton = documentRef.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "danger";
        deleteButton.textContent = "Delete";
        deleteButton.addEventListener("click", async (event) => {
          event.stopPropagation();
          await deleteLibrarySubnetwork(entry.subnetworkName);
        });
        actionRow.appendChild(deleteButton);
      }

      row.appendChild(actionRow);
      subnetworkLibraryList.appendChild(row);
    });
  }

  async function extractTemplateSpecByTensorIds(tensorIds, emptySelectionMessage) {
    if (isForModeActive()) {
      actions.setStatus(
        "Templates are only available in normal graph mode.",
        "error"
      );
      return null;
    }
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      actions.setStatus(emptySelectionMessage);
      return null;
    }
    try {
      const payload = await subnetworkService.extractSubnetwork({
        serializedSpec: actions.serializeCurrentSpec({
          persistViewSnapshots: false,
        }),
        tensorIds,
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return null;
      }
      return payload.spec;
    } catch (error) {
      actions.setStatus(`Could not prepare the template: ${error.message}`, "error");
      return null;
    }
  }

  async function exportSubnetworkByTensorIds(tensorIds, label = "subnetwork") {
    if (isForModeActive()) {
      actions.setStatus(
        "Subnetwork export is only available in normal graph mode.",
        "error"
      );
      return;
    }
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      actions.setStatus("Select one or more tensors to extract a subnetwork.");
      return;
    }
    try {
      const payload = await subnetworkService.extractSubnetwork({
        serializedSpec: actions.serializeCurrentSpec({
          persistViewSnapshots: false,
        }),
        tensorIds,
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      const resolvedDisplayName = promptForSubnetworkName(
        (payload.spec &&
          payload.spec.network &&
          typeof payload.spec.network.name === "string" &&
          payload.spec.network.name.trim()) ||
          label ||
          "subnetwork",
        "Subnetwork export cancelled."
      );
      if (!resolvedDisplayName) {
        return;
      }
      if (payload.spec && payload.spec.network) {
        payload.spec.network.name = resolvedDisplayName;
      }
      sessionUi.downloadText(
        `${actions.sanitizeFilename(resolvedDisplayName || "subnetwork")}.json`,
        JSON.stringify(payload.spec, null, 2),
        "application/json;charset=utf-8"
      );
      actions.setStatus(`Saved ${resolvedDisplayName} as JSON.`, "success");
    } catch (error) {
      actions.setStatus(`Could not export the subnetwork: ${error.message}`, "error");
    }
  }

  async function exportSelectedSubnetwork() {
    await exportSubnetworkByTensorIds(getSelectedTensorIds(), "subnetwork");
  }

  async function exportGroupSubnetwork(groupId) {
    const group = getGroupById(groupId);
    if (!group || !Array.isArray(group.tensor_ids) || !group.tensor_ids.length) {
      actions.setStatus("This group does not contain any tensors to extract.", "error");
      return;
    }
    await exportSubnetworkByTensorIds(group.tensor_ids, group.name || "group");
  }

  async function saveTemplateByTensorIds(tensorIds, baseDisplayName) {
    const serializedSpec = await extractTemplateSpecByTensorIds(
      tensorIds,
      "Select one or more tensors to save as a template."
    );
    if (!serializedSpec) {
      return;
    }
    const resolvedDisplayName = promptForTemplateDisplayName(
      actions.getNextSessionTemplateDisplayName(baseDisplayName),
      "Template save cancelled."
    );
    if (!resolvedDisplayName) {
      return;
    }
    if (actions.hasTemplateDisplayName(resolvedDisplayName)) {
      actions.setStatus(
        `Template name '${resolvedDisplayName}' is already in use.`,
        "error"
      );
      return;
    }
    const addResult = actions.addSessionTemplate({
      displayName: resolvedDisplayName,
      spec: serializedSpec,
    });
    if (!addResult.ok) {
      actions.setStatus("Could not save the selected template.", "error");
      return;
    }
    actions.setStatus(`Saved ${resolvedDisplayName} for this session.`, "success");
  }

  async function saveSelectionAsSessionTemplate() {
    await saveTemplateByTensorIds(getSelectedTensorIds(), "Selection Template");
  }

  async function promoteSelectedSubnetworkToTemplate() {
    await saveSelectionAsSessionTemplate();
  }

  async function promoteGroupToTemplate(groupId) {
    const group = getGroupById(groupId);
    if (!group || !Array.isArray(group.tensor_ids) || !group.tensor_ids.length) {
      actions.setStatus("This group does not contain any tensors to promote.", "error");
      return;
    }
    const baseDisplayName =
      typeof group.name === "string" && group.name.trim()
        ? group.name.trim()
        : "Group Template";
    await saveTemplateByTensorIds(group.tensor_ids, baseDisplayName);
  }

  function openSessionTemplatePicker() {
    sessionUi.openFilePicker(templateLoadInput);
  }

  async function exportSelectedTemplateSpec() {
    const serializedSpec = await extractTemplateSpecByTensorIds(
      getSelectedTensorIds(),
      "Select one or more tensors to export as a template."
    );
    if (!serializedSpec) {
      return;
    }
    const displayName = promptForTemplateDisplayName(
      (serializedSpec.network &&
        typeof serializedSpec.network.name === "string" &&
        serializedSpec.network.name.trim()) ||
        "Selection Template",
      "Template export cancelled."
    );
    if (!displayName) {
      return;
    }
    const payload = buildExportTemplatePayload(
      displayName,
      serializedSpec,
      actions.sanitizeFilename
    );
    sessionUi.downloadText(
      `${actions.sanitizeFilename(displayName || "template")}.json`,
      JSON.stringify(payload, null, 2),
      "application/json;charset=utf-8"
    );
    actions.setStatus(`Exported ${displayName} as a reusable template.`, "success");
  }

  async function insertTemplate() {
    const templateName = dom.templateSelect.value;
    if (!templateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    const templateSource = actions.getTemplateSource(templateName);
    try {
      let importedNetwork = null;
      if (templateSource === "session") {
        const serializedSpec = actions.getTemplateSpec(templateName);
        if (!serializedSpec || !serializedSpec.network) {
          actions.setStatus("Could not read the selected session template.", "error");
          return;
        }
        importedNetwork = serializedSpec.network;
      } else {
        const parameters = actions.persistTemplateParametersFromControls();
        const payload = await sessionService.buildTemplate({
          templateName,
          parameters,
        });
        importedNetwork = payload.spec.network;
      }
      const importedSpec = actions.uniquifyImportedSpec(
        importedNetwork,
        actions.makeId("template")
      );
      const translatedSpec = actions.translateImportedSpec(
        importedSpec,
        getSubnetworkTargetCenter()
      );
      actions.applyDesignChange(
        () => {
          state.spec.tensors.push(...translatedSpec.tensors);
          state.spec.edges.push(...translatedSpec.edges);
          state.spec.groups.push(...translatedSpec.groups);
          store.setLastImportedTensorIds(
            translatedSpec.tensors.map((tensor) => tensor.id)
          );
        },
        {
          invalidate: { lookups: true },
          selectionIds: translatedSpec.tensors.map((tensor) => tensor.id),
          primaryId: translatedSpec.tensors.length
            ? translatedSpec.tensors[translatedSpec.tensors.length - 1].id
            : null,
          statusMessage: `Inserted ${translatedSpec.name}.`,
        }
      );
    } catch (error) {
      actions.setStatus(`Could not insert the template: ${error.message}`, "error");
    }
  }

  async function renameSelectedTemplate() {
    const currentTemplateName = getCurrentTemplateName();
    if (!currentTemplateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    if (!selectors.isSessionTemplate(currentTemplateName)) {
      actions.setStatus(
        "Built-in and project templates are read-only in this editor.",
        "error"
      );
      return;
    }
    const currentEntry = actions
      .listTemplateEntries()
      .find((entry) => entry.templateName === currentTemplateName);
    const nextDisplayName = sessionUi.promptText(
      "Choose a new name for this session template.",
      currentEntry ? currentEntry.displayName : ""
    );
    if (typeof nextDisplayName !== "string") {
      actions.setStatus("Template rename cancelled.");
      return;
    }
    const trimmedDisplayName = nextDisplayName.trim();
    if (!trimmedDisplayName) {
      actions.setStatus("Template names cannot be empty.", "error");
      return;
    }
    if (actions.hasTemplateDisplayName(trimmedDisplayName, currentTemplateName)) {
      actions.setStatus(
        `Template name '${trimmedDisplayName}' is already in use.`,
        "error"
      );
      return;
    }
    actions.updateSessionTemplateDisplayNames([
      {
        templateName: currentTemplateName,
        displayName: trimmedDisplayName,
      },
    ]);
    actions.setStatus(`Renamed the template to ${trimmedDisplayName}.`, "success");
  }

  async function deleteSelectedTemplate() {
    const currentTemplateName = getCurrentTemplateName();
    if (!currentTemplateName) {
      actions.setStatus("Choose a template first.");
      return;
    }
    if (!selectors.isSessionTemplate(currentTemplateName)) {
      actions.setStatus(
        "Built-in and project templates are read-only in this editor.",
        "error"
      );
      return;
    }
    const currentEntry = actions
      .listTemplateEntries()
      .find((entry) => entry.templateName === currentTemplateName);
    if (
      !sessionUi.confirmAction(
        `Delete '${currentEntry ? currentEntry.displayName : "this template"}' from this session?`
      )
    ) {
      actions.setStatus("Template deletion cancelled.");
      return;
    }
    actions.removeSessionTemplate(currentTemplateName);
    actions.setStatus("Deleted the session template.", "success");
  }

  async function runSubnetworkCatalogMutation(
    requestFactory,
    {
      successMessage,
      duplicateMessage = "That library entry already exists. Overwrite it?",
    } = {}
  ) {
    try {
      let payload = await requestFactory(false);
      if (!payload.ok) {
        const errorMessage = payload.message || actions.formatIssues(payload.issues);
        if (
          /already|registered|exists/i.test(errorMessage) &&
          sessionUi.confirmAction(duplicateMessage)
        ) {
          payload = await requestFactory(true);
        }
      }
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return false;
      }
      commands.applySubnetworkCatalogUpdate(payload, successMessage);
      renderSubnetworkLibrary();
      actions.updateToolbarState();
      return true;
    } catch (error) {
      actions.setStatus(`Could not update the subnetwork library: ${error.message}`, "error");
      return false;
    }
  }

  function promptForLibrarySubnetworkName(defaultDisplayName, cancelledStatus) {
    const promptedDisplayName = promptForSubnetworkName(
      defaultDisplayName,
      cancelledStatus
    );
    if (!promptedDisplayName) {
      return null;
    }
    return {
      displayName: promptedDisplayName,
      subnetworkName: normalizeSubnetworkCatalogName(
        promptedDisplayName,
        "subnetwork"
      ),
    };
  }

  async function saveSubnetworkByTensorIdsToLibrary(tensorIds, baseDisplayName) {
    if (isForModeActive()) {
      actions.setStatus(
        "The subnetwork library is only available in normal graph mode.",
        "error"
      );
      return;
    }
    if (isBenchmarkSchemeView()) {
      actions.setStatus(
        "The subnetwork library is unavailable while viewing a benchmark scheme.",
        "error"
      );
      return;
    }
    if (!Array.isArray(tensorIds) || !tensorIds.length) {
      actions.setStatus("Select one or more tensors first.");
      return;
    }
    const naming = promptForLibrarySubnetworkName(
      baseDisplayName,
      "Subnetwork library save cancelled."
    );
    if (!naming) {
      return;
    }
    const tags = promptForSubnetworkTags([], "Subnetwork library save cancelled.");
    if (tags === null) {
      return;
    }
    await runSubnetworkCatalogMutation(
      (overwrite) =>
        subnetworkService.saveSubnetworkToLibrary({
          serializedSpec: actions.serializeCurrentSpec({
            persistViewSnapshots: false,
          }),
          tensorIds,
          subnetworkName: naming.subnetworkName,
          tags,
          overwrite,
        }),
      {
        successMessage: `Saved ${naming.displayName} to the subnetwork library.`,
        duplicateMessage: `A library entry named '${naming.subnetworkName}' already exists. Overwrite it?`,
      }
    );
  }

  async function saveSelectionToSubnetworkLibrary() {
    await saveSubnetworkByTensorIdsToLibrary(
      getSelectedTensorIds(),
      "Selection Block"
    );
  }

  async function saveGroupToSubnetworkLibrary(groupId) {
    const group = getGroupById(groupId);
    if (!group || !Array.isArray(group.tensor_ids) || !group.tensor_ids.length) {
      actions.setStatus("This group does not contain any tensors to save.", "error");
      return;
    }
    await saveSubnetworkByTensorIdsToLibrary(
      group.tensor_ids,
      group.name || "Group Block"
    );
  }

  async function insertSubnetworkFromLibrary(subnetworkName = state.selectedSubnetworkName) {
    if (isForModeActive()) {
      actions.setStatus(
        "Subnetwork insertion is only available in normal graph mode.",
        "error"
      );
      return;
    }
    if (isBenchmarkSchemeView()) {
      actions.setStatus(
        "Subnetwork insertion is unavailable while viewing a benchmark scheme.",
        "error"
      );
      return;
    }
    const entry = getSubnetworkEntryByName(subnetworkName);
    if (!entry) {
      actions.setStatus("Choose a saved subnetwork first.", "error");
      return;
    }
    try {
      const payload = await subnetworkService.prepareLibrarySubnetworkForInsert({
        subnetworkName: entry.subnetworkName,
        targetCenter: getSubnetworkTargetCenter(),
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      state.selectedSubnetworkName = entry.subnetworkName;
      commands.insertPreparedSubnetwork(
        payload.spec.network,
        entry.displayName || entry.subnetworkName
      );
      renderSubnetworkLibrary();
    } catch (error) {
      actions.setStatus(`Could not insert the subnetwork: ${error.message}`, "error");
    }
  }

  async function renameLibrarySubnetwork(subnetworkName) {
    const entry = getSubnetworkEntryByName(subnetworkName);
    if (!entry) {
      actions.setStatus("That subnetwork could not be found.", "error");
      return;
    }
    if (entry.source !== "project") {
      actions.setStatus("Shared subnetworks are read-only in this editor.", "error");
      return;
    }
    const naming = promptForLibrarySubnetworkName(
      entry.displayName,
      "Subnetwork rename cancelled."
    );
    if (!naming) {
      return;
    }
    if (naming.subnetworkName === entry.subnetworkName) {
      actions.setStatus("The subnetwork name is unchanged.");
      return;
    }
    await runSubnetworkCatalogMutation(
      (overwrite) =>
        subnetworkService.renameLibrarySubnetwork({
          subnetworkName: entry.subnetworkName,
          newSubnetworkName: naming.subnetworkName,
          overwrite,
        }),
      {
        successMessage: `Renamed ${entry.displayName} to ${naming.displayName}.`,
        duplicateMessage: `A library entry named '${naming.subnetworkName}' already exists. Overwrite it?`,
      }
    );
  }

  async function deleteLibrarySubnetwork(subnetworkName) {
    const entry = getSubnetworkEntryByName(subnetworkName);
    if (!entry) {
      actions.setStatus("That subnetwork could not be found.", "error");
      return;
    }
    if (entry.source !== "project") {
      actions.setStatus("Shared subnetworks are read-only in this editor.", "error");
      return;
    }
    if (!sessionUi.confirmAction(`Delete '${entry.displayName}' from the subnetwork library?`)) {
      actions.setStatus("Subnetwork deletion cancelled.");
      return;
    }
    try {
      const payload = await subnetworkService.deleteLibrarySubnetwork({
        subnetworkName: entry.subnetworkName,
      });
      if (!payload.ok) {
        actions.setStatus(
          payload.message || actions.formatIssues(payload.issues),
          "error"
        );
        return;
      }
      commands.applySubnetworkCatalogUpdate(
        payload,
        `Deleted ${entry.displayName} from the subnetwork library.`
      );
      renderSubnetworkLibrary();
      actions.updateToolbarState();
    } catch (error) {
      actions.setStatus(`Could not delete the subnetwork: ${error.message}`, "error");
    }
  }

  function updateSubnetworkLibrarySearch(query) {
    state.subnetworkLibrarySearchQuery = typeof query === "string" ? query : "";
    renderSubnetworkLibrary();
  }

  function updateSubnetworkLibraryTagFilter(tag) {
    state.subnetworkLibraryTagFilter = typeof tag === "string" ? tag : "";
    renderSubnetworkLibrary();
  }

  function openSubnetworkLibrary() {
    if (isForModeActive()) {
      actions.setStatus(
        "The subnetwork library is only available in normal graph mode.",
        "error"
      );
      return false;
    }
    if (isBenchmarkSchemeView()) {
      actions.setStatus(
        "The subnetwork library is unavailable while viewing a benchmark scheme.",
        "error"
      );
      return false;
    }
    state.isSubnetworkLibraryOpen = true;
    actions.syncSubnetworkLibraryModalState();
    renderSubnetworkLibrary();
    actions.updateToolbarState();
    if (
      subnetworkLibrarySearchInput &&
      typeof subnetworkLibrarySearchInput.focus === "function"
    ) {
      subnetworkLibrarySearchInput.focus();
    }
    return true;
  }

  return {
    openSubnetworkPicker,
    exportSelectedSubnetwork,
    exportGroupSubnetwork,
    saveSelectionToSubnetworkLibrary,
    saveGroupToSubnetworkLibrary,
    openSubnetworkLibrary,
    insertSelectedSubnetworkFromLibrary: () =>
      insertSubnetworkFromLibrary(state.selectedSubnetworkName),
    updateSubnetworkLibrarySearch,
    updateSubnetworkLibraryTagFilter,
    saveSelectionAsSessionTemplate,
    openSessionTemplatePicker,
    loadSessionTemplatesFromFile,
    exportSelectedTemplateSpec,
    promoteSelectedSubnetworkToTemplate,
    promoteGroupToTemplate,
    toggleTemplateManager,
    saveTemplateManagerChanges,
    discardTemplateManagerChanges,
    renameSelectedTemplate,
    deleteSelectedTemplate,
    loadSubnetworkFromFile,
    insertTemplate,
  };
}
