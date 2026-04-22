import {
  buildSerializedNetworkPreviewMarkup,
  escapeHtml,
} from "./sessionTemplateFlowSubnetworkLibrary.js";

function renderTrashIcon() {
  return `
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
    </svg>
  `;
}

export function createSessionTemplateManager({
  templateManagerList,
  documentRef,
  state,
  actions,
}) {
  let templateManagerDraft = null;

  function getTemplateCounts(serializedSpec) {
    const tensors =
      serializedSpec &&
      serializedSpec.network &&
      Array.isArray(serializedSpec.network.tensors)
        ? serializedSpec.network.tensors
        : [];
    const edges =
      serializedSpec &&
      serializedSpec.network &&
      Array.isArray(serializedSpec.network.edges)
        ? serializedSpec.network.edges
        : [];
    return {
      tensorCount: tensors.length,
      edgeCount: edges.length,
    };
  }

  function buildTemplateManagerRow(entry) {
    const { tensorCount, edgeCount } = getTemplateCounts(entry.spec);
    const row = documentRef.createElement("article");
    const preview = documentRef.createElement("div");
    const content = documentRef.createElement("div");
    const titleRow = documentRef.createElement("div");
    const title = documentRef.createElement("strong");
    const sourceBadge = documentRef.createElement("span");
    const meta = documentRef.createElement("p");
    const idLine = documentRef.createElement("p");
    const nameField = documentRef.createElement("label");
    const nameCaption = documentRef.createElement("span");
    const nameInput = documentRef.createElement("input");
    const actionRow = documentRef.createElement("div");
    row.className = "subnetwork-library-row template-manager-row";
    preview.className = "subnetwork-library-preview template-manager-preview";
    preview.innerHTML = buildSerializedNetworkPreviewMarkup(entry.spec);
    row.appendChild(preview);

    content.className = "subnetwork-library-content";
    titleRow.className = "subnetwork-library-title-row";
    title.textContent = entry.displayName;
    sourceBadge.className = "subnetwork-library-source source-session";
    sourceBadge.textContent = "Session";
    titleRow.append(title, sourceBadge);
    content.appendChild(titleRow);

    meta.className = "subnetwork-library-meta";
    meta.textContent = `${tensorCount} tensor(s) | ${edgeCount} connection(s)`;
    content.appendChild(meta);

    idLine.className = "subnetwork-library-id";
    idLine.innerHTML = `<code>${escapeHtml(entry.templateName)}</code>`;
    content.appendChild(idLine);

    nameField.className = "field-group template-manager-name-field";
    nameCaption.textContent = "Display name";
    nameInput.value =
      templateManagerDraft.nameByTemplateName.get(entry.templateName) || entry.displayName;
    nameInput.dataset.templateName = entry.templateName;
    nameInput.setAttribute("aria-label", `Template name for ${entry.displayName}`);
    nameInput.disabled = false;
    nameField.append(nameCaption, nameInput);
    content.appendChild(nameField);
    row.append(content);

    actionRow.className = "subnetwork-library-actions template-manager-actions";
    const deleteButton = documentRef.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "danger";
    deleteButton.setAttribute("aria-label", `Delete ${entry.displayName}`);
    deleteButton.title = `Delete ${entry.displayName}`;
    deleteButton.innerHTML = renderTrashIcon();
    deleteButton.addEventListener("click", () => {
      templateManagerDraft.deletedTemplateNames.add(entry.templateName);
      renderTemplateManager();
    });
    actionRow.appendChild(deleteButton);
    row.appendChild(actionRow);
    return row;
  }

  function renderTemplateManager() {
    if (!templateManagerList || !templateManagerDraft) {
      return;
    }
    templateManagerList.innerHTML = "";
    const sessionEntries = actions.listTemplateEntries().filter((entry) => {
      if (entry.source !== "session") {
        return false;
      }
      if (templateManagerDraft.deletedTemplateNames.has(entry.templateName)) {
        return false;
      }
      return true;
    });
    if (!sessionEntries.length) {
      const emptyState = documentRef.createElement("p");
      emptyState.className = "subnetwork-library-empty-state";
      emptyState.textContent = "No session templates have been saved yet.";
      templateManagerList.appendChild(emptyState);
      return;
    }
    sessionEntries.forEach((entry) => {
      templateManagerList.appendChild(buildTemplateManagerRow(entry));
    });
  }

  function openTemplateManager() {
    const sessionEntries = actions
      .listTemplateEntries()
      .filter((entry) => entry.source === "session");
    templateManagerDraft = {
      nameByTemplateName: new Map(
        sessionEntries.map((entry) => [entry.templateName, entry.displayName])
      ),
      deletedTemplateNames: new Set(),
    };
    state.isTemplateManagerOpen = true;
    actions.syncTemplateManagerModalState();
    actions.setTemplateManagerValidationMessage("");
    renderTemplateManager();
  }

  function collectTemplateManagerUpdates() {
    const sessionEntries = actions
      .listTemplateEntries()
      .filter((entry) => entry.source === "session");
    return sessionEntries
      .filter((entry) => !templateManagerDraft.deletedTemplateNames.has(entry.templateName))
      .map((entry) => {
        const input = templateManagerList.querySelector(
          `input[data-template-name="${entry.templateName}"]`
        );
        return {
          templateName: entry.templateName,
          displayName:
            input && typeof input.value === "string" ? input.value.trim() : entry.displayName,
        };
      });
  }

  function validateTemplateManagerUpdates(updates) {
    const lockedDisplayNames = new Set(
      actions
        .listTemplateEntries()
        .filter((entry) => entry.source !== "session")
        .map((entry) => entry.displayName)
    );
    const seenSessionNames = new Set();
    for (const update of updates) {
      if (!update.displayName) {
        return "Template names cannot be empty.";
      }
      if (
        lockedDisplayNames.has(update.displayName) ||
        seenSessionNames.has(update.displayName)
      ) {
        return `Template name '${update.displayName}' is already in use.`;
      }
      seenSessionNames.add(update.displayName);
    }
    return "";
  }

  function saveTemplateManagerChanges() {
    if (!state.isTemplateManagerOpen || !templateManagerDraft) {
      return false;
    }
    const updates = collectTemplateManagerUpdates();
    const validationMessage = validateTemplateManagerUpdates(updates);
    if (validationMessage) {
      actions.setTemplateManagerValidationMessage(validationMessage);
      return true;
    }
    templateManagerDraft.deletedTemplateNames.forEach((templateName) => {
      actions.removeSessionTemplate(templateName);
    });
    actions.updateSessionTemplateDisplayNames(updates);
    state.isTemplateManagerOpen = false;
    templateManagerDraft = null;
    actions.setTemplateManagerValidationMessage("");
    actions.syncTemplateManagerModalState();
    actions.updateToolbarState();
    actions.setStatus("Updated session templates.", "success");
    return false;
  }

  function discardTemplateManagerChanges() {
    if (!state.isTemplateManagerOpen) {
      return false;
    }
    state.isTemplateManagerOpen = false;
    templateManagerDraft = null;
    actions.setTemplateManagerValidationMessage("");
    actions.syncTemplateManagerModalState();
    actions.updateToolbarState();
    actions.setStatus("Discarded template changes.");
    return false;
  }

  function toggleTemplateManager(forceOpen) {
    const shouldOpen =
      typeof forceOpen === "boolean" ? forceOpen : !state.isTemplateManagerOpen;
    if (shouldOpen) {
      openTemplateManager();
      return true;
    }
    return discardTemplateManagerChanges();
  }

  return {
    discardTemplateManagerChanges,
    openTemplateManager,
    renderTemplateManager,
    saveTemplateManagerChanges,
    toggleTemplateManager,
  };
}
