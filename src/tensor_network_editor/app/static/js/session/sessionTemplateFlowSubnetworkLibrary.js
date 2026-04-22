export function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function buildSerializedNetworkPreviewMarkup(serializedSpec) {
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

export function createSubnetworkLibrarySupport({
  documentRef,
  state,
  subnetworkLibraryList,
  subnetworkLibrarySearchInput,
  subnetworkLibraryTagFilter,
  subnetworkLibrarySelectAllInput,
  subnetworkLibrarySelectionSummary,
  subnetworkLibraryAddSelectedButton,
  subnetworkLibraryWarning,
  insertSubnetworkFromLibrary,
  renameLibrarySubnetwork,
  deleteLibrarySubnetwork,
  toggleSubnetworkLibraryBatchSelection,
}) {
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

  function getSelectedSubnetworkLibraryNames() {
    return new Set(
      Array.isArray(state.selectedSubnetworkLibraryNames)
        ? state.selectedSubnetworkLibraryNames
        : []
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
    return buildSerializedNetworkPreviewMarkup(entry?.serializedSpec);
  }

  function formatSubnetworkSourceLabel(source) {
    if (source === "shared") {
      return "Shared";
    }
    if (source === "session") {
      return "Session";
    }
    return "Project";
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

  function syncSubnetworkLibraryBatchControls(filteredEntries = []) {
    const selectedNames = getSelectedSubnetworkLibraryNames();
    const totalSelectedCount = selectedNames.size;
    const visibleNames = filteredEntries.map((entry) => entry.subnetworkName);
    const visibleSelectedCount = visibleNames.filter((subnetworkName) =>
      selectedNames.has(subnetworkName)
    ).length;
    if (subnetworkLibrarySelectAllInput) {
      subnetworkLibrarySelectAllInput.checked =
        visibleNames.length > 0 &&
        visibleSelectedCount === visibleNames.length;
      subnetworkLibrarySelectAllInput.indeterminate =
        visibleSelectedCount > 0 &&
        visibleSelectedCount < visibleNames.length;
      subnetworkLibrarySelectAllInput.disabled = !visibleNames.length;
    }
    if (subnetworkLibrarySelectionSummary) {
      if (!totalSelectedCount) {
        subnetworkLibrarySelectionSummary.textContent = "No subnetworks selected.";
      } else if (visibleSelectedCount === totalSelectedCount) {
        subnetworkLibrarySelectionSummary.textContent = `${totalSelectedCount} selected`;
      } else {
        subnetworkLibrarySelectionSummary.textContent =
          `${totalSelectedCount} selected (${visibleSelectedCount} visible)`;
      }
    }
    if (subnetworkLibraryAddSelectedButton) {
      subnetworkLibraryAddSelectedButton.disabled = !totalSelectedCount;
      subnetworkLibraryAddSelectedButton.textContent = totalSelectedCount
        ? `Add to session templates (${totalSelectedCount})`
        : "Add to session templates";
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
    syncSubnetworkLibraryBatchControls(filteredEntries);
    if (!filteredEntries.length) {
      const emptyState = documentRef.createElement("p");
      emptyState.className = "subnetwork-library-empty-state";
      emptyState.textContent = state.availableSubnetworks.length
        ? "No subnetworks match the current filters."
        : "No reusable subnetworks have been saved yet.";
      subnetworkLibraryList.appendChild(emptyState);
      return;
    }
    const selectedNames = getSelectedSubnetworkLibraryNames();
    filteredEntries.forEach((entry) => {
      const row = documentRef.createElement("article");
      row.className = "subnetwork-library-row";
      if (entry.subnetworkName === state.selectedSubnetworkName) {
        row.classList.add("is-selected");
      }
      if (selectedNames.has(entry.subnetworkName)) {
        row.classList.add("is-batch-selected");
      }
      row.addEventListener("click", () => {
        state.selectedSubnetworkName = entry.subnetworkName;
        renderSubnetworkLibrary();
      });

      const selectionCell = documentRef.createElement("label");
      selectionCell.className = "subnetwork-library-select-cell";
      selectionCell.addEventListener("click", (event) => event.stopPropagation());
      const selectionInput = documentRef.createElement("input");
      selectionInput.type = "checkbox";
      selectionInput.checked = selectedNames.has(entry.subnetworkName);
      selectionInput.setAttribute("aria-label", `Select ${entry.displayName}`);
      selectionInput.addEventListener("click", (event) => event.stopPropagation());
      selectionInput.addEventListener("change", (event) => {
        event.stopPropagation();
        toggleSubnetworkLibraryBatchSelection(
          entry.subnetworkName,
          Boolean(event.target.checked)
        );
      });
      selectionCell.appendChild(selectionInput);
      row.appendChild(selectionCell);

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
      sourceBadge.textContent = formatSubnetworkSourceLabel(entry.source);
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

  return {
    normalizeSubnetworkCatalogName,
    getSubnetworkEntryByName,
    getFilteredSubnetworkEntries,
    renderSubnetworkLibrary,
  };
}
