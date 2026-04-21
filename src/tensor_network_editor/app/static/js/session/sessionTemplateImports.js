function extractFileStem(filename) {
  if (typeof filename !== "string" || !filename.trim()) {
    return "template";
  }
  const trimmedFilename = filename.trim();
  const lastDotIndex = trimmedFilename.lastIndexOf(".");
  return lastDotIndex > 0
    ? trimmedFilename.slice(0, lastDotIndex)
    : trimmedFilename;
}

export function buildExportTemplatePayload(
  displayName,
  serializedSpec,
  sanitizeFilename
) {
  return {
    schema_version: 1,
    templates: [
      {
        name: sanitizeFilename(displayName).replaceAll("-", "_") || "template",
        display_name: displayName,
        spec: serializedSpec,
      },
    ],
  };
}

export function createSessionTemplateImportSupport({
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
}) {
  function normalizeSerializedSpec(parsedValue) {
    if (
      parsedValue &&
      typeof parsedValue === "object" &&
      parsedValue.network &&
      typeof parsedValue.network === "object"
    ) {
      return parsedValue;
    }
    if (
      parsedValue &&
      typeof parsedValue === "object" &&
      parsedValue.spec &&
      typeof parsedValue.spec === "object" &&
      parsedValue.spec.network &&
      typeof parsedValue.spec.network === "object"
    ) {
      return parsedValue.spec;
    }
    return null;
  }

  function buildTemplateImportsFromFile(parsedValue, filename) {
    const templateEntries = [];
    if (
      parsedValue &&
      typeof parsedValue === "object" &&
      Array.isArray(parsedValue.templates)
    ) {
      parsedValue.templates.forEach((entry, index) => {
        if (!entry || typeof entry !== "object") {
          return;
        }
        const serializedSpec = normalizeSerializedSpec(entry.spec || entry);
        if (!serializedSpec) {
          return;
        }
        const displayName =
          (typeof entry.display_name === "string" && entry.display_name.trim()) ||
          (typeof entry.displayName === "string" && entry.displayName.trim()) ||
          (typeof entry.name === "string" && entry.name.trim()) ||
          (serializedSpec.network &&
            typeof serializedSpec.network.name === "string" &&
            serializedSpec.network.name.trim()) ||
          `${extractFileStem(filename)} ${index + 1}`;
        templateEntries.push({
          displayName,
          serializedSpec,
        });
      });
      return templateEntries;
    }
    const serializedSpec = normalizeSerializedSpec(parsedValue);
    if (!serializedSpec) {
      return templateEntries;
    }
    const displayName =
      (parsedValue &&
        typeof parsedValue === "object" &&
        typeof parsedValue.display_name === "string" &&
        parsedValue.display_name.trim()) ||
      (serializedSpec.network &&
        typeof serializedSpec.network.name === "string" &&
        serializedSpec.network.name.trim()) ||
      extractFileStem(filename);
    templateEntries.push({
      displayName,
      serializedSpec,
    });
    return templateEntries;
  }

  async function loadSessionTemplatesFromFile(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) {
      return;
    }
    let loadedCount = 0;
    let duplicateCount = 0;
    let invalidCount = 0;
    const reservedDisplayNames = new Set(
      actions.listTemplateEntries().map((entry) => entry.displayName)
    );

    try {
      for (const file of files) {
        const fileText = await sessionUi.requestFileText(file, "utf-8");
        const parsedValue = JSON.parse(fileText);
        const templateImports = buildTemplateImportsFromFile(parsedValue, file.name);
        for (const templateImport of templateImports) {
          const displayName = templateImport.displayName.trim();
          if (!displayName || reservedDisplayNames.has(displayName)) {
            duplicateCount += 1;
            continue;
          }
          const validationResponse = await sessionService.validateSerializedSpec(
            templateImport.serializedSpec
          );
          if (!validationResponse.ok) {
            invalidCount += 1;
            continue;
          }
          const addResult = actions.addSessionTemplate({
            displayName,
            spec: validationResponse.spec,
            selected: false,
          });
          if (!addResult.ok) {
            duplicateCount += 1;
            continue;
          }
          reservedDisplayNames.add(displayName);
          loadedCount += 1;
        }
      }
      if (!loadedCount && !duplicateCount && !invalidCount) {
        actions.setStatus("No reusable templates were found in the selected files.", "error");
        return;
      }
      const summaryParts = [];
      if (loadedCount) {
        summaryParts.push(`Loaded ${loadedCount} template(s)`);
      }
      if (duplicateCount) {
        summaryParts.push(`skipped ${duplicateCount} duplicate name(s)`);
      }
      if (invalidCount) {
        summaryParts.push(`skipped ${invalidCount} invalid template(s)`);
      }
      actions.setStatus(
        `${summaryParts.join(", ")}.`,
        loadedCount ? "success" : "error"
      );
    } catch (error) {
      actions.setStatus(`Could not load templates: ${error.message}`, "error");
    } finally {
      if (templateLoadInput) {
        templateLoadInput.value = "";
      }
    }
  }

  async function loadSubnetworkFromFile(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }
    if (isForModeActive()) {
      actions.setStatus(
        "Subnetwork insertion is only available in normal graph mode.",
        "error"
      );
      if (subnetworkLoadInput) {
        subnetworkLoadInput.value = "";
      }
      return;
    }

    try {
      const fileText = await sessionUi.requestFileText(file, "utf-8");
      const parsed = JSON.parse(fileText);
      const serializedSpec =
        parsed && typeof parsed === "object" && parsed.network
          ? parsed
          : {
              schema_version: state.schemaVersion,
              network: parsed,
            };
      const response = await subnetworkService.prepareSubnetworkForInsert({
        serializedSpec,
        targetCenter: getSubnetworkTargetCenter(),
      });
      if (!response.ok) {
        actions.setStatus(
          response.message || actions.formatIssues(response.issues),
          "error"
        );
        return;
      }
      commands.insertPreparedSubnetwork(
        response.spec.network,
        response.spec.network.name || file.name
      );
    } catch (error) {
      actions.setStatus(`Could not insert ${file.name}: ${error.message}`, "error");
    } finally {
      if (subnetworkLoadInput) {
        subnetworkLoadInput.value = "";
      }
    }
  }

  return {
    buildTemplateImportsFromFile,
    loadSessionTemplatesFromFile,
    loadSubnetworkFromFile,
    normalizeSerializedSpec,
  };
}
