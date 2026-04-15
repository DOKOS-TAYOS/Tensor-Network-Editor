const ENGINE_LABELS = {
  tensornetwork: "TensorNetwork",
  quimb: "Quimb",
  tensorkrowch: "TensorKrowch",
  einsum_numpy: "NumPy einsum",
  einsum_torch: "PyTorch einsum",
};

const ENGINE_DISPLAY_ORDER = [
  "tensorkrowch",
  "einsum_torch",
  "einsum_numpy",
  "quimb",
  "tensornetwork",
];

const COLLECTION_FORMAT_LABELS = {
  list: "List",
  matrix: "Matrix",
  dict: "Dictionary",
};

export function formatEngineLabel(engineName) {
  return Object.prototype.hasOwnProperty.call(ENGINE_LABELS, engineName)
    ? ENGINE_LABELS[engineName]
    : engineName;
}

export function sortEngineNamesForDisplay(engines) {
  const preferredOrder = new Map(
    ENGINE_DISPLAY_ORDER.map((engineName, position) => [engineName, position])
  );
  return [...engines].sort((leftEngine, rightEngine) => {
    const leftPriority = preferredOrder.has(leftEngine)
      ? preferredOrder.get(leftEngine)
      : Number.MAX_SAFE_INTEGER;
    const rightPriority = preferredOrder.has(rightEngine)
      ? preferredOrder.get(rightEngine)
      : Number.MAX_SAFE_INTEGER;
    if (leftPriority !== rightPriority) {
      return leftPriority - rightPriority;
    }
    return formatEngineLabel(leftEngine).localeCompare(
      formatEngineLabel(rightEngine)
    );
  });
}

export function formatCollectionFormatLabel(collectionFormat) {
  return Object.prototype.hasOwnProperty.call(
    COLLECTION_FORMAT_LABELS,
    collectionFormat
  )
    ? COLLECTION_FORMAT_LABELS[collectionFormat]
    : collectionFormat;
}

export function getTemplateDefinition(templateDefinitions, templateName) {
  if (
    !templateName ||
    !templateDefinitions ||
    typeof templateDefinitions !== "object"
  ) {
    return null;
  }
  return templateDefinitions[templateName] || null;
}

export function formatTemplateLabel(templateName, templateDefinitions) {
  const definition = getTemplateDefinition(templateDefinitions, templateName);
  if (
    definition &&
    typeof definition.display_name === "string" &&
    definition.display_name
  ) {
    return definition.display_name;
  }
  return templateName.replaceAll("_", " ");
}

export function sanitizeTemplateIntegerValue(value, fallback, minimum) {
  const numericValue = Number(value);
  if (!Number.isInteger(numericValue)) {
    return Math.max(minimum, fallback);
  }
  return Math.max(minimum, numericValue);
}

export function buildTemplateParameterState(templateNames, templateDefinitions) {
  return Object.fromEntries(
    templateNames.map((templateName) => {
      const definition = getTemplateDefinition(templateDefinitions, templateName);
      const defaults = definition && definition.defaults ? definition.defaults : {};
      return [
        templateName,
        {
          graph_size: sanitizeTemplateIntegerValue(defaults.graph_size, 2, 2),
          bond_dimension: sanitizeTemplateIntegerValue(
            defaults.bond_dimension,
            3,
            1
          ),
          physical_dimension: sanitizeTemplateIntegerValue(
            defaults.physical_dimension,
            2,
            1
          ),
        },
      ];
    })
  );
}

export function createTemplateOptionHelpers({
  state,
  document,
  engineSelect,
  collectionFormatSelect,
  templateSelect,
  templateParameterPanel,
  templateGraphSizeField,
  templateGraphSizeLabel,
  templateGraphSizeInput,
  templateBondDimensionField,
  templateBondDimensionInput,
  templatePhysicalDimensionField,
  templatePhysicalDimensionInput,
  enforceLinearPeriodicEngineSupport,
  updateToolbarState,
}) {
  function getStateTemplateDefinition(templateName = templateSelect.value) {
    return getTemplateDefinition(state.templateDefinitions, templateName);
  }

  function getTemplateSource(templateName = templateSelect.value) {
    const definition = getStateTemplateDefinition(templateName);
    return definition && typeof definition.source === "string"
      ? definition.source
      : "global";
  }

  function populateEngineOptions(engines) {
    engineSelect.innerHTML = "";
    sortEngineNamesForDisplay(engines).forEach((engineName) => {
      const option = document.createElement("option");
      option.value = engineName;
      option.textContent = formatEngineLabel(engineName);
      if (engineName === state.selectedEngine) {
        option.selected = true;
      }
      engineSelect.appendChild(option);
    });
    enforceLinearPeriodicEngineSupport();
  }

  function populateCollectionFormatOptions(collectionFormats) {
    if (!collectionFormatSelect) {
      return;
    }
    collectionFormatSelect.innerHTML = "";
    collectionFormats.forEach((collectionFormat) => {
      const option = document.createElement("option");
      option.value = collectionFormat;
      option.textContent = formatCollectionFormatLabel(collectionFormat);
      if (collectionFormat === state.selectedCollectionFormat) {
        option.selected = true;
      }
      collectionFormatSelect.appendChild(option);
    });
  }

  function populateTemplateOptions(templateNames) {
    templateSelect.innerHTML = "";
    templateNames.forEach((templateName) => {
      const option = document.createElement("option");
      option.value = templateName;
      option.textContent = formatTemplateLabel(
        templateName,
        state.templateDefinitions
      );
      templateSelect.appendChild(option);
    });
    if (templateNames.length && !templateSelect.value) {
      templateSelect.value = templateNames[0];
    }
  }

  function applyTemplateCatalogPayload({
    templateNames,
    templateDefinitions,
    selectedTemplate = null,
    templateCatalogWarnings = [],
  }) {
    const nextTemplateNames = Array.isArray(templateNames) ? [...templateNames] : [];
    const nextTemplateDefinitions =
      templateDefinitions && typeof templateDefinitions === "object"
        ? { ...templateDefinitions }
        : {};
    const previousParameters = state.templateParametersByTemplate || {};
    const nextParameters = buildTemplateParameterState(
      nextTemplateNames,
      nextTemplateDefinitions
    );
    nextTemplateNames.forEach((templateName) => {
      if (previousParameters[templateName]) {
        nextParameters[templateName] = {
          ...nextParameters[templateName],
          ...previousParameters[templateName],
        };
      }
    });
    state.availableTemplates = nextTemplateNames;
    state.templateDefinitions = nextTemplateDefinitions;
    state.templateCatalogWarnings = Array.isArray(templateCatalogWarnings)
      ? [...templateCatalogWarnings]
      : [];
    state.templateParametersByTemplate = nextParameters;
    const currentTemplateValue = templateSelect.value;
    populateTemplateOptions(nextTemplateNames);
    if (
      selectedTemplate &&
      nextTemplateNames.includes(selectedTemplate)
    ) {
      templateSelect.value = selectedTemplate;
    } else if (nextTemplateNames.includes(currentTemplateValue)) {
      templateSelect.value = currentTemplateValue;
    } else if (nextTemplateNames.length) {
      templateSelect.value = nextTemplateNames[0];
    } else {
      templateSelect.value = "";
    }
    syncTemplateParameterControls(templateSelect.value);
    updateToolbarState();
  }

  function syncTemplateParameterControls(templateName = templateSelect.value) {
    if (!templateParameterPanel) {
      return;
    }
    const definition = getStateTemplateDefinition(templateName);
    if (!definition) {
      templateParameterPanel.hidden = true;
      return;
    }
    templateParameterPanel.hidden = false;
    const supportsParameters = definition.supports_parameters !== false;
    if (templateGraphSizeField) {
      templateGraphSizeField.hidden = !supportsParameters;
    }
    if (templateBondDimensionField) {
      templateBondDimensionField.hidden = !supportsParameters;
    }
    if (templatePhysicalDimensionField) {
      templatePhysicalDimensionField.hidden = !supportsParameters;
    }
    if (!supportsParameters) {
      return;
    }
    const minimums = definition.minimums || {};
    const defaults = definition.defaults || {};
    const parameters =
      state.templateParametersByTemplate[templateName] ||
      buildTemplateParameterState([templateName], {
        [templateName]: definition,
      })[templateName];
    templateGraphSizeLabel.textContent = `Graph size (${
      definition.graph_size_label || "Graph size"
    })`;
    templateGraphSizeInput.min = String(
      sanitizeTemplateIntegerValue(minimums.graph_size, 2, 1)
    );
    templateBondDimensionInput.min = String(
      sanitizeTemplateIntegerValue(minimums.bond_dimension, 1, 1)
    );
    templatePhysicalDimensionInput.min = String(
      sanitizeTemplateIntegerValue(minimums.physical_dimension, 1, 1)
    );
    templateGraphSizeInput.value = String(
      sanitizeTemplateIntegerValue(
        parameters.graph_size,
        sanitizeTemplateIntegerValue(defaults.graph_size, 2, 2),
        sanitizeTemplateIntegerValue(minimums.graph_size, 2, 1)
      )
    );
    templateBondDimensionInput.value = String(
      sanitizeTemplateIntegerValue(
        parameters.bond_dimension,
        sanitizeTemplateIntegerValue(defaults.bond_dimension, 3, 1),
        sanitizeTemplateIntegerValue(minimums.bond_dimension, 1, 1)
      )
    );
    templatePhysicalDimensionInput.value = String(
      sanitizeTemplateIntegerValue(
        parameters.physical_dimension,
        sanitizeTemplateIntegerValue(defaults.physical_dimension, 2, 1),
        sanitizeTemplateIntegerValue(minimums.physical_dimension, 1, 1)
      )
    );
  }

  function readTemplateParametersFromControls() {
    const definition = getStateTemplateDefinition();
    if (!definition) {
      return {
        graph_size: 2,
        bond_dimension: 3,
        physical_dimension: 2,
      };
    }
    if (definition.supports_parameters === false) {
      const defaults = definition.defaults || {};
      return {
        graph_size: sanitizeTemplateIntegerValue(defaults.graph_size, 1, 1),
        bond_dimension: sanitizeTemplateIntegerValue(defaults.bond_dimension, 1, 1),
        physical_dimension: sanitizeTemplateIntegerValue(
          defaults.physical_dimension,
          1,
          1
        ),
      };
    }
    const minimums = definition.minimums || {};
    const defaults = definition.defaults || {};
    const parameters = {
      graph_size: sanitizeTemplateIntegerValue(
        templateGraphSizeInput.value,
        sanitizeTemplateIntegerValue(defaults.graph_size, 2, 2),
        sanitizeTemplateIntegerValue(minimums.graph_size, 2, 1)
      ),
      bond_dimension: sanitizeTemplateIntegerValue(
        templateBondDimensionInput.value,
        sanitizeTemplateIntegerValue(defaults.bond_dimension, 3, 1),
        sanitizeTemplateIntegerValue(minimums.bond_dimension, 1, 1)
      ),
      physical_dimension: sanitizeTemplateIntegerValue(
        templatePhysicalDimensionInput.value,
        sanitizeTemplateIntegerValue(defaults.physical_dimension, 2, 1),
        sanitizeTemplateIntegerValue(minimums.physical_dimension, 1, 1)
      ),
    };
    templateGraphSizeInput.value = String(parameters.graph_size);
    templateBondDimensionInput.value = String(parameters.bond_dimension);
    templatePhysicalDimensionInput.value = String(parameters.physical_dimension);
    return parameters;
  }

  function persistTemplateParametersFromControls() {
    const templateName = templateSelect.value;
    if (!templateName) {
      return null;
    }
    const parameters = readTemplateParametersFromControls();
    state.templateParametersByTemplate[templateName] = { ...parameters };
    return parameters;
  }

  function handleTemplateSelectionChange(event) {
    if (!event || !event.target) {
      return;
    }
    syncTemplateParameterControls(event.target.value);
    updateToolbarState();
  }

  function handleTemplateParameterInput() {
    persistTemplateParametersFromControls();
  }

  return {
    populateEngineOptions,
    formatEngineLabel,
    formatCollectionFormatLabel,
    populateCollectionFormatOptions,
    populateTemplateOptions,
    formatTemplateLabel: (templateName) =>
      formatTemplateLabel(templateName, state.templateDefinitions),
    getTemplateDefinition: getStateTemplateDefinition,
    getTemplateSource,
    buildTemplateParameterState: (templateNames, templateDefinitions) =>
      buildTemplateParameterState(templateNames, templateDefinitions),
    applyTemplateCatalogPayload,
    syncTemplateParameterControls,
    readTemplateParametersFromControls,
    persistTemplateParametersFromControls,
    handleTemplateSelectionChange,
    handleTemplateParameterInput,
  };
}
