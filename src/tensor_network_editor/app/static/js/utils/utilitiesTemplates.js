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

const SESSION_TEMPLATE_PREFIX = "session::";
const TEMPLATE_SOURCE_LABELS = {
  session: "Session",
  project: "Project",
  global: "Built-in",
};
const TEMPLATE_PARAMETER_DEFAULTS = {
  graph_size: 1,
  bond_dimension: 1,
  physical_dimension: 1,
};

function buildSessionTemplateName(templateId) {
  return `${SESSION_TEMPLATE_PREFIX}${templateId}`;
}

function getTemplateSourceLabel(source) {
  return TEMPLATE_SOURCE_LABELS[source] || TEMPLATE_SOURCE_LABELS.global;
}

function isSessionTemplateName(templateName) {
  return typeof templateName === "string"
    && templateName.startsWith(SESSION_TEMPLATE_PREFIX);
}

function getSessionTemplateEntry(sessionTemplates, templateName) {
  if (!isSessionTemplateName(templateName) || !Array.isArray(sessionTemplates)) {
    return null;
  }
  return (
    sessionTemplates.find((entry) => entry && entry.templateName === templateName) || null
  );
}

function buildSessionTemplateDefinition(sessionTemplateEntry) {
  return {
    display_name: sessionTemplateEntry.displayName,
    graph_size_label: "Tensors",
    defaults: { ...TEMPLATE_PARAMETER_DEFAULTS },
    minimums: { ...TEMPLATE_PARAMETER_DEFAULTS },
    supports_parameters: false,
    source: "session",
  };
}

function buildMergedTemplateNames(state) {
  const sessionTemplateNames = Array.isArray(state.sessionTemplates)
    ? state.sessionTemplates.map((entry) => entry.templateName)
    : [];
  const catalogTemplateNames = Array.isArray(state.catalogTemplateNames)
    ? [...state.catalogTemplateNames]
    : [];
  const projectTemplateNames = catalogTemplateNames.filter((templateName) => {
    const definition = state.catalogTemplateDefinitions[templateName];
    return definition && definition.source === "project";
  });
  const builtinTemplateNames = catalogTemplateNames.filter(
    (templateName) => !projectTemplateNames.includes(templateName)
  );
  return [...sessionTemplateNames, ...projectTemplateNames, ...builtinTemplateNames];
}

function buildMergedTemplateDefinitions(state) {
  const mergedDefinitions = {
    ...(state.catalogTemplateDefinitions || {}),
  };
  (state.sessionTemplates || []).forEach((entry) => {
    mergedDefinitions[entry.templateName] = buildSessionTemplateDefinition(entry);
  });
  return mergedDefinitions;
}

export function formatEngineLabel(engineName) {
  return Object.prototype.hasOwnProperty.call(ENGINE_LABELS, engineName)
    ? ENGINE_LABELS[engineName]
    : engineName;
}

function sortEngineNamesForDisplay(engines) {
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

function formatCollectionFormatLabel(collectionFormat) {
  return Object.prototype.hasOwnProperty.call(
    COLLECTION_FORMAT_LABELS,
    collectionFormat
  )
    ? COLLECTION_FORMAT_LABELS[collectionFormat]
    : collectionFormat;
}

export function getTemplateDefinition(templateDefinitions, templateName) {
  if (
    !templateName
    || !templateDefinitions
    || typeof templateDefinitions !== "object"
  ) {
    return null;
  }
  return templateDefinitions[templateName] || null;
}

export function formatTemplateLabel(templateName, templateDefinitions) {
  const definition = getTemplateDefinition(templateDefinitions, templateName);
  if (
    definition
    && typeof definition.display_name === "string"
    && definition.display_name
  ) {
    return definition.display_name;
  }
  return templateName.replaceAll("_", " ");
}

function sanitizeTemplateIntegerValue(value, fallback, minimum) {
  const numericValue = Number(value);
  if (!Number.isInteger(numericValue)) {
    return Math.max(minimum, fallback);
  }
  return Math.max(minimum, numericValue);
}

function sanitizeTemplateChoiceValue(value, fallback, options) {
  if (typeof value !== "string" || !value.trim()) {
    return fallback;
  }
  return options.includes(value) ? value : fallback;
}

function buildLegacyTemplateParameterFields(definition) {
  const defaults = definition && definition.defaults ? definition.defaults : {};
  const minimums = definition && definition.minimums ? definition.minimums : {};
  return [
    {
      name: "graph_size",
      label: `Graph size (${definition?.graph_size_label || "Graph size"})`,
      kind: "integer",
      default: sanitizeTemplateIntegerValue(defaults.graph_size, 2, 2),
      minimum: sanitizeTemplateIntegerValue(minimums.graph_size, 2, 1),
    },
    {
      name: "bond_dimension",
      label: "Bond dimension",
      kind: "integer",
      default: sanitizeTemplateIntegerValue(defaults.bond_dimension, 3, 1),
      minimum: sanitizeTemplateIntegerValue(minimums.bond_dimension, 1, 1),
    },
    {
      name: "physical_dimension",
      label: "Physical dimension",
      kind: "integer",
      default: sanitizeTemplateIntegerValue(defaults.physical_dimension, 2, 1),
      minimum: sanitizeTemplateIntegerValue(minimums.physical_dimension, 1, 1),
    },
  ];
}

function getTemplateParameterFields(definition) {
  if (
    definition
    && Array.isArray(definition.parameter_fields)
    && definition.parameter_fields.length
  ) {
    return definition.parameter_fields;
  }
  return buildLegacyTemplateParameterFields(definition);
}

function buildTemplateParameterState(templateNames, templateDefinitions) {
  return Object.fromEntries(
    templateNames.map((templateName) => {
      const definition = getTemplateDefinition(templateDefinitions, templateName);
      const parameterFields = getTemplateParameterFields(definition);
      const parameterState = {};
      parameterFields.forEach((parameterField) => {
        if (parameterField.kind === "choice") {
          parameterState[parameterField.name] = sanitizeTemplateChoiceValue(
            parameterField.default,
            parameterField.default,
            Array.isArray(parameterField.options)
              ? parameterField.options.map((option) => option.value)
              : []
          );
          return;
        }
        parameterState[parameterField.name] = sanitizeTemplateIntegerValue(
          parameterField.default,
          parameterField.default,
          parameterField.minimum || 1
        );
      });
      return [
        templateName,
        parameterState,
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
  templateBoundaryConditionField,
  templateBoundaryConditionSelect,
  templateSymmetryField,
  templateSymmetrySelect,
  templateInitialStateField,
  templateInitialStateSelect,
  enforceLinearPeriodicEngineSupport,
  updateToolbarState,
}) {
  const templateChoiceControls = {
    boundary_condition: {
      field: templateBoundaryConditionField,
      select: templateBoundaryConditionSelect,
    },
    symmetry: {
      field: templateSymmetryField,
      select: templateSymmetrySelect,
    },
    initial_state: {
      field: templateInitialStateField,
      select: templateInitialStateSelect,
    },
  };

  function getStateTemplateDefinition(templateName = templateSelect.value) {
    return getTemplateDefinition(state.templateDefinitions, templateName);
  }

  function getTemplateSource(templateName = templateSelect.value) {
    const definition = getStateTemplateDefinition(templateName);
    return definition && typeof definition.source === "string"
      ? definition.source
      : "global";
  }

  function getTemplateSpec(templateName = templateSelect.value) {
    const sessionTemplateEntry = getSessionTemplateEntry(
      state.sessionTemplates,
      templateName
    );
    return sessionTemplateEntry ? structuredClone(sessionTemplateEntry.spec) : null;
  }

  function listTemplateEntries() {
    return state.availableTemplates.map((templateName) => ({
      templateName,
      displayName: formatTemplateLabel(templateName, state.templateDefinitions),
      source: getTemplateSource(templateName),
      spec: getTemplateSpec(templateName),
    }));
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
    const groupedTemplateNames = {
      session: [],
      project: [],
      global: [],
    };
    templateNames.forEach((templateName) => {
      groupedTemplateNames[getTemplateSource(templateName)].push(templateName);
    });
    Object.entries(groupedTemplateNames).forEach(([source, names]) => {
      if (!names.length) {
        return;
      }
      const optgroup = document.createElement("optgroup");
      optgroup.label = getTemplateSourceLabel(source);
      names.forEach((templateName) => {
        const option = document.createElement("option");
        option.value = templateName;
        option.textContent = formatTemplateLabel(
          templateName,
          state.templateDefinitions
        );
        optgroup.appendChild(option);
      });
      templateSelect.appendChild(optgroup);
    });
    if (templateNames.length && !templateSelect.value) {
      templateSelect.value = templateNames[0];
    }
  }

  function rebuildTemplateCatalog({
    selectedTemplate = null,
    templateCatalogWarnings = null,
  } = {}) {
    const nextTemplateNames = buildMergedTemplateNames(state);
    const nextTemplateDefinitions = buildMergedTemplateDefinitions(state);
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
    if (templateCatalogWarnings !== null) {
      state.templateCatalogWarnings = Array.isArray(templateCatalogWarnings)
        ? [...templateCatalogWarnings]
        : [];
    }
    state.templateParametersByTemplate = nextParameters;
    const currentTemplateValue = templateSelect.value;
    populateTemplateOptions(nextTemplateNames);
    if (selectedTemplate && nextTemplateNames.includes(selectedTemplate)) {
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

  function applyTemplateCatalogPayload({
    templateNames,
    templateDefinitions,
    selectedTemplate = null,
    templateCatalogWarnings = [],
  }) {
    state.catalogTemplateNames = Array.isArray(templateNames) ? [...templateNames] : [];
    state.catalogTemplateDefinitions =
      templateDefinitions && typeof templateDefinitions === "object"
        ? { ...templateDefinitions }
        : {};
    rebuildTemplateCatalog({
      selectedTemplate,
      templateCatalogWarnings,
    });
  }

  function syncChoiceControl(parameterField, parameters) {
    const control = templateChoiceControls[parameterField.name];
    if (!control || !control.field || !control.select) {
      return;
    }
    control.field.hidden = false;
    control.select.innerHTML = "";
    (Array.isArray(parameterField.options) ? parameterField.options : []).forEach(
      (optionDefinition) => {
        const option = document.createElement("option");
        option.value = optionDefinition.value;
        option.textContent = optionDefinition.label;
        if (option.value === parameters[parameterField.name]) {
          option.selected = true;
        }
        control.select.appendChild(option);
      }
    );
    control.select.value = parameters[parameterField.name];
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
    Object.values(templateChoiceControls).forEach((control) => {
      if (control.field) {
        control.field.hidden = true;
      }
    });
    if (!supportsParameters) {
      return;
    }
    const parameterFields = getTemplateParameterFields(definition);
    const parameters =
      state.templateParametersByTemplate[templateName]
      || buildTemplateParameterState([templateName], {
        [templateName]: definition,
      })[templateName];
    parameterFields.forEach((parameterField) => {
      if (parameterField.kind === "choice") {
        syncChoiceControl(parameterField, parameters);
        return;
      }
      if (parameterField.name === "graph_size") {
        templateGraphSizeLabel.textContent = parameterField.label;
        templateGraphSizeInput.min = String(parameterField.minimum || 1);
        templateGraphSizeInput.value = String(
          sanitizeTemplateIntegerValue(
            parameters.graph_size,
            parameterField.default,
            parameterField.minimum || 1
          )
        );
      }
      if (parameterField.name === "bond_dimension") {
        templateBondDimensionInput.min = String(parameterField.minimum || 1);
        templateBondDimensionInput.value = String(
          sanitizeTemplateIntegerValue(
            parameters.bond_dimension,
            parameterField.default,
            parameterField.minimum || 1
          )
        );
      }
      if (parameterField.name === "physical_dimension") {
        templatePhysicalDimensionInput.min = String(parameterField.minimum || 1);
        templatePhysicalDimensionInput.value = String(
          sanitizeTemplateIntegerValue(
            parameters.physical_dimension,
            parameterField.default,
            parameterField.minimum || 1
          )
        );
      }
    });
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
    const parameterFields = getTemplateParameterFields(definition);
    const parameters = {};
    parameterFields.forEach((parameterField) => {
      if (parameterField.kind === "choice") {
        const control = templateChoiceControls[parameterField.name];
        parameters[parameterField.name] = sanitizeTemplateChoiceValue(
          control && control.select ? control.select.value : parameterField.default,
          parameterField.default,
          Array.isArray(parameterField.options)
            ? parameterField.options.map((option) => option.value)
            : []
        );
        if (control && control.select) {
          control.select.value = parameters[parameterField.name];
        }
        return;
      }
      if (parameterField.name === "graph_size") {
        parameters.graph_size = sanitizeTemplateIntegerValue(
          templateGraphSizeInput.value,
          parameterField.default,
          parameterField.minimum || 1
        );
        templateGraphSizeInput.value = String(parameters.graph_size);
      }
      if (parameterField.name === "bond_dimension") {
        parameters.bond_dimension = sanitizeTemplateIntegerValue(
          templateBondDimensionInput.value,
          parameterField.default,
          parameterField.minimum || 1
        );
        templateBondDimensionInput.value = String(parameters.bond_dimension);
      }
      if (parameterField.name === "physical_dimension") {
        parameters.physical_dimension = sanitizeTemplateIntegerValue(
          templatePhysicalDimensionInput.value,
          parameterField.default,
          parameterField.minimum || 1
        );
        templatePhysicalDimensionInput.value = String(parameters.physical_dimension);
      }
    });
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

  function hasTemplateDisplayName(displayName, excludedTemplateName = null) {
    const normalizedDisplayName =
      typeof displayName === "string" ? displayName.trim() : "";
    if (!normalizedDisplayName) {
      return false;
    }
    return listTemplateEntries().some(
      (entry) =>
        entry.templateName !== excludedTemplateName
        && entry.displayName === normalizedDisplayName
    );
  }

  function getNextSessionTemplateDisplayName(baseDisplayName = "Selection Template") {
    let candidateDisplayName = baseDisplayName;
    let suffix = 2;
    while (hasTemplateDisplayName(candidateDisplayName)) {
      candidateDisplayName = `${baseDisplayName} ${suffix}`;
      suffix += 1;
    }
    return candidateDisplayName;
  }

  function addSessionTemplate({ displayName, spec, selected = true }) {
    const normalizedDisplayName =
      typeof displayName === "string" ? displayName.trim() : "";
    if (!normalizedDisplayName || !spec) {
      return { ok: false, reason: "invalid" };
    }
    if (hasTemplateDisplayName(normalizedDisplayName)) {
      return { ok: false, reason: "duplicate" };
    }
    const templateId = `template_${state.nextSessionTemplateId}`;
    state.nextSessionTemplateId += 1;
    const templateName = buildSessionTemplateName(templateId);
    state.sessionTemplates = [
      ...state.sessionTemplates,
      {
        id: templateId,
        templateName,
        displayName: normalizedDisplayName,
        spec: structuredClone(spec),
      },
    ];
    rebuildTemplateCatalog({
      selectedTemplate: selected ? templateName : templateSelect.value,
    });
    return {
      ok: true,
      templateName,
      displayName: normalizedDisplayName,
    };
  }

  function updateSessionTemplateDisplayNames(updates) {
    const updateMap = new Map(
      (Array.isArray(updates) ? updates : []).map((update) => [
        update.templateName,
        update.displayName,
      ])
    );
    state.sessionTemplates = state.sessionTemplates.map((entry) =>
      updateMap.has(entry.templateName)
        ? {
            ...entry,
            displayName: updateMap.get(entry.templateName).trim(),
          }
        : entry
    );
    rebuildTemplateCatalog({
      selectedTemplate: templateSelect.value,
    });
  }

  function removeSessionTemplate(templateName) {
    if (!isSessionTemplateName(templateName)) {
      return false;
    }
    const nextSessionTemplates = state.sessionTemplates.filter(
      (entry) => entry.templateName !== templateName
    );
    if (nextSessionTemplates.length === state.sessionTemplates.length) {
      return false;
    }
    state.sessionTemplates = nextSessionTemplates;
    rebuildTemplateCatalog();
    return true;
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
    getTemplateSpec,
    listTemplateEntries,
    buildTemplateParameterState: (templateNames, templateDefinitions) =>
      buildTemplateParameterState(templateNames, templateDefinitions),
    applyTemplateCatalogPayload,
    rebuildTemplateCatalog,
    syncTemplateParameterControls,
    readTemplateParametersFromControls,
    persistTemplateParametersFromControls,
    handleTemplateSelectionChange,
    handleTemplateParameterInput,
    hasTemplateDisplayName,
    getNextSessionTemplateDisplayName,
    addSessionTemplate,
    updateSessionTemplateDisplayNames,
    removeSessionTemplate,
    isSessionTemplateName,
  };
}
