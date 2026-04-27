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

function sanitizeTemplateNumberValue(value, fallback, minimum = null) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    if (minimum === null || minimum === undefined) {
      return fallback;
    }
    return Math.max(minimum, fallback);
  }
  if (minimum === null || minimum === undefined) {
    return numericValue;
  }
  return Math.max(minimum, numericValue);
}

function sanitizeTemplateBooleanValue(value, fallback) {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalizedValue = value.trim().toLowerCase();
    if (["true", "1", "yes", "on"].includes(normalizedValue)) {
      return true;
    }
    if (["false", "0", "no", "off"].includes(normalizedValue)) {
      return false;
    }
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  return Boolean(fallback);
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

function getChoiceOptionValues(parameterField) {
  return Array.isArray(parameterField.options)
    ? parameterField.options.map((option) => option.value)
    : [];
}

function sanitizeTemplateParameterFieldValue(parameterField, value) {
  if (parameterField.kind === "choice") {
    return sanitizeTemplateChoiceValue(
      value,
      parameterField.default,
      getChoiceOptionValues(parameterField)
    );
  }
  if (parameterField.kind === "number") {
    return sanitizeTemplateNumberValue(
      value,
      parameterField.default,
      parameterField.minimum
    );
  }
  if (parameterField.kind === "boolean") {
    return sanitizeTemplateBooleanValue(value, parameterField.default);
  }
  return sanitizeTemplateIntegerValue(
    value,
    parameterField.default,
    parameterField.minimum || 1
  );
}

function buildTemplateParameterStateForDefinition(definition, sourceParameters = {}) {
  const parameterFields = getTemplateParameterFields(definition);
  const parameterState = {};
  parameterFields.forEach((parameterField) => {
    parameterState[parameterField.name] = sanitizeTemplateParameterFieldValue(
      parameterField,
      Object.prototype.hasOwnProperty.call(sourceParameters, parameterField.name)
        ? sourceParameters[parameterField.name]
        : parameterField.default
    );
  });
  return parameterState;
}

function buildTemplateParameterState(
  templateNames,
  templateDefinitions,
  sourceParametersByTemplate = {}
) {
  return Object.fromEntries(
    templateNames.map((templateName) => {
      const definition = getTemplateDefinition(templateDefinitions, templateName);
      return [
        templateName,
        buildTemplateParameterStateForDefinition(
          definition,
          sourceParametersByTemplate[templateName] || {}
        ),
      ];
    })
  );
}

function setElementAttribute(element, attributeName, attributeValue) {
  if (!element) {
    return;
  }
  if (typeof element.setAttribute === "function") {
    element.setAttribute(attributeName, attributeValue);
    return;
  }
  if (!element.attributes) {
    element.attributes = {};
  }
  element.attributes[attributeName] = String(attributeValue);
}

function findClosestElementByPredicate(target, predicate) {
  let currentTarget = target;
  while (currentTarget) {
    if (predicate(currentTarget)) {
      return currentTarget;
    }
    currentTarget = currentTarget.parentElement || currentTarget.parentNode || null;
  }
  return null;
}

function isRenderableTemplateParameterPanel(document, templateParameterPanel) {
  return Boolean(
    document
      && typeof document.createElement === "function"
      && templateParameterPanel
      && typeof templateParameterPanel.appendChild === "function"
  );
}

function clearTemplateParameterPanel(templateParameterPanel) {
  if (!templateParameterPanel) {
    return;
  }
  if (typeof templateParameterPanel.replaceChildren === "function") {
    templateParameterPanel.replaceChildren();
    return;
  }
  if ("innerHTML" in templateParameterPanel) {
    templateParameterPanel.innerHTML = "";
  }
  if (Array.isArray(templateParameterPanel.children)) {
    templateParameterPanel.children.length = 0;
  }
}

function appendTemplateParameterTitle(document, fieldElement, parameterField) {
  const title = document.createElement("span");
  title.className = "template-parameter-title";
  title.textContent = parameterField.label;
  fieldElement.appendChild(title);
}

function createTemplateNumberControl(document, parameterField, value) {
  const input = document.createElement("input");
  input.type = "number";
  input.value = String(value);
  input.step = parameterField.kind === "number" ? "any" : "1";
  input.inputMode = parameterField.kind === "number" ? "decimal" : "numeric";
  if (parameterField.minimum !== undefined && parameterField.minimum !== null) {
    input.min = String(parameterField.minimum);
  }
  return input;
}

function createTemplateChoiceControl(document, parameterField, value) {
  const select = document.createElement("select");
  (Array.isArray(parameterField.options) ? parameterField.options : []).forEach(
    (optionDefinition) => {
      const option = document.createElement("option");
      option.value = optionDefinition.value;
      option.textContent = optionDefinition.label;
      option.selected = option.value === value;
      select.appendChild(option);
    }
  );
  select.value = value;
  return select;
}

function createTemplateBooleanControl(document, parameterField, value) {
  const toggleShell = document.createElement("span");
  toggleShell.className = "template-boolean-toggle";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(value);
  const valueLabel = document.createElement("span");
  valueLabel.className = "template-boolean-label";
  valueLabel.textContent = "Enabled";
  toggleShell.appendChild(input);
  toggleShell.appendChild(valueLabel);
  return {
    control: input,
    shell: toggleShell,
  };
}

function renderTemplateParameterField(document, parameterField, value) {
  const fieldElement = document.createElement("label");
  fieldElement.id = `template-parameter-${parameterField.name}-field`;
  fieldElement.dataset.templateParameterName = parameterField.name;
  fieldElement.dataset.templateParameterKind = parameterField.kind;

  let control = null;
  if (parameterField.kind === "choice") {
    fieldElement.className =
      "template-parameter-field template-select-field select-chevron-field";
    setElementAttribute(fieldElement, "data-expanded", "false");
    appendTemplateParameterTitle(document, fieldElement, parameterField);
    control = createTemplateChoiceControl(document, parameterField, value);
    fieldElement.appendChild(control);
  } else if (parameterField.kind === "boolean") {
    fieldElement.className =
      "template-parameter-field template-boolean-field";
    appendTemplateParameterTitle(document, fieldElement, parameterField);
    const booleanControl = createTemplateBooleanControl(
      document,
      parameterField,
      value
    );
    control = booleanControl.control;
    fieldElement.appendChild(booleanControl.shell);
  } else {
    fieldElement.className =
      "template-parameter-field template-number-field";
    appendTemplateParameterTitle(document, fieldElement, parameterField);
    control = createTemplateNumberControl(document, parameterField, value);
    fieldElement.appendChild(control);
  }

  control.id = `template-parameter-${parameterField.name}-input`;
  control.dataset.templateParameterName = parameterField.name;
  control.dataset.templateParameterKind = parameterField.kind;
  control.setAttribute?.("aria-label", parameterField.label);
  if (parameterField.kind !== "boolean") {
    setElementAttribute(fieldElement, "for", control.id);
  }
  return { fieldElement, control };
}

export function createTemplateOptionHelpers({
  state,
  document,
  engineSelect,
  collectionFormatSelect,
  templateSelect,
  templateParameterPanel,
  enforceLinearPeriodicEngineSupport,
  updateToolbarState,
}) {
  const renderedParameterControls = new Map();

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
      nextTemplateDefinitions,
      previousParameters
    );
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

  function syncTemplateParameterControls(templateName = templateSelect.value) {
    if (!templateParameterPanel) {
      return;
    }
    renderedParameterControls.clear();
    const definition = getStateTemplateDefinition(templateName);
    const supportsParameters = Boolean(
      definition && definition.supports_parameters !== false
    );
    const parameterFields = definition ? getTemplateParameterFields(definition) : [];
    if (!definition || !supportsParameters || !parameterFields.length) {
      templateParameterPanel.hidden = true;
      clearTemplateParameterPanel(templateParameterPanel);
      return;
    }
    templateParameterPanel.hidden = false;
    if (!isRenderableTemplateParameterPanel(document, templateParameterPanel)) {
      return;
    }
    clearTemplateParameterPanel(templateParameterPanel);
    const parameters =
      state.templateParametersByTemplate[templateName]
      || buildTemplateParameterState([templateName], {
        [templateName]: definition,
      })[templateName];
    parameterFields.forEach((parameterField) => {
      const renderedField = renderTemplateParameterField(
        document,
        parameterField,
        parameters[parameterField.name]
      );
      renderedParameterControls.set(parameterField.name, {
        parameterField,
        fieldElement: renderedField.fieldElement,
        control: renderedField.control,
      });
      templateParameterPanel.appendChild(renderedField.fieldElement);
    });
  }

  function readTemplateParametersFromControls() {
    const templateName = templateSelect.value;
    const definition = getStateTemplateDefinition(templateName);
    if (!definition) {
      return buildTemplateParameterStateForDefinition({
        defaults: TEMPLATE_PARAMETER_DEFAULTS,
        minimums: TEMPLATE_PARAMETER_DEFAULTS,
      });
    }
    const fallbackParameters =
      state.templateParametersByTemplate[templateName]
      || buildTemplateParameterState([templateName], {
        [templateName]: definition,
      })[templateName];
    if (definition.supports_parameters === false || !renderedParameterControls.size) {
      return { ...fallbackParameters };
    }
    const parameters = {};
    getTemplateParameterFields(definition).forEach((parameterField) => {
      const renderedControl = renderedParameterControls.get(parameterField.name);
      const control = renderedControl ? renderedControl.control : null;
      if (parameterField.kind === "boolean") {
        parameters[parameterField.name] = sanitizeTemplateParameterFieldValue(
          parameterField,
          control ? control.checked : fallbackParameters[parameterField.name]
        );
        if (control) {
          control.checked = parameters[parameterField.name];
        }
        return;
      }
      parameters[parameterField.name] = sanitizeTemplateParameterFieldValue(
        parameterField,
        control ? control.value : fallbackParameters[parameterField.name]
      );
      if (control) {
        control.value = String(parameters[parameterField.name]);
      }
    });
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

  function handleTemplateParameterPanelDisclosure(event) {
    const target = event && event.target ? event.target : null;
    const fieldElement = findClosestElementByPredicate(
      target,
      (candidate) =>
        candidate
        && candidate.dataset
        && candidate.dataset.templateParameterName
        && typeof candidate.className === "string"
        && candidate.className.includes("select-chevron-field")
    );
    if (!fieldElement) {
      return;
    }
    const currentExpandedValue =
      typeof fieldElement.getAttribute === "function"
        ? fieldElement.getAttribute("data-expanded")
        : fieldElement.attributes?.["data-expanded"];
    setElementAttribute(
      fieldElement,
      "data-expanded",
      currentExpandedValue === "true" ? "false" : "true"
    );
  }

  function handleTemplateParameterPanelKeydown(event) {
    const target = event && event.target ? event.target : null;
    const fieldElement = findClosestElementByPredicate(
      target,
      (candidate) =>
        candidate
        && candidate.dataset
        && candidate.dataset.templateParameterName
        && typeof candidate.className === "string"
        && candidate.className.includes("select-chevron-field")
    );
    if (!fieldElement) {
      return;
    }
    if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
      setElementAttribute(fieldElement, "data-expanded", "true");
    }
    if (["Escape", "Tab"].includes(event.key)) {
      setElementAttribute(fieldElement, "data-expanded", "false");
    }
  }

  function handleTemplateParameterPanelFocusOut(event) {
    const target = event && event.target ? event.target : null;
    const fieldElement = findClosestElementByPredicate(
      target,
      (candidate) =>
        candidate
        && candidate.dataset
        && candidate.dataset.templateParameterName
        && typeof candidate.className === "string"
        && candidate.className.includes("select-chevron-field")
    );
    if (!fieldElement) {
      return;
    }
    setElementAttribute(fieldElement, "data-expanded", "false");
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
    buildTemplateParameterState: (
      templateNames,
      templateDefinitions,
      sourceParametersByTemplate = {}
    ) => buildTemplateParameterState(
      templateNames,
      templateDefinitions,
      sourceParametersByTemplate
    ),
    applyTemplateCatalogPayload,
    rebuildTemplateCatalog,
    syncTemplateParameterControls,
    readTemplateParametersFromControls,
    persistTemplateParametersFromControls,
    handleTemplateSelectionChange,
    handleTemplateParameterInput,
    handleTemplateParameterPanelDisclosure,
    handleTemplateParameterPanelKeydown,
    handleTemplateParameterPanelFocusOut,
    hasTemplateDisplayName,
    getNextSessionTemplateDisplayName,
    addSessionTemplate,
    updateSessionTemplateDisplayNames,
    removeSessionTemplate,
    isSessionTemplateName,
  };
}
