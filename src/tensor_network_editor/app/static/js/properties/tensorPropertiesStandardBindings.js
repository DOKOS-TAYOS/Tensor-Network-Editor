export function createStandardTensorPropertiesBindingSupport({
  bindDebouncedAutosave,
  bindImmediateAutosave,
  bindMetadataEditors,
  commands,
  ctx,
  document,
  propertiesPanel,
  propertyInvalidation,
  toggleTensorIndexDisclosure,
  dataSupport,
}) {
  const {
    getTensorShape,
    getTensorDataMode,
    getTensorDataDType,
    getTensorRandomDistribution,
    getTensorRandomSeed,
    getTensorExternalFilePath,
    getTensorExternalArrayKey,
    buildDefaultTensorLiteralValues,
    analyzeTensorDataFillInput,
    analyzeTensorLiteralInput,
  } = dataSupport;

  function bindListener(target, eventName, handler) {
    if (!target || typeof target.addEventListener !== "function") {
      return;
    }
    target.addEventListener(eventName, handler);
  }

  function readTensorDataModeChevronExpanded(fieldElement) {
    if (!fieldElement) {
      return false;
    }
    if (typeof fieldElement.getAttribute === "function") {
      return fieldElement.getAttribute("data-expanded") === "true";
    }
    return fieldElement.attributes?.["data-expanded"] === "true";
  }

  function setTensorDataModeChevronExpanded(fieldElement, isExpanded) {
    if (!fieldElement || typeof fieldElement.setAttribute !== "function") {
      return;
    }
    fieldElement.setAttribute("data-expanded", String(Boolean(isExpanded)));
  }

  function bindTensorDataModeChevronDisclosure(fieldElement, selectElement) {
    if (!fieldElement || !selectElement) {
      return;
    }
    setTensorDataModeChevronExpanded(fieldElement, false);
    bindListener(selectElement, "mousedown", () => {
      setTensorDataModeChevronExpanded(
        fieldElement,
        !readTensorDataModeChevronExpanded(fieldElement)
      );
    });
    bindListener(selectElement, "keydown", (event) => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        setTensorDataModeChevronExpanded(fieldElement, true);
      }
      if (["Escape", "Tab"].includes(event.key)) {
        setTensorDataModeChevronExpanded(fieldElement, false);
      }
    });
    bindListener(selectElement, "change", () => {
      setTensorDataModeChevronExpanded(fieldElement, false);
    });
    bindListener(selectElement, "blur", () => {
      setTensorDataModeChevronExpanded(fieldElement, false);
    });
  }

  function bindStandardTensorProperties(tensor) {
    const tensorShape = getTensorShape(tensor);
    const tensorDataMode = getTensorDataMode(tensor);
    const tensorDataDType = getTensorDataDType(tensor);
    const tensorFillValue =
      tensorDataMode === "fill" && tensor.tensor_data?.fill_value !== undefined
        ? tensor.tensor_data.fill_value
        : 0;
    const tensorRandomSeed = getTensorRandomSeed(tensor);
    const tensorRandomDistribution = getTensorRandomDistribution(tensor);
    const tensorExternalFilePath = getTensorExternalFilePath(tensor);
    const tensorExternalArrayKey = getTensorExternalArrayKey(tensor);
    const tensorNameInput = document.getElementById("tensor-name-input");
    const tensorColorInput = document.getElementById("tensor-color-input");
    const tensorTagsInput = document.getElementById("tensor-tags-input");
    const tensorCustomMetadataInput = document.getElementById(
      "tensor-custom-metadata-input"
    );
    const tensorDataModeField = document.getElementById("tensor-data-mode-field");
    const tensorDataModeSelect = document.getElementById("tensor-data-mode-select");
    const tensorDataDTypeField = document.getElementById("tensor-data-dtype-field");
    const tensorDataDTypeSelect = document.getElementById("tensor-data-dtype-select");
    const tensorDataFillInput = document.getElementById("tensor-data-fill-input");
    const tensorDataValuesInput = document.getElementById("tensor-data-values-input");
    const tensorDataExternalPathInput = document.getElementById(
      "tensor-data-external-path-input"
    );
    const tensorDataExternalArrayKeyInput = document.getElementById(
      "tensor-data-external-array-key-input"
    );
    const tensorDataRandomSeedInput = document.getElementById(
      "tensor-data-random-seed-input"
    );
    const tensorDataRandomDistributionField = document.getElementById(
      "tensor-data-random-distribution-field"
    );
    const tensorDataRandomDistributionSelect = document.getElementById(
      "tensor-data-random-distribution-select"
    );
    const tensorDataValidationMessage = document.getElementById(
      "tensor-data-validation-message"
    );

    function setTensorDataValidationMessage(message = "") {
      if (!tensorDataValidationMessage) {
        return;
      }
      tensorDataValidationMessage.textContent = message;
      tensorDataValidationMessage.hidden = !message;
    }

    function reportTensorDataValidationError(message) {
      setTensorDataValidationMessage(message);
      if (typeof ctx.setStatus === "function") {
        ctx.setStatus(message, "error");
      }
    }

    function readTensorDataDType() {
      const dtype = String(
        tensorDataDTypeSelect?.value || tensorDataDType || ""
      ).trim();
      return ["float32", "float64", "complex64", "complex128"].includes(dtype)
        ? dtype
        : "";
    }

    function withSelectedDType(tensorData) {
      const dtype = readTensorDataDType();
      if (dtype) {
        return {
          ...tensorData,
          dtype,
        };
      }
      return tensorData;
    }

    function validateTensorDataModeForShape(nextMode) {
      if (
        nextMode === "identity" &&
        (tensorShape.length !== 2 || tensorShape[0] !== tensorShape[1])
      ) {
        return `Identity requires a square rank-2 tensor; current shape is [${tensorShape.join(", ")}].`;
      }
      if (
        nextMode === "copy" &&
        (!tensorShape.length ||
          tensorShape.some((dimension) => dimension !== tensorShape[0]))
      ) {
        return `Copy tensor requires all index dimensions to match; current shape is [${tensorShape.join(", ")}].`;
      }
      return "";
    }

    function readRandomSeed() {
      const rawSeed =
        tensorDataRandomSeedInput?.value !== undefined
          ? tensorDataRandomSeedInput.value
          : String(tensorRandomSeed);
      const parsedSeed = Number(rawSeed);
      if (!Number.isInteger(parsedSeed) || parsedSeed < 0) {
        return null;
      }
      return parsedSeed;
    }

    function readRandomDistribution() {
      const distribution = String(
        tensorDataRandomDistributionSelect?.value || tensorRandomDistribution
      ).trim();
      return distribution === "uniform" ? "uniform" : "normal";
    }

    function readExternalFilePath() {
      const rawPath = tensorDataExternalPathInput?.value;
      return String(rawPath || tensorExternalFilePath).trim();
    }

    function readExternalArrayKey() {
      const rawArrayKey = tensorDataExternalArrayKeyInput?.value;
      return String(rawArrayKey || tensorExternalArrayKey).trim();
    }

    function buildExternalTensorData() {
      const filePath = readExternalFilePath();
      const arrayKey = readExternalArrayKey();
      const tensorData = {
        mode: "external",
        file_path: filePath,
      };
      if (arrayKey) {
        tensorData.array_key = arrayKey;
      }
      return tensorData;
    }

    function buildTensorDataForMode(nextMode) {
      const shapeError = validateTensorDataModeForShape(nextMode);
      if (shapeError) {
        return {
          ok: false,
          message: shapeError,
        };
      }
      if (nextMode === "zeros") {
        return {
          ok: true,
          tensorData: withSelectedDType({ mode: "zeros" }),
        };
      }
      if (nextMode === "ones") {
        return {
          ok: true,
          tensorData: withSelectedDType({ mode: "ones" }),
        };
      }
      if (nextMode === "fill") {
        const validation = analyzeTensorDataFillInput(
          tensorDataFillInput ? tensorDataFillInput.value : tensorFillValue
        );
        if (!validation.ok) {
          return validation;
        }
        return {
          ok: true,
          tensorData: withSelectedDType(validation.tensorData),
        };
      }
      if (nextMode === "literal") {
        const values =
          tensorDataValuesInput && tensorDataValuesInput.value.trim()
            ? analyzeTensorLiteralInput(tensorDataValuesInput.value, tensorShape)
            : {
                ok: true,
                tensorData: {
                  mode: "literal",
                  values: buildDefaultTensorLiteralValues(tensor, tensorShape),
                },
              };
        if (!values.ok) {
          return values;
        }
        return {
          ok: true,
          tensorData: withSelectedDType(values.tensorData),
        };
      }
      if (nextMode === "identity") {
        return {
          ok: true,
          tensorData: withSelectedDType({ mode: "identity" }),
        };
      }
      if (nextMode === "copy") {
        return {
          ok: true,
          tensorData: withSelectedDType({ mode: "copy" }),
        };
      }
      if (nextMode === "random") {
        const seed = readRandomSeed();
        if (seed === null) {
          return {
            ok: false,
            message: "Random seed must be a non-negative integer.",
          };
        }
        return {
          ok: true,
          tensorData: withSelectedDType({
            mode: "random",
            seed,
            distribution: readRandomDistribution(),
          }),
        };
      }
      if (nextMode === "external") {
        return {
          ok: true,
          tensorData: withSelectedDType(buildExternalTensorData()),
        };
      }
      return {
        ok: true,
        tensorData: null,
      };
    }

    function updateTensorDataFromControls(nextMode, invalidate = {}) {
      const validation = buildTensorDataForMode(nextMode);
      if (!validation.ok) {
        reportTensorDataValidationError(validation.message);
        return;
      }
      setTensorDataValidationMessage("");
      commands.updateTensorData({
        tensorId: tensor.id,
        nextTensorData: validation.tensorData,
        invalidate: propertyInvalidation(invalidate),
        statusMessage: `Updated tensor ${tensor.name}.`,
      });
    }

    bindDebouncedAutosave(
      tensorNameInput,
      `tensor:${tensor.id}:name`,
      () => {
        commands.renameTensor({
          tensor,
          proposedName: tensorNameInput.value,
          invalidate: propertyInvalidation({ graph: true, planner: true }),
          statusMessage: `Updated tensor ${tensorNameInput.value.trim()}.`,
        });
      }
    );
    bindImmediateAutosave(
      tensorColorInput,
      `tensor:${tensor.id}:color`,
      () => {
        commands.updateTargetColor({
          target: tensor,
          nextColor: tensorColorInput.value,
          invalidate: propertyInvalidation({ graph: true, minimap: true }),
          statusMessage: `Updated tensor ${tensor.name}.`,
        });
      },
      "input"
    );
    bindMetadataEditors({
      target: tensor,
      tagsInput: tensorTagsInput,
      tagsFieldKey: `tensor:${tensor.id}:tags`,
      customMetadataInput: tensorCustomMetadataInput,
      customMetadataFieldKey: `tensor:${tensor.id}:custom-metadata`,
      statusMessage: `Updated tensor ${tensor.name}.`,
      invalidate: propertyInvalidation(),
      annotationScope: "tensor",
    });
    bindTensorDataModeChevronDisclosure(tensorDataModeField, tensorDataModeSelect);
    bindTensorDataModeChevronDisclosure(tensorDataDTypeField, tensorDataDTypeSelect);
    bindTensorDataModeChevronDisclosure(
      tensorDataRandomDistributionField,
      tensorDataRandomDistributionSelect
    );
    bindImmediateAutosave(
      tensorDataModeSelect,
      `tensor:${tensor.id}:tensor-data-mode`,
      () => {
        const nextMode = String(tensorDataModeSelect.value || "zeros");
        setTensorDataModeChevronExpanded(tensorDataModeField, false);
        updateTensorDataFromControls(nextMode, { properties: true });
      }
    );
    bindImmediateAutosave(
      tensorDataDTypeSelect,
      `tensor:${tensor.id}:tensor-data-dtype`,
      () => {
        const nextMode = String(tensorDataModeSelect?.value || tensorDataMode);
        setTensorDataModeChevronExpanded(tensorDataDTypeField, false);
        updateTensorDataFromControls(nextMode, { properties: true });
      }
    );

    if (tensorDataFillInput) {
      tensorDataFillInput.addEventListener("input", () => {
        const validation = analyzeTensorDataFillInput(tensorDataFillInput.value);
        setTensorDataValidationMessage(validation.ok ? "" : validation.message);
      });
      bindDebouncedAutosave(
        tensorDataFillInput,
        `tensor:${tensor.id}:tensor-data-fill`,
        () => {
          const validation = analyzeTensorDataFillInput(tensorDataFillInput.value);
          if (!validation.ok) {
            reportTensorDataValidationError(validation.message);
            return;
          }
          setTensorDataValidationMessage("");
          commands.updateTensorData({
            tensorId: tensor.id,
            nextTensorData: withSelectedDType(validation.tensorData),
            invalidate: propertyInvalidation({ properties: true }),
            statusMessage: `Updated tensor ${tensor.name}.`,
          });
        },
        {
          scheduleOnInput: false,
        }
      );
    }

    if (tensorDataValuesInput) {
      tensorDataValuesInput.addEventListener("input", () => {
        const validation = analyzeTensorLiteralInput(
          tensorDataValuesInput.value,
          tensorShape
        );
        setTensorDataValidationMessage(validation.ok ? "" : validation.message);
      });
      bindDebouncedAutosave(
        tensorDataValuesInput,
        `tensor:${tensor.id}:tensor-data-values`,
        () => {
          const validation = analyzeTensorLiteralInput(
            tensorDataValuesInput.value,
            tensorShape
          );
          if (!validation.ok) {
            reportTensorDataValidationError(validation.message);
            return;
          }
          setTensorDataValidationMessage("");
          commands.updateTensorData({
            tensorId: tensor.id,
            nextTensorData: withSelectedDType(validation.tensorData),
            invalidate: propertyInvalidation({ properties: true }),
            statusMessage: `Updated tensor ${tensor.name}.`,
          });
        },
        {
          scheduleOnInput: false,
          commitOnEnter: false,
        }
      );
    }

    if (tensorDataExternalPathInput) {
      bindDebouncedAutosave(
        tensorDataExternalPathInput,
        `tensor:${tensor.id}:tensor-data-external-path`,
        () => {
          updateTensorDataFromControls("external", { properties: true });
        },
        {
          scheduleOnInput: false,
        }
      );
    }

    if (tensorDataExternalArrayKeyInput) {
      bindDebouncedAutosave(
        tensorDataExternalArrayKeyInput,
        `tensor:${tensor.id}:tensor-data-external-array-key`,
        () => {
          updateTensorDataFromControls("external", { properties: true });
        },
        {
          scheduleOnInput: false,
        }
      );
    }

    if (tensorDataRandomSeedInput) {
      tensorDataRandomSeedInput.addEventListener("input", () => {
        const seed = readRandomSeed();
        setTensorDataValidationMessage(
          seed === null ? "Random seed must be a non-negative integer." : ""
        );
      });
      bindDebouncedAutosave(
        tensorDataRandomSeedInput,
        `tensor:${tensor.id}:tensor-data-random-seed`,
        () => {
          updateTensorDataFromControls("random", { properties: true });
        },
        {
          scheduleOnInput: false,
        }
      );
    }

    bindImmediateAutosave(
      tensorDataRandomDistributionSelect,
      `tensor:${tensor.id}:tensor-data-random-distribution`,
      () => {
        setTensorDataModeChevronExpanded(tensorDataRandomDistributionField, false);
        updateTensorDataFromControls("random", { properties: true });
      }
    );

    document.getElementById("add-index-button").addEventListener("click", () => {
      commands.addTensorIndex({
        tensor,
        selectionIds: [tensor.id],
        primaryId: tensor.id,
        statusMessage: `Added one index to ${tensor.name}.`,
      });
    });
    document
      .getElementById("delete-tensor-button")
      .addEventListener("click", () => {
        commands.deleteTensor({
          tensorId: tensor.id,
          selectionIds: [],
          statusMessage: `Deleted tensor ${tensor.name}.`,
        });
      });

    propertiesPanel.querySelectorAll("[data-index-toggle]").forEach((button) => {
      button.addEventListener("click", () => {
        toggleTensorIndexDisclosure(tensor.id, button.dataset.indexToggle);
      });
    });

    tensor.indices.forEach((index, indexPosition) => {
      const indexColorInput = document.getElementById(`index-color-input-${index.id}`);
      const indexNameInput = document.getElementById(`index-name-input-${index.id}`);
      const indexDimensionInput = document.getElementById(
        `index-dimension-input-${index.id}`
      );
      const indexTagsInput = document.getElementById(`index-tags-input-${index.id}`);
      const indexCustomMetadataInput = document.getElementById(
        `index-custom-metadata-input-${index.id}`
      );
      const moveIndexUpButton = document.getElementById(
        `move-index-up-button-${index.id}`
      );
      const moveIndexDownButton = document.getElementById(
        `move-index-down-button-${index.id}`
      );
      const deleteIndexButton = document.getElementById(
        `delete-index-button-${index.id}`
      );

      bindDebouncedAutosave(
        indexNameInput,
        `index:${index.id}:name`,
        () => {
          commands.renameIndex({
            tensor,
            index,
            proposedName: indexNameInput.value,
            invalidate: propertyInvalidation({ graph: true }),
            statusMessage: `Updated index ${indexNameInput.value.trim()}.`,
          });
        }
      );
      bindDebouncedAutosave(
        indexDimensionInput,
        `index:${index.id}:dimension`,
        () => {
          commands.updateIndexDimension({
            indexId: index.id,
            rawValue: indexDimensionInput.value,
            invalidate: propertyInvalidation({
              graph: true,
              analysis: true,
            }),
            statusMessage: `Updated index ${index.name}.`,
          });
        }
      );
      bindImmediateAutosave(
        indexColorInput,
        `index:${index.id}:color`,
        () => {
          commands.updateTargetColor({
            target: index,
            nextColor: indexColorInput.value,
            invalidate: propertyInvalidation({ graph: true, minimap: true }),
            statusMessage: `Updated index ${index.name}.`,
          });
        },
        "input"
      );
      bindMetadataEditors({
        target: index,
        tagsInput: indexTagsInput,
        tagsFieldKey: `index:${index.id}:tags`,
        customMetadataInput: indexCustomMetadataInput,
        customMetadataFieldKey: `index:${index.id}:custom-metadata`,
        statusMessage: `Updated index ${index.name}.`,
        invalidate: propertyInvalidation(),
        annotationScope: "index",
      });

      if (moveIndexUpButton) {
        moveIndexUpButton.addEventListener("click", () => {
          commands.moveTensorIndex({
            tensorId: tensor.id,
            indexPosition,
            direction: -1,
            invalidate: propertyInvalidation({
              graph: true,
              lookups: true,
              properties: true,
            }),
            selectionIds: [index.id],
            primaryId: index.id,
            statusMessage: `Moved index ${index.name}.`,
          });
        });
      }
      if (moveIndexDownButton) {
        moveIndexDownButton.addEventListener("click", () => {
          commands.moveTensorIndex({
            tensorId: tensor.id,
            indexPosition,
            direction: 1,
            invalidate: propertyInvalidation({
              graph: true,
              lookups: true,
              properties: true,
            }),
            selectionIds: [index.id],
            primaryId: index.id,
            statusMessage: `Moved index ${index.name}.`,
          });
        });
      }
      if (deleteIndexButton) {
        deleteIndexButton.addEventListener("click", () => {
          commands.deleteTensorIndex({
            tensorId: tensor.id,
            indexId: index.id,
            selectionIds: [tensor.id],
            primaryId: tensor.id,
            statusMessage: `Deleted index ${index.name}.`,
          });
        });
      }
    });
  }

  return {
    bindStandardTensorProperties,
  };
}
