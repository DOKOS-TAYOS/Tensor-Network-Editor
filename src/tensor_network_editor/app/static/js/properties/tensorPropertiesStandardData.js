export function createStandardTensorDataSupport() {
  function getTensorShape(tensor) {
    return Array.isArray(tensor?.indices)
      ? tensor.indices.map((index) => index.dimension)
      : [];
  }

  function formatTensorShape(shape) {
    return Array.isArray(shape) && shape.length ? `[${shape.join(", ")}]` : "scalar";
  }

  function tensorShapesMatch(leftShape, rightShape) {
    if (!Array.isArray(leftShape) || !Array.isArray(rightShape)) {
      return false;
    }
    if (leftShape.length !== rightShape.length) {
      return false;
    }
    return leftShape.every((dimension, position) => dimension === rightShape[position]);
  }

  function normalizeTensorLiteralNode(value) {
    if (typeof value === "boolean") {
      return null;
    }
    if (typeof value === "number") {
      if (!Number.isFinite(value)) {
        return null;
      }
      return {
        normalized: value,
        shape: [],
      };
    }
    if (!Array.isArray(value) || !value.length) {
      return null;
    }

    const normalizedChildren = [];
    let childShape = null;
    for (const item of value) {
      const normalizedItem = normalizeTensorLiteralNode(item);
      if (!normalizedItem) {
        return null;
      }
      if (!childShape) {
        childShape = normalizedItem.shape;
      } else if (!tensorShapesMatch(childShape, normalizedItem.shape)) {
        return null;
      }
      normalizedChildren.push(normalizedItem.normalized);
    }

    return {
      normalized: normalizedChildren,
      shape: [normalizedChildren.length, ...(childShape || [])],
    };
  }

  function getTensorDataMode(tensor) {
    const mode = String(tensor?.tensor_data?.mode || "").trim();
    if (mode === "ones" || mode === "fill" || mode === "literal") {
      return mode;
    }
    return "zeros";
  }

  function cloneTensorLiteral(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function buildFilledTensorLiteral(shape, fillValue) {
    if (!Array.isArray(shape) || !shape.length) {
      return fillValue;
    }
    return Array.from({ length: shape[0] }, () =>
      buildFilledTensorLiteral(shape.slice(1), fillValue)
    );
  }

  function buildDefaultTensorLiteralValues(tensor, tensorShape) {
    if (tensor?.tensor_data?.mode === "literal" && tensor.tensor_data.values !== undefined) {
      return cloneTensorLiteral(tensor.tensor_data.values);
    }
    if (tensor?.tensor_data?.mode === "ones") {
      return buildFilledTensorLiteral(tensorShape, 1);
    }
    if (
      tensor?.tensor_data?.mode === "fill" &&
      typeof tensor.tensor_data.fill_value === "number" &&
      Number.isFinite(tensor.tensor_data.fill_value)
    ) {
      return buildFilledTensorLiteral(tensorShape, tensor.tensor_data.fill_value);
    }
    return buildFilledTensorLiteral(tensorShape, 0);
  }

  function analyzeTensorDataFillInput(rawValue) {
    const trimmedValue = String(rawValue || "").trim();
    if (!trimmedValue) {
      return {
        ok: false,
        message: "Fill value must be a finite number.",
      };
    }
    const parsedValue = Number(trimmedValue);
    if (!Number.isFinite(parsedValue)) {
      return {
        ok: false,
        message: "Fill value must be a finite number.",
      };
    }
    return {
      ok: true,
      tensorData: {
        mode: "fill",
        fill_value: parsedValue,
      },
    };
  }

  function analyzeTensorLiteralInput(rawValue, expectedShape) {
    let parsedValue = null;
    try {
      parsedValue = JSON.parse(String(rawValue || ""));
    } catch (error) {
      return {
        ok: false,
        message: `Explicit values must be valid JSON: ${error.message}`,
      };
    }

    const normalizedLiteral = normalizeTensorLiteralNode(parsedValue);
    if (!normalizedLiteral) {
      return {
        ok: false,
        message: "Explicit values must be finite numbers arranged as a non-ragged tensor.",
      };
    }
    if (!tensorShapesMatch(normalizedLiteral.shape, expectedShape)) {
      return {
        ok: false,
        message: `Explicit values must match the tensor shape ${formatTensorShape(
          expectedShape
        )}.`,
      };
    }
    return {
      ok: true,
      tensorData: {
        mode: "literal",
        values: normalizedLiteral.normalized,
      },
    };
  }

  function describeTensorData(tensor, tensorShape) {
    const tensorDataMode = getTensorDataMode(tensor);
    if (tensorDataMode === "ones") {
      return `Current initializer: ones for ${formatTensorShape(tensorShape)}.`;
    }
    if (tensorDataMode === "fill") {
      return `Current initializer: fill with ${tensor.tensor_data.fill_value}.`;
    }
    if (tensorDataMode === "literal") {
      return `Current initializer: explicit JSON values for ${formatTensorShape(
        tensorShape
      )}.`;
    }
    return `Current initializer: generated zeros for ${formatTensorShape(tensorShape)}.`;
  }

  return {
    getTensorShape,
    formatTensorShape,
    tensorShapesMatch,
    normalizeTensorLiteralNode,
    getTensorDataMode,
    cloneTensorLiteral,
    buildFilledTensorLiteral,
    buildDefaultTensorLiteralValues,
    analyzeTensorDataFillInput,
    analyzeTensorLiteralInput,
    describeTensorData,
  };
}
