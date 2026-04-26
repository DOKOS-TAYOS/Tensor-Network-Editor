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

  function isPortableComplexLiteral(value) {
    return (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      typeof value.real === "number" &&
      typeof value.imag === "number" &&
      Number.isFinite(value.real) &&
      Number.isFinite(value.imag)
    );
  }

  function normalizeTensorScalarNode(value) {
    if (typeof value === "boolean") {
      return null;
    }
    if (typeof value === "number") {
      return Number.isFinite(value) ? value : null;
    }
    if (typeof value === "string") {
      const parsedValue = parseTensorScalarInput(value);
      return parsedValue.ok ? parsedValue.value : null;
    }
    if (isPortableComplexLiteral(value)) {
      return {
        real: value.real,
        imag: value.imag,
      };
    }
    return null;
  }

  function normalizeTensorLiteralNode(value) {
    const scalarValue = normalizeTensorScalarNode(value);
    if (scalarValue !== null) {
      return {
        normalized: scalarValue,
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
    if (
      [
        "zeros",
        "ones",
        "fill",
        "literal",
        "identity",
        "copy",
        "random",
      ].includes(mode)
    ) {
      return mode;
    }
    return "zeros";
  }

  function getTensorDataDType(tensor) {
    const dtype = String(tensor?.tensor_data?.dtype || "").trim();
    return ["float32", "float64", "complex64", "complex128"].includes(dtype)
      ? dtype
      : "";
  }

  function getTensorRandomDistribution(tensor) {
    const distribution = String(tensor?.tensor_data?.distribution || "normal").trim();
    return distribution === "uniform" ? "uniform" : "normal";
  }

  function getTensorRandomSeed(tensor) {
    const seed = tensor?.tensor_data?.seed;
    return Number.isInteger(seed) && seed >= 0 ? seed : 0;
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
      normalizeTensorScalarNode(tensor.tensor_data.fill_value) !== null
    ) {
      return buildFilledTensorLiteral(tensorShape, tensor.tensor_data.fill_value);
    }
    return buildFilledTensorLiteral(tensorShape, 0);
  }

  function formatTensorScalarLiteral(value) {
    if (isPortableComplexLiteral(value)) {
      const sign = value.imag < 0 ? "" : "+";
      return `${value.real}${sign}${value.imag}j`;
    }
    return String(value);
  }

  function parseTensorScalarInput(rawValue) {
    const trimmedValue = String(rawValue ?? "").trim();
    if (!trimmedValue) {
      return {
        ok: false,
        message: "Value must be a finite real or complex number.",
      };
    }
    const realValue = Number(trimmedValue);
    if (Number.isFinite(realValue)) {
      return {
        ok: true,
        value: realValue,
      };
    }
    try {
      const parsedJson = JSON.parse(trimmedValue);
      const normalizedJson = normalizeTensorScalarNode(parsedJson);
      if (normalizedJson !== null) {
        return {
          ok: true,
          value: normalizedJson,
        };
      }
    } catch {
      // Friendly complex forms such as 1+2j are handled below.
    }
    const complexValue = parseFriendlyComplexInput(trimmedValue);
    if (complexValue) {
      return {
        ok: true,
        value: complexValue,
      };
    }
    return {
      ok: false,
      message: "Value must be a finite real number or a complex value like 1+2j.",
    };
  }

  function parseFriendlyComplexInput(rawValue) {
    const normalizedValue = String(rawValue ?? "")
      .replace(/\s+/gu, "")
      .replace(/i$/u, "j");
    if (!normalizedValue.endsWith("j")) {
      return null;
    }
    const body = normalizedValue.slice(0, -1);
    if (body === "" || body === "+") {
      return { real: 0, imag: 1 };
    }
    if (body === "-") {
      return { real: 0, imag: -1 };
    }
    const splitIndex = findComplexSplitIndex(body);
    const realText = splitIndex > 0 ? body.slice(0, splitIndex) : "0";
    let imagText = splitIndex > 0 ? body.slice(splitIndex) : body;
    if (imagText === "+") {
      imagText = "1";
    } else if (imagText === "-") {
      imagText = "-1";
    }
    const real = Number(realText);
    const imag = Number(imagText);
    if (!Number.isFinite(real) || !Number.isFinite(imag)) {
      return null;
    }
    return { real, imag };
  }

  function findComplexSplitIndex(value) {
    for (let index = value.length - 1; index > 0; index -= 1) {
      const character = value[index];
      const previousCharacter = value[index - 1];
      if (
        (character === "+" || character === "-") &&
        previousCharacter !== "e" &&
        previousCharacter !== "E"
      ) {
        return index;
      }
    }
    return -1;
  }

  function analyzeTensorDataFillInput(rawValue) {
    const parsedValue = parseTensorScalarInput(rawValue);
    if (!parsedValue.ok) {
      return parsedValue;
    }
    return {
      ok: true,
      tensorData: {
        mode: "fill",
        fill_value: parsedValue.value,
      },
    };
  }

  function analyzeTensorLiteralInput(rawValue, expectedShape) {
    let parsedValue = null;
    try {
      parsedValue = JSON.parse(String(rawValue ?? ""));
    } catch (error) {
      return {
        ok: false,
        message: `Explicit values must be valid JSON: ${error.message}`,
      };
    }
    const normalizedScalar = normalizeTensorScalarNode(parsedValue);
    if (normalizedScalar !== null) {
      parsedValue = normalizedScalar;
    }

    const normalizedLiteral = normalizeTensorLiteralNode(parsedValue);
    if (!normalizedLiteral) {
      return {
        ok: false,
        message:
          "Explicit values must be finite numbers or complex values arranged as a non-ragged tensor.",
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
    if (tensorDataMode === "identity") {
      return `Current initializer: identity matrix for ${formatTensorShape(tensorShape)}.`;
    }
    if (tensorDataMode === "copy") {
      return `Current initializer: copy/delta tensor for ${formatTensorShape(tensorShape)}.`;
    }
    if (tensorDataMode === "random") {
      return `Current initializer: seeded random values for ${formatTensorShape(tensorShape)}.`;
    }
    if (tensorDataMode === "fill") {
      return `Current initializer: fill with ${formatTensorScalarLiteral(
        tensor.tensor_data.fill_value
      )}.`;
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
    getTensorDataDType,
    getTensorRandomDistribution,
    getTensorRandomSeed,
    cloneTensorLiteral,
    buildFilledTensorLiteral,
    buildDefaultTensorLiteralValues,
    formatTensorScalarLiteral,
    parseTensorScalarInput,
    analyzeTensorDataFillInput,
    analyzeTensorLiteralInput,
    describeTensorData,
  };
}
