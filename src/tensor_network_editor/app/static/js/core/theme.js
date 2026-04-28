export const UI_THEME = Object.freeze({
  fontFamily:
    '"Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif',
  fontMono:
    '"Cascadia Code", "Consolas", "SFMono-Regular", "Roboto Mono", monospace',
});

export const EDITOR_THEME_NAMES = Object.freeze(["dark", "light", "contrast", "colorblind", "shiny"]);
export const DEFAULT_EDITOR_THEME_NAME = "dark";
export const EDITOR_THEME_STORAGE_KEY = "tensor-network-editor.theme";
export const EDITOR_THEME_LABELS = Object.freeze({
  dark: "Dark",
  light: "Light",
  contrast: "High contrast",
  colorblind: "Colorblind-friendly",
  shiny: "Shiny",
});

const EDITOR_THEMES = Object.freeze({
  dark: Object.freeze({
    colorScheme: "dark",
    graph: Object.freeze({
      canvasBackground: "#0b0d12",
      canvasBackgroundAlt: "#111723",
      selection: "#a78bfa",
      selectionFill: "rgba(167, 139, 250, 0.14)",
      selectionTextBackground: "#171c28",
      edgeLabelBackground: "#171c28",
      edgeLabelText: "#c6d3e6",
      pendingTensor: "#f19a89",
      pendingIndex: "#78d1f7",
      tensorFallback: "#151922",
      tensorBorderFallback: "#2f3340",
      edge: "#7e8aa3",
      indexConnected: "#7ed3cf",
      indexOpen: "#d7ae68",
      groupDefault: "#8f7cf7",
      noteDefault: "#a286ff",
      emptyStateText: "#8f9bb1",
    }),
  }),
  light: Object.freeze({
    colorScheme: "light",
    graph: Object.freeze({
      canvasBackground: "#f6f8fc",
      canvasBackgroundAlt: "#e9edf6",
      selection: "#4f46e5",
      selectionFill: "rgba(79, 70, 229, 0.16)",
      selectionTextBackground: "#eef2ff",
      edgeLabelBackground: "#ffffff",
      edgeLabelText: "#172033",
      pendingTensor: "#c2410c",
      pendingIndex: "#0369a1",
      tensorFallback: "#ffffff",
      tensorBorderFallback: "#000000",
      edge: "#64748b",
      indexConnected: "#0f766e",
      indexOpen: "#b45309",
      groupDefault: "#6d28d9",
      noteDefault: "#7c3aed",
      emptyStateText: "#64748b",
    }),
  }),
  contrast: Object.freeze({
    colorScheme: "dark",
    graph: Object.freeze({
      canvasBackground: "#000000",
      canvasBackgroundAlt: "#101010",
      selection: "#ffff00",
      selectionFill: "rgba(255, 255, 0, 0.2)",
      selectionTextBackground: "#000000",
      edgeLabelBackground: "#000000",
      edgeLabelText: "#ffffff",
      pendingTensor: "#ff5f5f",
      pendingIndex: "#00ffff",
      tensorFallback: "#050505",
      tensorBorderFallback: "#1f1f1f",
      edge: "#ffffff",
      indexConnected: "#00ffff",
      indexOpen: "#ffff00",
      groupDefault: "#ff00ff",
      noteDefault: "#ffff00",
      emptyStateText: "#ffffff",
    }),
  }),
  colorblind: Object.freeze({
    colorScheme: "light",
    graph: Object.freeze({
      canvasBackground: "#f7f7f2",
      canvasBackgroundAlt: "#e8e6dc",
      selection: "#0072b2",
      selectionFill: "rgba(0, 114, 178, 0.16)",
      selectionTextBackground: "#e3f2fb",
      edgeLabelBackground: "#ffffff",
      edgeLabelText: "#202124",
      pendingTensor: "#d55e00",
      pendingIndex: "#009e73",
      tensorFallback: "#ffffff",
      tensorBorderFallback: "#000000",
      edge: "#5b5b5b",
      indexConnected: "#009e73",
      indexOpen: "#e69f00",
      groupDefault: "#cc79a7",
      noteDefault: "#0072b2",
      emptyStateText: "#5b5b5b",
    }),
  }),
  shiny: Object.freeze({
    colorScheme: "dark",
    graph: Object.freeze({
      canvasBackground: "#070915",
      canvasBackgroundAlt: "#11152c",
      selection: "#22d3ee",
      selectionFill: "rgba(34, 211, 238, 0.16)",
      selectionTextBackground: "#071824",
      edgeLabelBackground: "#071824",
      edgeLabelText: "#c4b5fd",
      pendingTensor: "#fb7185",
      pendingIndex: "#5eead4",
      tensorFallback: "#121631",
      tensorBorderFallback: "#2c304b",
      edge: "#94a3b8",
      indexConnected: "#34d399",
      indexOpen: "#facc15",
      groupDefault: "#e879f9",
      noteDefault: "#38bdf8",
      emptyStateText: "#a5b4fc",
    }),
  }),
});

export const GRAPH_THEME = { ...EDITOR_THEMES[DEFAULT_EDITOR_THEME_NAME].graph };

function resolveStoredEditorThemeName(value) {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  const normalizedName = value.trim().toLowerCase();
  return EDITOR_THEME_NAMES.includes(normalizedName) ? normalizedName : null;
}

export function normalizeEditorThemeName(themeName) {
  if (typeof themeName !== "string" || !themeName.trim()) {
    return DEFAULT_EDITOR_THEME_NAME;
  }
  const normalizedName = themeName.trim().toLowerCase();
  return EDITOR_THEME_NAMES.includes(normalizedName)
    ? normalizedName
    : DEFAULT_EDITOR_THEME_NAME;
}

export function formatEditorThemeLabel(themeName) {
  const normalizedName = normalizeEditorThemeName(themeName);
  return EDITOR_THEME_LABELS[normalizedName] || EDITOR_THEME_LABELS[DEFAULT_EDITOR_THEME_NAME];
}

export function readStoredEditorThemeName({ storageRef = null } = {}) {
  if (!storageRef || typeof storageRef.getItem !== "function") {
    return null;
  }
  try {
    return resolveStoredEditorThemeName(
      storageRef.getItem(EDITOR_THEME_STORAGE_KEY)
    );
  } catch {
    return null;
  }
}

export function persistEditorThemeName(themeName, { storageRef = null } = {}) {
  const normalizedName = normalizeEditorThemeName(themeName);
  if (!storageRef || typeof storageRef.setItem !== "function") {
    return normalizedName;
  }
  try {
    storageRef.setItem(EDITOR_THEME_STORAGE_KEY, normalizedName);
  } catch {
    return normalizedName;
  }
  return normalizedName;
}

export function resolvePreferredEditorThemeName({
  bootstrapThemeName = null,
  storageRef = null,
} = {}) {
  const storedThemeName = readStoredEditorThemeName({ storageRef });
  if (storedThemeName) {
    return storedThemeName;
  }
  return normalizeEditorThemeName(bootstrapThemeName);
}

export function applyEditorTheme(
  themeName,
  { documentRef = null, storageRef = null, persist = false } = {}
) {
  const normalizedName = normalizeEditorThemeName(themeName);
  const theme = EDITOR_THEMES[normalizedName];
  Object.assign(GRAPH_THEME, theme.graph);
  const root = documentRef?.documentElement;
  if (root) {
    root.dataset.theme = normalizedName;
    root.style.colorScheme = theme.colorScheme;
  }
  if (persist) {
    persistEditorThemeName(normalizedName, { storageRef });
  }
  return normalizedName;
}
