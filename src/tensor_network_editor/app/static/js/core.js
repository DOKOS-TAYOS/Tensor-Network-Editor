export const TENSOR_WIDTH = 180;
export const TENSOR_HEIGHT = 108;
export const INDEX_RADIUS = 15;
export const INDEX_PADDING = 8;
export const HISTORY_LIMIT = 100;
export const REDO_SHORTCUT_LABEL = "Ctrl+Shift+Z";
export const DEFAULT_INDEX_SLOTS = [
  { x: -58, y: -20 },
  { x: 58, y: -20 },
  { x: -58, y: 20 },
  { x: 58, y: 20 },
  { x: 0, y: -28 },
  { x: 0, y: 28 },
  { x: -24, y: -34 },
  { x: 24, y: 34 },
];

export const state = {
  spec: null,
  selectedEngine: null,
  selectedElement: null,
  selectionIds: [],
  primarySelectionId: null,
  generatedCode: "",
  connectMode: false,
  pendingIndexId: null,
  cy: null,
  hasFitCanvas: false,
  editorFinished: false,
  tensorOrder: [],
  syncingIndexPositions: false,
  syncingTensorPositions: false,
  undoStack: [],
  redoStack: [],
  lastMutationClearedCode: false,
  activeTensorDrag: null,
  activeIndexDrag: null,
  boxSelection: null,
  isHelpOpen: false,
  minimapDrag: null,
  minimapTransform: null,
};

export const refs = {
  statusMessage: document.getElementById("status-message"),
  propertiesPanel: document.getElementById("properties-panel"),
  generatedCode: document.getElementById("generated-code"),
  engineSelect: document.getElementById("engine-select"),
  connectButton: document.getElementById("connect-button"),
  loadInput: document.getElementById("load-input"),
  undoButton: document.getElementById("undo-button"),
  redoButton: document.getElementById("redo-button"),
  exportPngButton: document.getElementById("export-png-button"),
  exportSvgButton: document.getElementById("export-svg-button"),
  helpButton: document.getElementById("help-button"),
  helpModal: document.getElementById("help-modal"),
  helpBackdrop: document.getElementById("help-backdrop"),
  helpCloseButton: document.getElementById("help-close-button"),
  canvasShell: document.getElementById("canvas-shell"),
  selectionBox: document.getElementById("canvas-selection-box"),
  minimapCanvas: document.getElementById("minimap"),
};

export const editor = {
  constants: {
    TENSOR_WIDTH,
    TENSOR_HEIGHT,
    INDEX_RADIUS,
    INDEX_PADDING,
    HISTORY_LIMIT,
    REDO_SHORTCUT_LABEL,
    DEFAULT_INDEX_SLOTS,
  },
  refs,
  state,
};
