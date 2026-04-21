import { GRAPH_THEME } from "../core/theme.js";
import { createNotesClipboardActions } from "./notesClipboard.js";
import { createNotesSupport } from "./notesSupport.js";

export function registerNotesFeature(ctx) {
  const state = ctx.state;
  const { addNoteButton, notesLayer } = ctx.dom;
  const notesSupport = createNotesSupport({
    ctx,
    state,
    constants: ctx.constants,
  });
  const notesClipboardActions = createNotesClipboardActions({ ctx, state });
  const {
    addNoteAtCenter,
    applyCanvasSelectionDragDelta,
    buildCanvasSelectionDragState,
    createNote,
    findNoteById,
    finishActiveNoteDrag,
    finishActiveNoteResize,
    formatNoteColorAlpha,
    getCanvasZoom,
    getRenderableNoteSize,
    noteCanvasBounds,
    noteInvalidation,
    removeNote,
    selectNoteIfNeeded,
    startNoteDrag,
    startNoteResize,
    toggleNoteCollapse,
    updateActiveNoteDrag,
    updateActiveNoteResize,
  } = notesSupport;
  const { copySelectedSubgraphToClipboard, pasteClipboardToCanvas } =
    notesClipboardActions;

  function renderNotes() {
    if (!notesLayer) {
      return;
    }
    notesLayer.innerHTML = "";
    const noteZoom = getCanvasZoom();
    state.spec.notes.forEach((note) => {
      const isCollapsed = Boolean(note.metadata && note.metadata.collapsed);
      const bounds = noteCanvasBounds(note);
      const noteSize = getRenderableNoteSize(note);
      const noteElement = document.createElement("article");
      noteElement.className = "canvas-note";
      noteElement.dataset.noteId = note.id;
      noteElement.style.left = `${bounds.x1}px`;
      noteElement.style.top = `${bounds.y1}px`;
      noteElement.style.width = `${bounds.width}px`;
      noteElement.style.height = `${bounds.height}px`;
      const frame = document.createElement("div");
      frame.className = "canvas-note-frame";
      if (state.selectionIds.includes(note.id)) {
        frame.classList.add("is-selected");
      }
      if (isCollapsed) {
        frame.classList.add("is-collapsed");
      }
      frame.style.width = `${noteSize.width}px`;
      frame.style.height = `${noteSize.height}px`;
      frame.style.transform = `scale(${noteZoom})`;
      frame.style.transformOrigin = "top left";
      const noteColor = ctx.getMetadataColor(note.metadata, GRAPH_THEME.noteDefault);
      frame.style.borderColor = noteColor;
      frame.style.setProperty("--note-accent-color", noteColor);
      frame.style.setProperty(
        "--note-surface-color",
        formatNoteColorAlpha(ctx.shiftColor(noteColor, -18), 0.96)
      );
      frame.style.setProperty(
        "--note-surface-color-strong",
        formatNoteColorAlpha(ctx.shiftColor(noteColor, -46), 0.98)
      );
      frame.style.setProperty(
        "--note-header-color",
        ctx.readableTextColor(ctx.shiftColor(noteColor, -6))
      );

      if (isCollapsed) {
        const collapsedToggle = createNoteCollapseButton(note);
        collapsedToggle.classList.add("canvas-note-collapsed-toggle");
        frame.appendChild(collapsedToggle);
        frame.addEventListener("mousedown", (event) => {
          if (event.target.closest(".toggle-note-collapse")) {
            return;
          }
          startNoteDrag(event, note.id);
        });
      } else {
        const header = document.createElement("div");
        header.className = "canvas-note-header";
        header.textContent = "Note";
        header.addEventListener("mousedown", (event) => startNoteDrag(event, note.id));
        header.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          selectNoteIfNeeded(note.id, {
            additive:
              typeof ctx.isAdditiveSelectionModifier === "function" &&
              ctx.isAdditiveSelectionModifier(event),
          });
        });

        const actions = document.createElement("div");
        actions.className = "canvas-note-actions";

        const { colorButton, colorInput } = createNoteColorControl(note);
        actions.appendChild(colorButton);
        actions.appendChild(colorInput);

        const collapseButton = createNoteCollapseButton(note);
        actions.appendChild(collapseButton);

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "canvas-note-delete danger";
        deleteButton.dataset.tooltipEnabled = "true";
        deleteButton.dataset.shortcutLabel = "Delete note";
        deleteButton.dataset.shortcutDescription = "Remove this note from the canvas.";
        deleteButton.setAttribute(
          "aria-label",
          "Delete note. Remove this note from the canvas."
        );
        deleteButton.removeAttribute("title");
        deleteButton.textContent = "×";
        deleteButton.innerHTML = `
          <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
            <path d="M6.5 1.5h3l.5 1H13A1.5 1.5 0 0 1 14.5 4v1h-13V4A1.5 1.5 0 0 1 3 2.5h3zM2.5 6h11l-.7 7.1A1.5 1.5 0 0 1 11.3 14.5H4.7a1.5 0 0 1-1.5-1.4zm3 1.3a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0zm3 0a.5.5 0 0 0-1 0v4.9a.5.5 0 0 0 1 0z"/>
          </svg>
        `;
        deleteButton.addEventListener("mousedown", (event) => {
          event.stopPropagation();
        });
        deleteButton.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          ctx.applyDesignChange(
            () => {
              removeNote(note.id);
            },
            {
              selectionIds: [],
              invalidate: noteInvalidation({ lookups: true }),
              statusMessage: "Deleted a canvas note.",
            }
          );
        });
        actions.appendChild(deleteButton);
        header.appendChild(actions);

        const textarea = document.createElement("textarea");
        textarea.className = "canvas-note-body";
        textarea.value = note.text;
        textarea.spellcheck = false;
        textarea.addEventListener("mousedown", (event) => {
          event.stopPropagation();
        });
        textarea.addEventListener("keydown", (event) => {
          event.stopPropagation();
        });
        textarea.addEventListener("click", (event) => {
          event.stopPropagation();
          selectNoteIfNeeded(note.id, {
            additive:
              typeof ctx.isAdditiveSelectionModifier === "function" &&
              ctx.isAdditiveSelectionModifier(event),
          });
        });
        textarea.addEventListener("focus", () => {
          if (state.selectionIds.length === 1 && state.selectionIds[0] === note.id) {
            return;
          }
          ctx.setSelection([note.id], { primaryId: note.id });
        });
        ctx.bindDebouncedAutosave(
          textarea,
          `note:${note.id}:canvas-text`,
          () => {
            const proposedText = textarea.value.trim();
            if (!proposedText) {
              textarea.value = note.text;
              ctx.setStatus("Notes cannot be empty.", "error");
              return;
            }
            if (proposedText === note.text) {
              return;
            }
            ctx.applyDesignChange(
              () => {
                note.text = proposedText;
              },
              {
                selectionIds: [note.id],
                primaryId: note.id,
                invalidate: noteInvalidation({ overlays: false }),
                statusMessage: "Updated the note text.",
              }
            );
          },
          { commitOnEnter: false, scheduleOnInput: false }
        );

        const resizeHandle = document.createElement("div");
        resizeHandle.className = "canvas-note-resize-handle";
        resizeHandle.addEventListener("mousedown", (event) => startNoteResize(event, note.id));

        frame.appendChild(header);
        frame.appendChild(textarea);
        frame.appendChild(resizeHandle);
      }
      noteElement.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectNoteIfNeeded(note.id, {
          additive:
            typeof ctx.isAdditiveSelectionModifier === "function" &&
            ctx.isAdditiveSelectionModifier(event),
        });
      });
      noteElement.appendChild(frame);
      notesLayer.appendChild(noteElement);
    });
  }

  function createNoteColorControl(note) {
    const colorButton = document.createElement("button");
    colorButton.type = "button";
    colorButton.className = "canvas-note-color-button";
    colorButton.dataset.tooltipEnabled = "true";
    colorButton.dataset.shortcutLabel = "Change note color";
    colorButton.dataset.shortcutDescription = "Choose a new color for this note.";
    colorButton.setAttribute(
      "aria-label",
      "Change note color. Choose a new color for this note."
    );
    colorButton.removeAttribute("title");
    colorButton.innerHTML = `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M11.6 1.5a1.9 1.9 0 0 1 2.7 2.7l-1 1-2.7-2.7zm-1.7 1.7L2.2 10.9a2.5 2.5 0 0 0-.6 1l-.7 2.5a.7.7 0 0 0 .9.9l2.5-.7a2.5 2.5 0 0 0 1-.6L13 6.2z"/>
      </svg>
    `;
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.className = "canvas-note-color-input";
    colorInput.value = ctx.getMetadataColor(note.metadata, GRAPH_THEME.noteDefault);
    colorInput.setAttribute("tabindex", "-1");
    colorInput.setAttribute("aria-hidden", "true");

    colorButton.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    colorButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (typeof colorInput.showPicker === "function") {
        colorInput.showPicker();
      } else {
        colorInput.click();
      }
    });
    ctx.bindImmediateAutosave(
      colorInput,
      `note:${note.id}:canvas-color`,
      () => {
        if (
          colorInput.value ===
          ctx.getMetadataColor(note.metadata, GRAPH_THEME.noteDefault)
        ) {
          return;
        }
        ctx.applyDesignChange(
          () => {
            note.metadata.color = colorInput.value;
          },
          {
            selectionIds: [note.id],
            primaryId: note.id,
            invalidate: noteInvalidation(),
            statusMessage: "Updated the note.",
          }
        );
      },
      "input"
    );
    return { colorButton, colorInput };
  }

  function createNoteCollapseButton(note) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "toggle-note-collapse";
    const isCollapsed = Boolean(note.metadata && note.metadata.collapsed);
    const tooltipLabel = isCollapsed ? "Expand note" : "Collapse note";
    const tooltipDescription = isCollapsed
      ? "Show the hidden note body."
      : "Hide this note body until you expand it again.";
    button.dataset.tooltipEnabled = "true";
    button.dataset.shortcutLabel = tooltipLabel;
    button.dataset.shortcutDescription = tooltipDescription;
    button.setAttribute(
      "aria-label",
      `${tooltipLabel}. ${tooltipDescription}`
    );
    button.removeAttribute("title");
    button.innerHTML = `
      <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M3 2.5h10A1.5 1.5 0 0 1 14.5 4v6A1.5 1.5 0 0 1 13 11.5H8.6L5 14v-2.5H3A1.5 1.5 0 0 1 1.5 10V4A1.5 1.5 0 0 1 3 2.5Zm1 3.25a.75.75 0 0 0 0 1.5h8a.75.75 0 0 0 0-1.5Zm0 2.75a.75.75 0 0 0 0 1.5h5.5a.75.75 0 0 0 0-1.5Z"/>
      </svg>
    `;
    button.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      toggleNoteCollapse(note.id);
    });
    return button;
  }

  Object.assign(ctx, {
    addNoteAtCenter,
    createNote,
    findNoteById,
    getRenderableNoteSize,
    noteCanvasBounds,
    buildCanvasSelectionDragState,
    applyCanvasSelectionDragDelta,
    removeNote,
    renderNotes,
    startNoteDrag,
    updateActiveNoteDrag,
    finishActiveNoteDrag,
    startNoteResize,
    updateActiveNoteResize,
    finishActiveNoteResize,
    toggleNoteCollapse,
    copySelectedSubgraphToClipboard,
    pasteClipboardToCanvas,
  });
}
