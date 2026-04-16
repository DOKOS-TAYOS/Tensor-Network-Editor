const DEFAULT_AUTOSAVE_DELAY_MS = 300;

export function createPropertyAutosaveBindings({
  windowRef,
  delayMs = DEFAULT_AUTOSAVE_DELAY_MS,
}) {
  const autosaveTimers = new Map();

  function clearAutosaveTimer(fieldKey) {
    const timerId = autosaveTimers.get(fieldKey);
    if (typeof timerId === "number") {
      windowRef.clearTimeout(timerId);
    }
    autosaveTimers.delete(fieldKey);
  }

  function commitAutosave(fieldKey, commit) {
    clearAutosaveTimer(fieldKey);
    commit();
  }

  function scheduleAutosave(fieldKey, commit) {
    clearAutosaveTimer(fieldKey);
    autosaveTimers.set(
      fieldKey,
      windowRef.setTimeout(() => {
        autosaveTimers.delete(fieldKey);
        commit();
      }, delayMs)
    );
  }

  function bindDebouncedAutosave(element, fieldKey, commit, options = {}) {
    if (!element) {
      return;
    }
    element.dataset.focusKey = fieldKey;
    element.addEventListener("input", () => {
      scheduleAutosave(fieldKey, commit);
    });
    element.addEventListener("blur", () => {
      commitAutosave(fieldKey, commit);
    });
    if (options.commitOnEnter !== false) {
      element.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" || event.shiftKey) {
          return;
        }
        event.preventDefault();
        commitAutosave(fieldKey, commit);
      });
    }
  }

  function bindImmediateAutosave(
    element,
    fieldKey,
    commit,
    eventName = "change"
  ) {
    if (!element) {
      return;
    }
    if (fieldKey) {
      element.dataset.focusKey = fieldKey;
    }
    element.addEventListener(eventName, () => {
      commit();
    });
  }

  return {
    clearAutosaveTimer,
    commitAutosave,
    scheduleAutosave,
    bindDebouncedAutosave,
    bindImmediateAutosave,
  };
}
