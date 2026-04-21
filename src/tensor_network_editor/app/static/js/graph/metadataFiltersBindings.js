import {
  collectFilterOptionsForScope,
  normalizeText,
} from "./metadataFiltersState.js";

export function createMetadataFilterBindingsSupport({ document }) {
  function bindListener(target, eventName, handler) {
    if (!target || typeof target.addEventListener !== "function") {
      return;
    }
    target.addEventListener(eventName, handler);
  }

  function readSelectChevronExpanded(fieldElement) {
    if (!fieldElement) {
      return false;
    }
    if (typeof fieldElement.getAttribute === "function") {
      return fieldElement.getAttribute("data-expanded") === "true";
    }
    return fieldElement.attributes?.["data-expanded"] === "true";
  }

  function setSelectChevronExpanded(
    fieldElement,
    isExpanded,
    selectElement = null
  ) {
    if (!fieldElement || typeof fieldElement.setAttribute !== "function") {
      if (selectElement && selectElement.dataset) {
        selectElement.dataset.expanded = String(Boolean(isExpanded));
      }
      return;
    }
    fieldElement.setAttribute("data-expanded", String(Boolean(isExpanded)));
    if (selectElement && selectElement.dataset) {
      selectElement.dataset.expanded = String(Boolean(isExpanded));
    }
  }

  function bindSelectChevronDisclosure(selectElement, fieldElement) {
    if (!selectElement) {
      return;
    }
    setSelectChevronExpanded(fieldElement, false, selectElement);
    bindListener(selectElement, "mousedown", () => {
      setSelectChevronExpanded(
        fieldElement,
        !readSelectChevronExpanded(fieldElement),
        selectElement
      );
    });
    bindListener(selectElement, "keydown", (event) => {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        setSelectChevronExpanded(fieldElement, true, selectElement);
      }
      if (["Escape", "Tab"].includes(event.key)) {
        setSelectChevronExpanded(fieldElement, false, selectElement);
      }
    });
    bindListener(selectElement, "change", () => {
      setSelectChevronExpanded(fieldElement, false, selectElement);
    });
    bindListener(selectElement, "blur", () => {
      setSelectChevronExpanded(fieldElement, false, selectElement);
    });
  }

  function bindMetadataFilterControls({
    state,
    shouldFocusSearchInput,
    setShouldFocusSearchInput,
    renderMetadataFilters,
    requestHighlightRender,
    updateMetadataFilters,
    updateNameSearch,
    normalizeNameSearch,
    normalizeMetadataFilters,
    collectTagsForScope,
    getCheckboxIdForTag,
  }) {
    const filters = normalizeMetadataFilters();
    const search = normalizeNameSearch();
    const filterButton = document.getElementById("canvas-metadata-filter-button");
    const searchButton = document.getElementById("canvas-name-search-button");
    const filterScopeSelect = document.getElementById(
      "canvas-metadata-filter-scope-select"
    );
    const filterSelectAllButton = document.getElementById(
      "canvas-metadata-filter-select-all-button"
    );
    const filterScopeField = document.getElementById(
      "canvas-metadata-filter-scope-field"
    );
    const filterClearButton = document.getElementById(
      "canvas-metadata-filter-clear-button"
    );
    const filterSelectNoneButton = document.getElementById(
      "canvas-metadata-filter-select-none-button"
    );
    const searchScopeSelect = document.getElementById(
      "canvas-name-search-scope-select"
    );
    const searchScopeField = document.getElementById(
      "canvas-name-search-scope-field"
    );
    const searchInput = document.getElementById("canvas-name-search-input");

    bindSelectChevronDisclosure(filterScopeSelect, filterScopeField);
    bindSelectChevronDisclosure(searchScopeSelect, searchScopeField);

    if (filterButton) {
      filterButton.addEventListener("click", () => {
        const nextPopover =
          state.openCanvasToolPopover === "filter" ? null : "filter";
        setShouldFocusSearchInput(false);
        state.openCanvasToolPopover = nextPopover;
        state.nameSearch = {
          ...normalizeNameSearch(),
          enabled: false,
        };
        renderMetadataFilters();
        requestHighlightRender();
      });
    }

    if (searchButton) {
      searchButton.addEventListener("click", () => {
        const nextPopover =
          state.openCanvasToolPopover === "search" ? null : "search";
        setShouldFocusSearchInput(nextPopover === "search");
        state.openCanvasToolPopover = nextPopover;
        state.metadataFilters = {
          ...normalizeMetadataFilters(),
          enabled: false,
        };
        renderMetadataFilters();
        requestHighlightRender();
      });
    }

    if (filterScopeSelect) {
      filterScopeSelect.addEventListener("change", () => {
        const nextScope = filterScopeSelect.value;
        const nextSelectedTags = normalizeMetadataFilters().selectedTags.filter((tag) =>
          tag === "__not_specified__" ||
          collectTagsForScope(nextScope).some(
            (candidate) => candidate.toLowerCase() === tag.toLowerCase()
          )
        );
        updateMetadataFilters({
          enabled: true,
          scope: nextScope,
          selectedTags: nextSelectedTags,
        });
      });
    }

    if (filterSelectAllButton) {
      filterSelectAllButton.addEventListener("click", () => {
        updateMetadataFilters({
          enabled: true,
          selectedTags: collectFilterOptionsForScope(
            collectTagsForScope,
            filters.scope
          ),
        });
      });
    }

    if (filterClearButton) {
      filterClearButton.addEventListener("click", () => {
        updateMetadataFilters(
          {
            enabled: false,
            selectedTags: [],
          },
          { openPopover: "filter" }
        );
      });
    }

    if (filterSelectNoneButton) {
      filterSelectNoneButton.addEventListener("click", () => {
        updateMetadataFilters({
          enabled: true,
          selectedTags: [],
        });
      });
    }

    collectFilterOptionsForScope(collectTagsForScope, filters.scope).forEach((tag) => {
      const checkbox = document.getElementById(getCheckboxIdForTag(tag));
      if (!checkbox) {
        return;
      }
      checkbox.addEventListener("change", () => {
        const nextSelectedTags = collectFilterOptionsForScope(
          collectTagsForScope,
          filters.scope
        ).filter((candidate) => {
          const checkboxElement = document.getElementById(
            getCheckboxIdForTag(candidate)
          );
          return checkboxElement && checkboxElement.checked;
        });
        updateMetadataFilters({
          enabled: true,
          selectedTags: nextSelectedTags,
        });
      });
    });

    if (searchScopeSelect) {
      searchScopeSelect.addEventListener("change", () => {
        updateNameSearch({
          enabled: Boolean(normalizeText(searchInput ? searchInput.value : search.query)),
          scope: searchScopeSelect.value,
        });
      });
    }

    if (searchInput) {
      searchInput.addEventListener("input", () => {
        updateNameSearch(
          {
            enabled: Boolean(normalizeText(searchInput.value)),
            query: searchInput.value,
          },
          { renderPanel: false }
        );
      });
      if (
        shouldFocusSearchInput &&
        state.openCanvasToolPopover === "search" &&
        typeof searchInput.focus === "function"
      ) {
        searchInput.focus();
      }
    }
    setShouldFocusSearchInput(false);
  }

  return {
    bindMetadataFilterControls,
  };
}
