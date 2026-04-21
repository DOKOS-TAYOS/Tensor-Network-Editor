export function createUtilityUiGeneratedCodeSupport({
  state,
  dom,
  setTooltipDescription,
  syncToolbarTransientUi,
  highlightCodeElement,
}) {
  const {
    generatedCode,
    generatedCodeView,
    generatedCodeModalView,
    copyCodeButton,
    expandGeneratedCodeButton,
    generatedCodeModal,
    generatedCodeModalCloseButton,
  } = dom;

  function syncGeneratedCodeActionState(
    renderedCode =
      generatedCode && typeof generatedCode.value === "string"
        ? generatedCode.value
        : state.generatedCode
  ) {
    const hasGeneratedCode =
      typeof renderedCode === "string" && Boolean(renderedCode.trim());
    setTooltipDescription(
      copyCodeButton,
      hasGeneratedCode
        ? "Copy generated code without import lines."
        : "Generate code first."
    );
    setTooltipDescription(
      expandGeneratedCodeButton,
      hasGeneratedCode
        ? "Open the generated code in a larger modal preview."
        : "Generate code first."
    );
    if (copyCodeButton) {
      copyCodeButton.disabled = !hasGeneratedCode;
    }
    if (expandGeneratedCodeButton) {
      expandGeneratedCodeButton.disabled = !hasGeneratedCode;
    }
    if (!hasGeneratedCode) {
      state.isGeneratedCodeModalOpen = false;
    }
    return hasGeneratedCode;
  }

  function syncGeneratedCodeModalState() {
    if (generatedCodeModal?.classList) {
      generatedCodeModal.classList.toggle("is-hidden", !state.isGeneratedCodeModalOpen);
    }
    if (generatedCodeModal) {
      generatedCodeModal.hidden = !state.isGeneratedCodeModalOpen;
    }
    if (
      state.isGeneratedCodeModalOpen &&
      generatedCodeModalCloseButton &&
      typeof generatedCodeModalCloseButton.focus === "function"
    ) {
      generatedCodeModalCloseButton.focus();
    }
  }

  function toggleGeneratedCodeModal(forceOpen) {
    const hasGeneratedCode = syncGeneratedCodeActionState();
    const nextOpen =
      typeof forceOpen === "boolean" ? forceOpen : !state.isGeneratedCodeModalOpen;
    state.isGeneratedCodeModalOpen = hasGeneratedCode ? nextOpen : false;
    state.openToolbarMenu = null;
    state.isTemplateSettingsOpen = false;
    state.isReflowLayoutOpen = false;
    syncToolbarTransientUi();
    syncGeneratedCodeModalState();
    return state.isGeneratedCodeModalOpen;
  }

  function renderGeneratedCodePreview(code = state.generatedCode) {
    const renderedCode = typeof code === "string" ? code : "";
    if (generatedCode) {
      generatedCode.value = renderedCode;
    }
    if (generatedCodeView) {
      generatedCodeView.textContent = renderedCode;
      if (typeof highlightCodeElement === "function") {
        void highlightCodeElement(generatedCodeView);
      }
    }
    if (generatedCodeModalView) {
      generatedCodeModalView.textContent = renderedCode;
      if (typeof highlightCodeElement === "function") {
        void highlightCodeElement(generatedCodeModalView);
      }
    }
    syncGeneratedCodeActionState(renderedCode);
    syncGeneratedCodeModalState();
  }

  return {
    syncGeneratedCodeActionState,
    syncGeneratedCodeModalState,
    toggleGeneratedCodeModal,
    renderGeneratedCodePreview,
  };
}
