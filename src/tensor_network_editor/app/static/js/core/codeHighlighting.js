function getLoadedPrism(windowRef) {
  if (!windowRef || typeof windowRef !== "object") {
    return null;
  }
  const prism = windowRef.Prism;
  return prism && typeof prism.highlightElement === "function" ? prism : null;
}

function buildVersionedAssetPath(windowRef, assetPath) {
  const assetVersion =
    windowRef &&
    typeof windowRef === "object" &&
    typeof windowRef.__TNE_ASSET_VERSION__ === "string"
      ? windowRef.__TNE_ASSET_VERSION__
      : "";
  return assetVersion ? `${assetPath}?v=${assetVersion}` : assetPath;
}

function appendScript(documentRef, sourcePath) {
  if (
    !documentRef ||
    typeof documentRef.createElement !== "function" ||
    !documentRef.head ||
    typeof documentRef.head.appendChild !== "function"
  ) {
    return Promise.resolve(false);
  }
  return new Promise((resolve, reject) => {
    const script = documentRef.createElement("script");
    script.async = false;
    script.src = sourcePath;
    script.onload = () => resolve(true);
    script.onerror = () =>
      reject(new Error(`Could not load syntax highlighter asset '${sourcePath}'.`));
    documentRef.head.appendChild(script);
  });
}

export function createCodeHighlightingSupport({ windowRef, documentRef }) {
  async function loadPrismHighlighter() {
    const loadedPrism = getLoadedPrism(windowRef);
    if (loadedPrism) {
      return loadedPrism;
    }
    if (!windowRef || typeof windowRef !== "object") {
      return null;
    }
    if (windowRef.__tnePrismLoaderPromise) {
      return windowRef.__tnePrismLoaderPromise;
    }
    windowRef.__tnePrismLoaderPromise = (async () => {
      await appendScript(
        documentRef,
        buildVersionedAssetPath(windowRef, "/vendor/prism-core.min.js")
      );
      if (!windowRef.Prism || typeof windowRef.Prism !== "object") {
        windowRef.Prism = {};
      }
      windowRef.Prism.manual = true;
      await appendScript(
        documentRef,
        buildVersionedAssetPath(windowRef, "/vendor/prism-python.min.js")
      );
      return getLoadedPrism(windowRef);
    })().catch(() => null);
    return windowRef.__tnePrismLoaderPromise;
  }

  async function highlightElement(element) {
    if (!element) {
      return false;
    }
    const loadedPrism = getLoadedPrism(windowRef);
    if (loadedPrism) {
      loadedPrism.highlightElement(element);
      return true;
    }
    const prism = await loadPrismHighlighter();
    if (!prism) {
      return false;
    }
    prism.highlightElement(element);
    return true;
  }

  return {
    loadPrismHighlighter,
    highlightElement,
  };
}
