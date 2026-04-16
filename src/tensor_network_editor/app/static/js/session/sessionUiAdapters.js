export function createSessionUiAdapters({
  windowRef = null,
  documentRef = null,
  urlRef = typeof URL !== "undefined" ? URL : null,
  blobCtor = typeof Blob !== "undefined" ? Blob : null,
  fileReaderCtor = typeof FileReader !== "undefined" ? FileReader : null,
  promptText = null,
  confirmAction = null,
  copyText = null,
  downloadText = null,
  downloadBlob = null,
  requestFileText = null,
  openFilePicker = null,
  schedule = null,
  closeWindow = null,
} = {}) {
  const resolvedPromptText =
    typeof promptText === "function"
      ? promptText
      : (message, defaultValue = "") =>
          windowRef && typeof windowRef.prompt === "function"
            ? windowRef.prompt(message, defaultValue)
            : defaultValue;
  const resolvedConfirmAction =
    typeof confirmAction === "function"
      ? confirmAction
      : (message) =>
          !windowRef || typeof windowRef.confirm !== "function"
            ? true
            : windowRef.confirm(message);
  const resolvedCopyText =
    typeof copyText === "function"
      ? copyText
      : async (text) => {
          const clipboard =
            windowRef &&
            windowRef.navigator &&
            windowRef.navigator.clipboard &&
            typeof windowRef.navigator.clipboard.writeText === "function"
              ? windowRef.navigator.clipboard
              : null;
          if (!clipboard) {
            throw new Error("Clipboard access is not available in this browser.");
          }
          await clipboard.writeText(text);
        };
  const resolvedDownloadBlob =
    typeof downloadBlob === "function"
      ? downloadBlob
      : (filename, blobLike) => {
          if (!documentRef || !urlRef || typeof urlRef.createObjectURL !== "function") {
            throw new Error("File downloads are not available in this browser.");
          }
          const anchor = documentRef.createElement("a");
          anchor.href = urlRef.createObjectURL(blobLike);
          anchor.download = filename;
          anchor.click();
          urlRef.revokeObjectURL(anchor.href);
        };
  const resolvedDownloadText =
    typeof downloadText === "function"
      ? downloadText
      : (filename, text, contentType = "text/plain;charset=utf-8") => {
          if (typeof blobCtor !== "function") {
            throw new Error("Text downloads are not available in this browser.");
          }
          resolvedDownloadBlob(filename, new blobCtor([text], { type: contentType }));
        };
  const resolvedRequestFileText =
    typeof requestFileText === "function"
      ? requestFileText
      : async (file, encoding = "utf-8") => {
          if (file && typeof file.text === "function") {
            return file.text();
          }
          if (typeof fileReaderCtor !== "function") {
            throw new Error("File loading is not available in this browser.");
          }
          return new Promise((resolve, reject) => {
            const reader = new fileReaderCtor();
            reader.onload = () => {
              resolve(typeof reader.result === "string" ? reader.result : "");
            };
            reader.onerror = () => {
              reject(reader.error || new Error("Could not read the selected file."));
            };
            reader.readAsText(file, encoding);
          });
        };
  const resolvedOpenFilePicker =
    typeof openFilePicker === "function"
      ? openFilePicker
      : (input) => {
          if (input && typeof input.click === "function") {
            input.click();
          }
        };
  const resolvedSchedule =
    typeof schedule === "function"
      ? schedule
      : (callback, delayMs) =>
          windowRef && typeof windowRef.setTimeout === "function"
            ? windowRef.setTimeout(callback, delayMs)
            : globalThis.setTimeout(callback, delayMs);
  const resolvedCloseWindow =
    typeof closeWindow === "function"
      ? closeWindow
      : () => {
          if (windowRef && typeof windowRef.close === "function") {
            windowRef.close();
          }
        };

  return {
    promptText: resolvedPromptText,
    confirmAction: resolvedConfirmAction,
    copyText: resolvedCopyText,
    downloadText: resolvedDownloadText,
    downloadBlob: resolvedDownloadBlob,
    requestFileText: resolvedRequestFileText,
    openFilePicker: resolvedOpenFilePicker,
    schedule: resolvedSchedule,
    closeWindow: resolvedCloseWindow,
  };
}
