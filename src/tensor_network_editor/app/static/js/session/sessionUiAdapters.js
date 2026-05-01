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
  rasterizeSvgToPng = null,
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
  const resolvePywebviewApi = () =>
    windowRef &&
    windowRef.pywebview &&
    windowRef.pywebview.api &&
    typeof windowRef.pywebview.api.save_text_file === "function" &&
    typeof windowRef.pywebview.api.save_binary_file === "function"
      ? windowRef.pywebview.api
      : null;
  const encodeBytesToBase64 = (bytes) => {
    if (
      typeof Buffer !== "undefined" &&
      typeof Buffer.from === "function"
    ) {
      return Buffer.from(bytes).toString("base64");
    }
    if (typeof globalThis.btoa !== "function") {
      throw new Error("Binary downloads are not available in this browser.");
    }
    let binaryText = "";
    for (const byte of bytes) {
      binaryText += String.fromCharCode(byte);
    }
    return globalThis.btoa(binaryText);
  };
  const resolvedDownloadBlob =
    typeof downloadBlob === "function"
      ? downloadBlob
      : async (filename, blobLike) => {
          const resolvedPywebviewApi = resolvePywebviewApi();
          if (resolvedPywebviewApi) {
            if (!blobLike || typeof blobLike.arrayBuffer !== "function") {
              throw new Error("Binary downloads are not available in this browser.");
            }
            const buffer = await blobLike.arrayBuffer();
            const bytes = new Uint8Array(buffer);
            return resolvedPywebviewApi.save_binary_file(
              filename,
              encodeBytesToBase64(bytes),
              blobLike.type || "application/octet-stream"
            );
          }
          if (!documentRef || !urlRef || typeof urlRef.createObjectURL !== "function") {
            throw new Error("File downloads are not available in this browser.");
          }
          const anchor = documentRef.createElement("a");
          anchor.href = urlRef.createObjectURL(blobLike);
          anchor.download = filename;
          anchor.click();
          urlRef.revokeObjectURL(anchor.href);
          return true;
        };
  const resolvedDownloadText =
    typeof downloadText === "function"
      ? downloadText
      : async (filename, text, contentType = "text/plain;charset=utf-8") => {
          const resolvedPywebviewApi = resolvePywebviewApi();
          if (resolvedPywebviewApi) {
            return resolvedPywebviewApi.save_text_file(filename, text, contentType);
          }
          if (typeof blobCtor !== "function") {
            throw new Error("Text downloads are not available in this browser.");
          }
          return resolvedDownloadBlob(
            filename,
            new blobCtor([text], { type: contentType })
          );
        };
  const resolvedRasterizeSvgToPng =
    typeof rasterizeSvgToPng === "function"
      ? rasterizeSvgToPng
      : async ({
          svgText,
          sourceContentType = "image/svg+xml;charset=utf-8",
        }) => {
          const imageCtor =
            windowRef && typeof windowRef.Image === "function"
              ? windowRef.Image
              : typeof Image === "function"
                ? Image
                : null;
          if (
            !documentRef ||
            !urlRef ||
            typeof urlRef.createObjectURL !== "function" ||
            typeof urlRef.revokeObjectURL !== "function" ||
            typeof blobCtor !== "function" ||
            typeof imageCtor !== "function"
          ) {
            throw new Error("PNG fallback rendering is not available in this browser.");
          }
          const svgBlob = new blobCtor([svgText], { type: sourceContentType });
          const svgUrl = urlRef.createObjectURL(svgBlob);
          try {
            const image = await new Promise((resolve, reject) => {
              const nextImage = new imageCtor();
              nextImage.onload = () => resolve(nextImage);
              nextImage.onerror = () => {
                reject(
                  new Error(
                    "Could not load the SVG export for browser PNG conversion."
                  )
                );
              };
              nextImage.src = svgUrl;
            });
            const canvas = documentRef.createElement("canvas");
            if (!canvas || typeof canvas.getContext !== "function") {
              throw new Error("Canvas rendering is not available in this browser.");
            }
            const width = Math.max(
              1,
              Math.round(image.naturalWidth || image.width || 1)
            );
            const height = Math.max(
              1,
              Math.round(image.naturalHeight || image.height || 1)
            );
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext("2d");
            if (!context) {
              throw new Error("Canvas rendering is not available in this browser.");
            }
            context.drawImage(image, 0, 0, width, height);
            return await new Promise((resolve, reject) => {
              if (typeof canvas.toBlob !== "function") {
                reject(
                  new Error(
                    "Canvas PNG export is not available in this browser."
                  )
                );
                return;
              }
              canvas.toBlob((blob) => {
                if (blob) {
                  resolve(blob);
                  return;
                }
                reject(new Error("Could not convert the SVG export into PNG."));
              }, "image/png");
            });
          } finally {
            urlRef.revokeObjectURL(svgUrl);
          }
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
    rasterizeSvgToPng: resolvedRasterizeSvgToPng,
    requestFileText: resolvedRequestFileText,
    openFilePicker: resolvedOpenFilePicker,
    schedule: resolvedSchedule,
    closeWindow: resolvedCloseWindow,
  };
}
