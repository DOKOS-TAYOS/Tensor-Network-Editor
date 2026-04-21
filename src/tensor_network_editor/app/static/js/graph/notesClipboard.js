export function createNotesClipboardActions({ ctx, state }) {
  function copySelectedSubgraphToClipboard() {
    const tensorIds = ctx.getSelectedIdsByKind("tensor");
    if (!tensorIds.length) {
      ctx.setStatus("Select one or more tensors to copy.");
      return;
    }
    const tensorIdSet = new Set(tensorIds);
    const clipboardPayload = {
      tensors: ctx.deepClone(
        state.spec.tensors.filter((tensor) => tensorIdSet.has(tensor.id))
      ),
      edges: ctx.deepClone(
        state.spec.edges.filter(
          (edge) =>
            tensorIdSet.has(edge.left.tensor_id) &&
            tensorIdSet.has(edge.right.tensor_id)
        )
      ),
      groups: ctx.deepClone(
        state.spec.groups.filter(
          (group) =>
            group.tensor_ids.length &&
            group.tensor_ids.every((tensorId) => tensorIdSet.has(tensorId))
        )
      ),
      pasteCount: 0,
    };
    state.clipboard = clipboardPayload;
    ctx.setStatus(
      `Copied ${clipboardPayload.tensors.length} tensor${clipboardPayload.tensors.length === 1 ? "" : "s"} to the editor clipboard.`,
      "success"
    );
  }

  function pasteClipboardToCanvas() {
    if (!state.clipboard || !Array.isArray(state.clipboard.tensors) || !state.clipboard.tensors.length) {
      ctx.setStatus("There is no copied tensor subgraph to paste.");
      return;
    }
    const pasteCount = (state.clipboard.pasteCount || 0) + 1;
    const offset = 40 * pasteCount;
    const tensorIdMap = {};
    const indexIdMap = {};
    const clipboard = ctx.deepClone(state.clipboard);

    clipboard.tensors.forEach((tensor) => {
      const nextTensorId = ctx.makeId("tensor");
      tensorIdMap[tensor.id] = nextTensorId;
      tensor.id = nextTensorId;
      tensor.position.x += offset;
      tensor.position.y += offset;
      tensor.indices.forEach((index) => {
        const nextIndexId = ctx.makeId("index");
        indexIdMap[index.id] = nextIndexId;
        index.id = nextIndexId;
      });
    });
    clipboard.edges.forEach((edge) => {
      edge.id = ctx.makeId("edge");
      edge.left.tensor_id = tensorIdMap[edge.left.tensor_id];
      edge.right.tensor_id = tensorIdMap[edge.right.tensor_id];
      edge.left.index_id = indexIdMap[edge.left.index_id];
      edge.right.index_id = indexIdMap[edge.right.index_id];
    });
    clipboard.groups.forEach((group) => {
      group.id = ctx.makeId("group");
      group.tensor_ids = group.tensor_ids.map((tensorId) => tensorIdMap[tensorId]);
    });

    state.clipboard.pasteCount = pasteCount;
    ctx.applyDesignChange(
      () => {
        state.spec.tensors.push(...clipboard.tensors);
        state.spec.edges.push(...clipboard.edges);
        state.spec.groups.push(...clipboard.groups);
        clipboard.tensors.forEach((tensor) => {
          ctx.bringTensorToFront(tensor.id);
        });
      },
      {
        selectionIds: clipboard.tensors.map((tensor) => tensor.id),
        primaryId: clipboard.tensors.length
          ? clipboard.tensors[clipboard.tensors.length - 1].id
          : null,
        statusMessage: `Pasted ${clipboard.tensors.length} tensor${clipboard.tensors.length === 1 ? "" : "s"} from the editor clipboard.`,
      }
    );
  }

  return {
    copySelectedSubgraphToClipboard,
    pasteClipboardToCanvas,
  };
}
