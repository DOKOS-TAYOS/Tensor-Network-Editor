function summarizeSerializedSpec(serializedSpec) {
  const network =
    serializedSpec && typeof serializedSpec === "object" && serializedSpec.network
      ? serializedSpec.network
      : null;
  if (!network || typeof network !== "object") {
    return {};
  }
  return {
    tensor_count: Array.isArray(network.tensors) ? network.tensors.length : 0,
    edge_count: Array.isArray(network.edges) ? network.edges.length : 0,
    group_count: Array.isArray(network.groups) ? network.groups.length : 0,
    note_count: Array.isArray(network.notes) ? network.notes.length : 0,
    mode:
      network.linear_periodic_chain
        ? "linear_periodic"
        : network.grid_periodic_grid
          ? "grid_periodic"
          : network.tree_periodic_tree
            ? "tree_periodic"
            : "normal",
  };
}

export function createTemplateCatalogService({ apiPost }) {
  return {
    promoteTemplate({ serializedSpec, tensorIds, templateName, overwrite = false }) {
      return apiPost(
        "/api/template/promote",
        {
          spec: serializedSpec,
          tensor_ids: tensorIds,
          template_name: templateName,
          overwrite,
        },
        {
          operation: "template.promote",
          context: {
            template_name: templateName,
            tensor_id_count: Array.isArray(tensorIds) ? tensorIds.length : 0,
            overwrite,
            ...summarizeSerializedSpec(serializedSpec),
          },
        }
      );
    },
    renameTemplate({ templateName, newTemplateName, overwrite = false }) {
      return apiPost(
        "/api/template/rename",
        {
          template_name: templateName,
          new_template_name: newTemplateName,
          overwrite,
        },
        {
          operation: "template.rename",
          context: {
            template_name: templateName,
            selected_template: newTemplateName,
            overwrite,
          },
        }
      );
    },
    deleteTemplate({ templateName }) {
      return apiPost(
        "/api/template/delete",
        {
          template_name: templateName,
        },
        {
          operation: "template.delete",
          context: {
            template_name: templateName,
          },
        }
      );
    },
  };
}
