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

export function createSubnetworkService({ apiPost }) {
  return {
    extractSubnetwork({ serializedSpec, tensorIds }) {
      return apiPost(
        "/api/subnetwork/extract",
        {
          spec: serializedSpec,
          tensor_ids: tensorIds,
        },
        {
          operation: "subnetwork.extract",
          context: {
            tensor_id_count: Array.isArray(tensorIds) ? tensorIds.length : 0,
            ...summarizeSerializedSpec(serializedSpec),
          },
        }
      );
    },
    prepareSubnetworkForInsert({ serializedSpec, targetCenter }) {
      return apiPost(
        "/api/subnetwork/prepare-insert",
        {
          spec: serializedSpec,
          target_center: targetCenter,
        },
        {
          operation: "subnetwork.prepare_insert",
          context: summarizeSerializedSpec(serializedSpec),
        }
      );
    },
    saveSubnetworkToLibrary({
      serializedSpec,
      tensorIds,
      subnetworkName,
      tags = [],
      overwrite = false,
    }) {
      return apiPost(
        "/api/subnetwork-library/save",
        {
          spec: serializedSpec,
          tensor_ids: tensorIds,
          subnetwork_name: subnetworkName,
          tags,
          overwrite,
        },
        {
          operation: "subnetwork.save_library",
          context: {
            subnetwork_name: subnetworkName,
            tensor_id_count: Array.isArray(tensorIds) ? tensorIds.length : 0,
            tag_count: Array.isArray(tags) ? tags.length : 0,
            overwrite,
            ...summarizeSerializedSpec(serializedSpec),
          },
        }
      );
    },
    renameLibrarySubnetwork({
      subnetworkName,
      newSubnetworkName,
      overwrite = false,
    }) {
      return apiPost(
        "/api/subnetwork-library/rename",
        {
          subnetwork_name: subnetworkName,
          new_subnetwork_name: newSubnetworkName,
          overwrite,
        },
        {
          operation: "subnetwork.rename_library",
          context: {
            subnetwork_name: subnetworkName,
            selected_subnetwork: newSubnetworkName,
            overwrite,
          },
        }
      );
    },
    deleteLibrarySubnetwork({ subnetworkName }) {
      return apiPost(
        "/api/subnetwork-library/delete",
        {
          subnetwork_name: subnetworkName,
        },
        {
          operation: "subnetwork.delete_library",
          context: {
            subnetwork_name: subnetworkName,
          },
        }
      );
    },
    prepareLibrarySubnetworkForInsert({ subnetworkName, targetCenter }) {
      return apiPost(
        "/api/subnetwork-library/prepare-insert",
        {
          subnetwork_name: subnetworkName,
          target_center: targetCenter,
        },
        {
          operation: "subnetwork.prepare_library_insert",
          context: {
            subnetwork_name: subnetworkName,
          },
        }
      );
    },
  };
}
