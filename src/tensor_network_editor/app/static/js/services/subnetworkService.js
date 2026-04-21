export function createSubnetworkService({ apiPost }) {
  return {
    extractSubnetwork({ serializedSpec, tensorIds }) {
      return apiPost("/api/subnetwork/extract", {
        spec: serializedSpec,
        tensor_ids: tensorIds,
      });
    },
    prepareSubnetworkForInsert({ serializedSpec, targetCenter }) {
      return apiPost("/api/subnetwork/prepare-insert", {
        spec: serializedSpec,
        target_center: targetCenter,
      });
    },
    saveSubnetworkToLibrary({
      serializedSpec,
      tensorIds,
      subnetworkName,
      tags = [],
      overwrite = false,
    }) {
      return apiPost("/api/subnetwork-library/save", {
        spec: serializedSpec,
        tensor_ids: tensorIds,
        subnetwork_name: subnetworkName,
        tags,
        overwrite,
      });
    },
    renameLibrarySubnetwork({
      subnetworkName,
      newSubnetworkName,
      overwrite = false,
    }) {
      return apiPost("/api/subnetwork-library/rename", {
        subnetwork_name: subnetworkName,
        new_subnetwork_name: newSubnetworkName,
        overwrite,
      });
    },
    deleteLibrarySubnetwork({ subnetworkName }) {
      return apiPost("/api/subnetwork-library/delete", {
        subnetwork_name: subnetworkName,
      });
    },
    prepareLibrarySubnetworkForInsert({ subnetworkName, targetCenter }) {
      return apiPost("/api/subnetwork-library/prepare-insert", {
        subnetwork_name: subnetworkName,
        target_center: targetCenter,
      });
    },
  };
}
