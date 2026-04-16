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
  };
}
