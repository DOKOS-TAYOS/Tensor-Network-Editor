export function createTemplateCatalogService({ apiPost }) {
  return {
    promoteTemplate({ serializedSpec, tensorIds, templateName, overwrite = false }) {
      return apiPost("/api/template/promote", {
        spec: serializedSpec,
        tensor_ids: tensorIds,
        template_name: templateName,
        overwrite,
      });
    },
    renameTemplate({ templateName, newTemplateName, overwrite = false }) {
      return apiPost("/api/template/rename", {
        template_name: templateName,
        new_template_name: newTemplateName,
        overwrite,
      });
    },
    deleteTemplate({ templateName }) {
      return apiPost("/api/template/delete", {
        template_name: templateName,
      });
    },
  };
}
