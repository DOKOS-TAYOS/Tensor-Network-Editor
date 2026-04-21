export function createSessionTemplateDialogs({ sessionUi, actions }) {
  function promptForTemplateDisplayName(defaultDisplayName, cancelledStatus) {
    const promptedDisplayName = sessionUi.promptText(
      "Choose a name for this template.",
      defaultDisplayName
    );
    if (typeof promptedDisplayName !== "string") {
      actions.setStatus(cancelledStatus);
      return null;
    }
    const trimmedDisplayName = promptedDisplayName.trim();
    if (!trimmedDisplayName) {
      actions.setStatus("Template names cannot be empty.", "error");
      return null;
    }
    return trimmedDisplayName;
  }

  function promptForSubnetworkName(defaultDisplayName, cancelledStatus) {
    const promptedDisplayName = sessionUi.promptText(
      "Choose a name for this subnetwork.",
      defaultDisplayName
    );
    if (typeof promptedDisplayName !== "string") {
      actions.setStatus(cancelledStatus);
      return null;
    }
    const trimmedDisplayName = promptedDisplayName.trim();
    if (!trimmedDisplayName) {
      actions.setStatus("Subnetwork names cannot be empty.", "error");
      return null;
    }
    return trimmedDisplayName;
  }

  function promptForSubnetworkTags(defaultTags = [], cancelledStatus) {
    const promptedTags = sessionUi.promptText(
      "Add tags separated by commas. Leave empty for no tags.",
      Array.isArray(defaultTags) ? defaultTags.join(", ") : ""
    );
    if (typeof promptedTags !== "string") {
      actions.setStatus(cancelledStatus);
      return null;
    }
    return promptedTags
      .split(",")
      .map((tag) => tag.trim())
      .filter((tag) => tag);
  }

  return {
    promptForTemplateDisplayName,
    promptForSubnetworkName,
    promptForSubnetworkTags,
  };
}
