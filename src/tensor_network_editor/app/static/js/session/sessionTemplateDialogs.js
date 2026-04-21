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

  return {
    promptForTemplateDisplayName,
    promptForSubnetworkName,
  };
}
