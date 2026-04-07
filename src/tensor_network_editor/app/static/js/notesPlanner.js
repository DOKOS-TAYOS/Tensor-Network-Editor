import { registerNotesFeature } from "./notes.js";
import { registerPlannerFeature } from "./planner.js";

export function registerNotesPlanner(ctx) {
  registerNotesFeature(ctx);
  registerPlannerFeature(ctx);
}
