import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/website-smoke.smoke.ts"],
    environment: "node",
    passWithNoTests: false,
  },
});
