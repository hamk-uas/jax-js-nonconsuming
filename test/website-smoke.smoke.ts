import { existsSync } from "node:fs";

import { describe, expect, test } from "vitest";

describe("website build smoke", () => {
  test("produces key static pages", () => {
    expect(existsSync("website/build/index.html")).toBe(true);
    expect(existsSync("website/build/repl.html")).toBe(true);
    expect(existsSync("website/build/mnist.html")).toBe(true);
  });
});
