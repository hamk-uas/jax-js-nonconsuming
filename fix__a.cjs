const fs = require("fs");

for (const file of [
  "src/frontend/block-map-executor.ts",
  "src/frontend/scan-executor.ts",
  "src/frontend/jit.ts",
]) {
  let content = fs.readFileSync(file, "utf8");
  content = content.replace(
    /import type \{ PendingExecute \} from "\.\/array";/g,
    'import type { PendingExecute, IPendingExecute } from "./array";',
  );
  content = content.replace(
    /import \{.*?PendingExecute.*?\} from "\.\/array";/g,
    (match) => {
      if (!match.includes("IPendingExecute")) {
        return match.replace(
          /PendingExecute/g,
          "PendingExecute, IPendingExecute",
        );
      }
      return match;
    },
  );
  fs.writeFileSync(file, content);
}
