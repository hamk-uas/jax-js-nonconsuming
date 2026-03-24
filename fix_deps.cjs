const fs = require("fs");

let webgpu = fs.readFileSync("src/backend/webgpu.ts", "utf8");

// Fix imports in webgpu.ts
if (!webgpu.includes("IPendingExecute")) {
  webgpu = webgpu.replace(
    /import type { JitId, JitStep }/g,
    'import type { IPendingExecute } from "../frontend/array";\nimport type { JitId, JitStep }',
  );
}

// Fix missing outputSizes when building Tape
webgpu = webgpu.replace(
  /outputTableIdxs,\n\s*allocatedIdxs/g,
  "outputTableIdxs,\n      outputSizes: outputTableIdxs.map(idx => knownSizes.get(idx)!),\n      allocatedIdxs",
);

// Fix d.pipeline usage
webgpu = webgpu.replace(
  /pe.setPipeline\(d.pipeline\);/g,
  "pe.setPipeline(this.pipelines.prepareSync(d.shader));",
);

fs.writeFileSync("src/backend/webgpu.ts", webgpu);

const repPending = (file) => {
  let content = fs.readFileSync(file, "utf8");
  content = content.replace(
    /import \{.*?PendingExecute.*\} from "\.\/array"/g,
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
  content = content.replace(
    /pending:\s*PendingExecute\[\]/g,
    "pending: IPendingExecute[]",
  );
  content = content.replace(/PendingExecute\[\]\s*=/g, "IPendingExecute[] =");
  content = content.replace(/Array<PendingExecute>/g, "Array<IPendingExecute>");
  content = content.replace(
    /flushPending\((.*?): PendingExecute\[\]/g,
    "flushPending($1: IPendingExecute[]",
  );
  // wait exactly match imports
  if (!content.includes("IPendingExecute")) {
    content = content.replace(
      /import \{ (.*?) \} from "\.\/array";/,
      'import { $1, IPendingExecute } from "./array";',
    );
  }
  fs.writeFileSync(file, content);
};

[
  "src/frontend/block-map-executor.ts",
  "src/frontend/scan-executor.ts",
  "src/frontend/jit.ts",
].forEach(repPending);
