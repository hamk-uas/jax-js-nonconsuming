const fs = require("fs");

let file = fs.readFileSync("src/backend/webgpu/command-tape.ts", "utf8");
if (!file.includes("outputSizes: number[]")) {
  file = file.replace(
    "outputTableIdxs: number[];",
    "outputTableIdxs: number[];\n  /** Sizes of the outputs */\n  outputSizes: number[];",
  );
  fs.writeFileSync("src/backend/webgpu/command-tape.ts", file);
  console.log("added to interface");
}
