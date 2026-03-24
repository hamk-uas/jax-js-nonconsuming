const fs = require("fs");
let code = fs.readFileSync("src/backend/webgpu.ts", "utf8");

code = code.replace(
  "  resetCalibration() {",
  `  resetCalibration() {
    // @ts-ignore
    this.capabilities.calibrated = false;
  }`,
);

code = code.replace(
  `  applyCalibration(beliefState: any) {
    this.capabilities.bandwidthGBs = beliefState.bandwidthGBs;
    this.capabilities.tflops = beliefState.tflops;
    this.capabilities.barrierRatio = beliefState.barrierRatio;
    this.capabilities.dispatchOverhead = beliefState.dispatchOverhead;
    this.capabilities.rOptWords = beliefState.rOptWords;
    this.capabilities.calibrated = true;
    _clearJitCompileCache();
  }`,
  `  applyCalibration(beliefState: any) {
    Object.assign(this.capabilities as any, beliefState);
    (this.capabilities as any).calibrated = true;
  }`,
);

fs.writeFileSync("src/backend/webgpu.ts", code);
