const fs = require('fs');
let code = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

code = code.replace(
  '  readonly capabilities: BackendCapabilities;',
  `  readonly capabilities: BackendCapabilities;

  // Added dynamically by JIT calibration
  resetCalibration() {
    this.capabilities.calibrated = false;
  }

  applyCalibration(beliefState: PerformanceBeliefState) {
    this.capabilities.bandwidthGBs = beliefState.bandwidthGBs;
    this.capabilities.tflops = beliefState.tflops;
    this.capabilities.barrierRatio = beliefState.barrierRatio;
    this.capabilities.dispatchOverhead = beliefState.dispatchOverhead;
    this.capabilities.rOptWords = beliefState.rOptWords;
    this.capabilities.calibrated = true;
    _clearJitCompileCache();
  }`
);

fs.writeFileSync('src/backend/webgpu.ts', code);
