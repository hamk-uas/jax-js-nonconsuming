const fs = require('fs');
let code = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

// 1. applyCalibration / resetCalibration
code = code.replace(
  '  readonly limits = {',
  `  resetCalibration() {
    this._microcalibrated = false;
  }

  applyCalibration(beliefState: PerformanceBeliefState) {
    this.deviceCapabilities.bandwidthGBs = beliefState.bandwidthGBs;
    this.deviceCapabilities.tflops = beliefState.tflops;
    this.deviceCapabilities.barrierRatio = beliefState.barrierRatio;
    this.deviceCapabilities.dispatchOverhead = beliefState.dispatchOverhead;
    this.deviceCapabilities.rOptWords = beliefState.rOptWords;
    this._microcalibrated = true;
    _clearJitCompileCache();
  }

  readonly limits = {`
);

// 2. mallocDeferred
code = code.replace(
  '  malloc(size: number, outSlot: TypedSlot): void {',
  `  mallocDeferred(size: number, outSlot: TypedSlot): void {
    outSlot.size = size;
    outSlot.sizeBytes = size * bytesPerElement(outSlot.dtype);
    outSlot.val = null; // Buffer allocated later via arena or pool
    if (DEBUG >= 2) {
      console.log(\`  mallocDeferred \${size} bytes for slot \${outSlot.idx}\`);
    }
  }

  malloc(size: number, outSlot: TypedSlot): void {`
);

// 3. compileCommandTape prep replacements
// I need to search and replace where compileCommandTape uses prepareKernelSync
// Actually, I can just find prepareKernelSync within compileCommandTape and replace it.
// The easiest way is to use regex. Since compileCommandTape loop handles kernel, assoc_scan etc.
// In step type "execute":
// old:
// const pipeline = this.#shaderCache.prepareKernelSync(step.source, shader);
// tape.push({ ... })
code = code.replace(
  /const pipeline = this\.#shaderCache\.prepareKernelSync\(step\.source, shader\);\s+tape\.push\(\{[\s\S]*?pipeline,/,
  `// pipeline generation deferred
        tape.push({
          type: "execute",
          step,
          shader,`
);

// In step type "assoc_scan" phase 1:
code = code.replace(
  /const pipeline1 = this\.#shaderCache\.prepareKernelSync\(\s+step\.phase1Source,\s+shader1,\s+\);\s+tape\.push\(\{[\s\S]*?pipeline: pipeline1,/,
  `tape.push({
            type: "execute",
            step,
            shader: shader1,`
);

// In step type "assoc_scan" phase 2:
code = code.replace(
  /const pipeline2 = this\.#shaderCache\.prepareKernelSync\(\s+step\.phase2Source,\s+shader2,\s+\);\s+tape\.push\(\{[\s\S]*?pipeline: pipeline2,/,
  `tape.push({
            type: "execute",
            step,
            shader: shader2,`
);

// In step type "block_map"
code = code.replace(
  /const pipeline = this\.#shaderCache\.prepareKernelSync\(step\.source, shader\);\s+tape\.push\(\{[\s\S]*?pipeline,/,
  `tape.push({
          type: "execute",
          step,
          shader,`
);

// In step type "routine"
code = code.replace(
  /const pipeline = this\.#shaderCache\.prepareRoutineSync\(\s+step\.routineParams\.type,\s+shader\.code,\s+shader,\s+\);\s+tape\.push\(\{[\s\S]*?pipeline,/,
  `tape.push({
          type: "execute",
          step,
          shader,`
);

// In step type "workgroup_assoc_scan"
code = code.replace(
  /const pipeline = this\.#shaderCache\.prepareKernelSync\(step\.source, shader\);\s+tape\.push\(\{[\s\S]*?pipeline,/,
  `tape.push({
          type: "execute",
          step,
          shader,`
);


// Malloc using mallocDeferred instead of malloc if no initialData
// In compileCommandTape malloc switch:
code = code.replace(
  /this\.malloc\(step\.numElements, step\.outBlks\[0\]\);\n\s+tape\.push\(\{\n\s+type: "malloc",\n\s+step,\n\s+slot: step\.outBlks\[0\],\n\s+\}\);/,
  `if (step.initialData) {
          this.malloc(step.numElements, step.outBlks[0]);
        } else {
          this.mallocDeferred(step.numElements, step.outBlks[0]);
        }
        tape.push({
          type: "malloc",
          step,
          slot: step.outBlks[0],
        });`
);


fs.writeFileSync('src/backend/webgpu.ts', code);
