const fs = require('fs');
let code = fs.readFileSync('src/backend/webgpu.ts', 'utf8');

// The compileCommandTape function pushes to `ops`
// We need to change where it assigns pipeline: pipeline to shader: shader
// Previously, I replaced pipeline: with shader: using my restore script, but then I did 'git checkout' so they are reverted.
// Let's replace 'pipeline' with 'shader' in the tape op:
code = code.replace(
  /const pipeline = this\.#shaderCache\.prepareKernelSync\(step\.source, shader\);\n\s+const bindGroupLayout = this\.#shaderCache\.#getLayout\(shader\);/g,
  `const bindGroupLayout = this.#shaderCache.getLayout(shader);`
);

code = code.replace(
  /const pipeline1 = this\.#shaderCache\.prepareKernelSync\(\n\s+step\.phase1Source,\n\s+shader1,\n\s+\);\n\s+const bindGroupLayout1 = this\.#shaderCache\.#getLayout\(shader1\);/g,
  `const bindGroupLayout1 = this.#shaderCache.getLayout(shader1);`
);

code = code.replace(
  /const pipeline2 = this\.#shaderCache\.prepareKernelSync\(\n\s+step\.phase2Source,\n\s+shader2,\n\s+\);\n\s+const bindGroupLayout2 = this\.#shaderCache\.#getLayout\(shader2\);/g,
  `const bindGroupLayout2 = this.#shaderCache.getLayout(shader2);`
);

code = code.replace(
  /const pipeline = this\.#shaderCache\.prepareRoutineSync\(\n\s+step\.routineParams\.type,\n\s+shader\.code,\n\s+shader,\n\s+\);\n\s+const bindGroupLayout = this\.#shaderCache\.#getLayout\(shader\);/g,
  `const bindGroupLayout = this.#shaderCache.getLayout(shader);`
);

// Now change `pipeline` field in the push calls to `shader`:

code = code.replace(/pipeline,\n\s+bindGroupLayout,/g, `shader,\n                  bindGroupLayout,`);
code = code.replace(/pipeline: pipeline1,\n\s+bindGroupLayout: bindGroupLayout1,/g, `shader: shader1,\n                  bindGroupLayout: bindGroupLayout1,`);
code = code.replace(/pipeline: pipeline2,\n\s+bindGroupLayout: bindGroupLayout2,/g, `shader: shader2,\n                  bindGroupLayout: bindGroupLayout2,`);

fs.writeFileSync('src/backend/webgpu.ts', code);
