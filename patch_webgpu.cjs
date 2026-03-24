const fs = require("fs");
let code = fs.readFileSync("src/backend/webgpu.ts", "utf8");

// 1. replace pipeline: GPUComputePipeline with shader: ShaderInfo
code = code.replace(
  "  prepareKernelSync(source: string, info: ShaderInfo): GPUComputePipeline {",
  `
  // Added bindGroup caches
  #bindGroup0Cache: Map<string, GPUBindGroupLayout> = new Map();
  #bindGroup1Cache: Map<string, GPUBindGroupLayout> = new Map();

  getLayout(shader: ShaderInfo): GPUBindGroupLayout {
    const key = \`\${shader.numInputs}:\${shader.numOutputs}\`;
    let cached = this.#bindGroup0Cache.get(key);
    if (!cached) {
      cached = this.device.createBindGroupLayout({
        entries: range(shader.numInputs + shader.numOutputs).map((i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: i < shader.numInputs ? "read-only-storage" : "storage",
          },
        })),
      });
      this.#bindGroup0Cache.set(key, cached);
    }
    return cached;
  }

  getUniformLayout(shader: ShaderInfo): GPUBindGroupLayout {
    const nuc = shader.numUniformConsts ?? 0;
    const key = \`\${shader.hasUniform ? 1 : 0}:\${nuc}\`;
    let cached = this.#bindGroup1Cache.get(key);
    if (!cached) {
      if (nuc > 0) {
        cached = this.device.createBindGroupLayout({
          entries: range(nuc).map((i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          })),
        });
      } else if (shader.hasUniform) {
        cached = this.device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "uniform", hasDynamicOffset: true },
            },
          ],
        });
      } else {
        throw new Error("Shader has no uniform layout");
      }
      this.#bindGroup1Cache.set(key, cached);
    }
    return cached;
  }

  prepareKernelSync(source: string, info: ShaderInfo): GPUComputePipeline {`,
);

// 2. update #getLayout to use getLayout and getUniformLayout
code = code.replace(
  "  #getLayout(shader: ShaderInfo): GPUPipelineLayout {\n    const bindGroupLayouts: GPUBindGroupLayout[] = [];\n\n    const storageKey = `${shader.numInputs}:${shader.numOutputs}`;\n    let storageCached = this.#layoutCache.get(storageKey);\n    if (!storageCached) {\n      // this logic previously created a layout inside #getLayout, replacing with getLayout calls\n",
  "  #getLayout(shader: ShaderInfo): GPUPipelineLayout {\n",
);

code = code.replace(
  /  #getLayout\(shader: ShaderInfo\): GPUPipelineLayout \{[\s\S]*?    const key = `\$\{storageKey\}-\$\{uniformKey\}`;/,
  `  #getLayout(shader: ShaderInfo): GPUPipelineLayout {
    let storageKey = \`\${shader.numInputs}:\${shader.numOutputs}\`;
    let uniformKey = "none";
    if (shader.hasUniform || (shader.numUniformConsts ?? 0) > 0) {
      const nuc = shader.numUniformConsts ?? 0;
      uniformKey = \`\${shader.hasUniform ? 1 : 0}:\${nuc}\`;
    }
    const key = \`\${storageKey}-\${uniformKey}\`;`,
);

code = code.replace(
  /      const layout = this\.device\.createPipelineLayout\(\{ bindGroupLayouts \}\);\n      this\.#layoutCache\.set\(key, layout\);\n      return layout;/g,
  `      const bindGroupLayouts = shader.hasUniform || (shader.numUniformConsts ?? 0) > 0 ? [this.getLayout(shader), this.getUniformLayout(shader)] : [this.getLayout(shader)];
      const layout = this.device.createPipelineLayout({ bindGroupLayouts });
      this.#layoutCache.set(key, layout);
      return layout;`,
);

fs.writeFileSync("src/backend/webgpu.ts", code);
