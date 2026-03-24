const fs = require("fs");
let code = fs.readFileSync("src/backend/webgpu.ts", "utf8");

code = code.replace(
  "  #getLayout(shader: ShaderInfo): GPUPipelineLayout {",
  `
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

  #getLayout(shader: ShaderInfo): GPUPipelineLayout {`,
);

let start = code.indexOf(
  "    const bindGroupLayouts: GPUBindGroupLayout[] = [",
);
let end = code.indexOf(
  "    const layout = this.device.createPipelineLayout({ bindGroupLayouts });",
  start,
);
console.log(start, end);

if (start !== -1 && end !== -1) {
  let toReplace = code.substring(start, end);
  code = code.replace(
    toReplace,
    `    const bindGroupLayouts = shader.hasUniform || (shader.numUniformConsts ?? 0) > 0 ? [this.getLayout(shader), this.getUniformLayout(shader)] : [this.getLayout(shader)];\n\n`,
  );
}

fs.writeFileSync("src/backend/webgpu.ts", code);
