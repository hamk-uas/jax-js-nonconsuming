import fs from "fs";
let code = fs.readFileSync("src/backend/webgpu.ts", "utf8");

code = code.replace(
  "  #layoutCache: Map<string, GPUPipelineLayout>;",
  "  #layoutCache: Map<string, GPUPipelineLayout>;\n  #bindGroup0Cache: Map<string, GPUBindGroupLayout>;\n  #bindGroup1Cache: Map<string, GPUBindGroupLayout>;",
);

code = code.replace(
  "    this.#layoutCache = new Map();\n  }",
  "    this.#layoutCache = new Map();\n    this.#bindGroup0Cache = new Map();\n    this.#bindGroup1Cache = new Map();\n  }",
);

const layoutCodeReplacement = `  getLayout(shader: ShaderInfo): GPUBindGroupLayout {
    const key = \`\${shader.numInputs}:\${shader.numOutputs}\`;
    let cached = this.#bindGroup0Cache.get(key);
      cached = this.device.createBindGroupLayout({
        entries: range(shader.numInputs + shader.numOutputs).map((i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: i < shader.numInputs ? 'read-only-storage' : 'storage',
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
      if (nuc > 0) {
        cached = this.device.createBindGroupLayout({
          entries: range(nuc).map((i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' as const },
          })),
        });
      } else if (shader.hasUniform) {
        cached = this.device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform', hasDynamicOffset: true },
            },
          ],
        });
      } else {
        throw new Error('Shader has no uniform layout');
      }
      this.#bindGroup1Cache.set(key, cached);
    }
    return cached;
  }

  #getLayout(shader: ShaderInfo): GPUPipelineLayout {`;

code = code.replace(
  "  #getLayout(shader: ShaderInfo): GPUPipelineLayout {",
  layoutCodeReplacement,
);

code = code.replace(
  "      const layout = this.device.createPipelineLayout({ bindGroupLayouts });",
  "      const layout = this.device.createPipelineLayout({\n        bindGroupLayouts: shader.hasUniform || (shader.numUniformConsts ?? 0) > 0\n          ? [this.getLayout(shader), this.getUniformLayout(shader)]\n          : [this.getLayout(shader)]\n      });",
);

fs.writeFileSync("src/backend/webgpu.ts", code);
