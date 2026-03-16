import {
  defaultDevice,
  getWebGPUDevice,
  init,
  jit,
  numpy as np,
  withBatch,
} from "@hamk-uas/jax-js-nonconsuming";

// --- Simulation parameters ---
const JACOBI_ITERS = 10;
const DT = 1;
const DX = 1;
const INFLOW_SPEED = 3;
const OBSTACLE_RAD_FRAC = 0.02;

// --- Simulation state ---
let velocity: np.Array;
let pressure: np.Array;
let material: np.Array;

let W = 0;
let H = 0;
let AW = 0;
let AH = 0;

let obstacleX = 0;
let obstacleY = 0;
let obstacleRad = 0;

let running = false;

let mouseDown = false;
let movingObstacle = false;
let mouseX = 0;
let mouseY = 0;
let lastMouseX = 0;
let lastMouseY = 0;

// --- Helper: shift a 2D field along an axis ---
// Periodic (wrap) in Y, zero-pad in X.
function shift2D(field: np.Array, axis: number, amount: number): np.Array {
  const [h, w] = field.shape;
  if (axis === 0) {
    // Periodic in Y: wrap around
    if (amount === 1) {
      const body = field.slice([1, h], []);
      const wrap = field.slice([0, 1], []);
      return np.concatenate([body, wrap], 0);
    } else {
      const wrap = field.slice([h - 1, h], []);
      const body = field.slice([0, h - 1], []);
      return np.concatenate([wrap, body], 0);
    }
  } else {
    // Zero-pad in X (inflow/outflow)
    if (amount === 1) {
      return np.pad(field.slice([], [1, w]), { 1: [0, 1] });
    } else {
      return np.pad(field.slice([], [0, w - 1]), { 1: [1, 0] });
    }
  }
}

// --- Core simulation steps (jit, no number args) ---

const computeDivergence = jit(function computeDivergence(
  vel: np.Array,
): np.Array {
  const vx = vel.slice([], [], 0);
  const vy = vel.slice([], [], 1);
  const e = shift2D(vx, 1, 1);
  const w = shift2D(vx, 1, -1);
  const n = shift2D(vy, 0, 1);
  const s = shift2D(vy, 0, -1);
  const coeff = 0.5 / DX;
  return e.sub(w).add(n.sub(s)).mul(coeff);
});

const jacobiStep = jit(function jacobiStep(
  divergence: np.Array,
  p: np.Array,
): np.Array {
  const n = shift2D(p, 0, 1);
  const s = shift2D(p, 0, -1);
  const e = shift2D(p, 1, 1);
  const w = shift2D(p, 1, -1);
  const alpha = -(DX * DX);
  return n.add(s).add(e).add(w).add(divergence.mul(alpha)).mul(0.25);
});

const subtractGradient = jit(function subtractGradient(
  vel: np.Array,
  p: np.Array,
): np.Array {
  const pE = shift2D(p, 1, 1);
  const pW = shift2D(p, 1, -1);
  const pN = shift2D(p, 0, 1);
  const pS = shift2D(p, 0, -1);
  const coeff = 0.5 / DX;
  const gradX = pE.sub(pW).mul(coeff);
  const gradY = pN.sub(pS).mul(coeff);
  const grad = np.stack([gradX, gradY], 2);
  return vel.sub(grad);
});

// --- Advection: created as closures after dimensions are known ---
let advectVel: (field: np.Array, vel: np.Array) => np.Array;
let advectMat: (field: np.Array, vel: np.Array) => np.Array;

function makeAdvect(fieldH: number, fieldW: number, velScale: number) {
  return jit(function advect(field: np.Array, vel: np.Array): np.Array {
    const gy = np.arange(fieldH);
    const gx = np.arange(fieldW);
    const [GX, GY] = np.meshgrid([gx, gy]);

    // Sample velocity at field coordinates
    const sampleY = np.clip(
      GY.astype(np.float32).mul(velScale).astype(np.int32),
      0,
      vel.shape[0] - 1,
    );
    const sampleX = np.clip(
      GX.astype(np.float32).mul(velScale).astype(np.int32),
      0,
      vel.shape[1] - 1,
    );
    const sampledVel = vel.slice(sampleY, sampleX);

    // Backtrace
    const vx = sampledVel.slice([], [], 0);
    const vy = sampledVel.slice([], [], 1);
    const srcX = GX.astype(np.float32).sub(vx.mul(DT / velScale));
    const srcY = GY.astype(np.float32).sub(vy.mul(DT / velScale));

    const srcXc = np.clip(srcX, 0, fieldW - 1.001);
    const srcYc = np.clip(srcY, 0, fieldH - 1.001);

    // Bilinear interpolation
    const x0 = np.floor(srcXc).astype(np.int32);
    const y0 = np.floor(srcYc).astype(np.int32);
    const x1 = np.clip(x0.add(1), 0, fieldW - 1);
    const y1 = np.clip(y0.add(1), 0, fieldH - 1);

    const fx = srcXc.sub(x0.astype(np.float32));
    const fy = srcYc.sub(y0.astype(np.float32));

    // Gather 4 neighbors
    const f00 = field.slice(y0, x0);
    const f10 = field.slice(y0, x1);
    const f01 = field.slice(y1, x0);
    const f11 = field.slice(y1, x1);

    // Expand weights for channel dimension if needed
    const hasChannels = f00.shape.length === 3;
    const fxW: np.Array = hasChannels ? fx.slice([], [], null) : fx;
    const fyW: np.Array = hasChannels ? fy.slice([], [], null) : fy;

    // Bilinear blend
    const oneMinusFx = np.ones(fxW.shape).sub(fxW);
    const oneMinusFy = np.ones(fyW.shape).sub(fyW);
    const top = f00.mul(oneMinusFx).add(f10.mul(fxW));
    const bot = f01.mul(oneMinusFx).add(f11.mul(fxW));
    return top.mul(oneMinusFy).add(bot.mul(fyW));
  });
}

const applyBoundary = jit(function applyBoundary(
  vel: np.Array,
  obsXv: np.Array,
  obsYv: np.Array,
  obsRadV: np.Array,
): np.Array {
  const [h, w] = vel.shape;
  const gy = np.arange(h).astype(np.float32);
  const gx = np.arange(w).astype(np.float32);
  const [GX, GY] = np.meshgrid([gx, gy]);

  // Direction and distance from obstacle center
  const dxArr = GX.sub(obsXv);
  const dyArr = GY.sub(obsYv);
  const distSq = dxArr.mul(dxArr).add(dyArr.mul(dyArr));
  const dist = np.sqrt(distSq);

  // Masks: inside obstacle, 2px shell, outside
  const insideMask = distSq
    .less(np.square(obsRadV.sub(2.0)))
    .astype(np.float32);
  const withinShell = dist.less(obsRadV).astype(np.float32);
  const shellMask = withinShell.sub(insideMask);
  const outsideMask = np.ones(shellMask.shape).sub(insideMask).sub(shellMask);

  // Reflected position for shell: pos + normalize(dir) * 2
  const safeDist = dist.add(1e-6);
  const reflX = np.clip(
    GX.add(dxArr.div(safeDist).mul(2)).astype(np.int32),
    0,
    w - 1,
  );
  const reflY = np.clip(
    GY.add(dyArr.div(safeDist).mul(2)).astype(np.int32),
    0,
    h - 1,
  );

  // Gather reflected velocity (negated for no-slip)
  const reflVel = vel.slice(reflY, reflX);
  const reflVx = reflVel.slice([], [], 0).mul(-1);
  const reflVy = reflVel.slice([], [], 1).mul(-1);

  // Split velocity channels
  const vx = vel.slice([], [], 0);
  const vy = vel.slice([], [], 1);

  // Combine: outside keeps original, shell gets negated reflection, inside is zero
  const vxCombined = vx.mul(outsideMask).add(reflVx.mul(shellMask));
  const vyCombined = vy.mul(outsideMask).add(reflVy.mul(shellMask));

  // Inflow: left column vx=INFLOW_SPEED
  const leftMask = GX.less(1).astype(np.float32);
  const notLeft = np.ones(leftMask.shape).sub(leftMask);
  const vxFinal = vxCombined.mul(notLeft).add(leftMask.mul(INFLOW_SPEED));

  return np.stack([vxFinal, vyCombined], 2);
});

const applyForce = jit(function applyForce(
  vel: np.Array,
  mx: np.Array,
  my: np.Array,
  fdx: np.Array,
  fdy: np.Array,
  radius: np.Array,
): np.Array {
  const [h, w] = vel.shape;
  const gy = np.arange(h).astype(np.float32);
  const gx = np.arange(w).astype(np.float32);
  const [GX, GY] = np.meshgrid([gx, gy]);
  const dxArr = GX.sub(mx);
  const dyArr = GY.sub(my);
  const distSq = dxArr.mul(dxArr).add(dyArr.mul(dyArr));
  const gauss = np.exp(distSq.mul(radius).mul(-1));
  const forceX = gauss.mul(fdx.mul(DT));
  const forceY = gauss.mul(fdy.mul(DT));
  const force = np.stack([forceX, forceY], 2);
  return vel.add(force);
});

// applyMaterialBoundary: baked via closure
let applyMatBoundary: (mat: np.Array) => np.Array;

function makeApplyMatBoundary(matH: number) {
  let numBands = Math.floor(matH / 10);
  if (numBands % 2 === 1) numBands--;
  const bandFreq = (Math.PI * numBands) / matH;

  return jit(function applyMatBoundary(mat: np.Array): np.Array {
    const [h, w] = mat.shape;
    const gy = np.arange(h);
    const gx = np.arange(w);
    const [GX, GY] = np.meshgrid([gx, gy]);

    const bandVal = np
      .sin(GY.astype(np.float32).mul(bandFreq))
      .greater(0)
      .astype(np.float32);

    const leftMask = GX.less(1).astype(np.float32);
    const notLeft = np.ones(leftMask.shape).sub(leftMask);
    return mat.mul(notLeft).add(bandVal.mul(leftMask));
  });
}

// --- WebGPU Rendering ---
let gpuDevice: GPUDevice;
let renderPipeline: GPURenderPipeline;
let renderBindGroupLayout: GPUBindGroupLayout;
let canvasContext: GPUCanvasContext;
let uniformBuffer: GPUBuffer;

function initRenderer(canvas: HTMLCanvasElement) {
  gpuDevice = getWebGPUDevice();

  canvasContext = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  canvasContext.configure({ device: gpuDevice, format, alphaMode: "opaque" });

  const shaderCode = /* wgsl */ `
    struct Uniforms {
      texSize: vec2f,
      obstaclePos: vec2f,
      obstacleRad: f32,
    };

    @group(0) @binding(0) var<uniform> u: Uniforms;
    @group(0) @binding(1) var<storage, read> material: array<f32>;

    struct VertexOutput {
      @builtin(position) pos: vec4f,
      @location(0) uv: vec2f,
    };

    @vertex fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
      var pos = array<vec2f, 3>(
        vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3)
      );
      var out: VertexOutput;
      out.pos = vec4f(pos[vi], 0, 1);
      out.uv = (pos[vi] + 1) * 0.5;
      return out;
    }

    @fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
      let texW = u32(u.texSize.x);
      let texH = u32(u.texSize.y);
      let px = u32(in.uv.x * f32(texW));
      let py = u32((1.0 - in.uv.y) * f32(texH));
      let idx = py * texW + px;

      let fragX = in.uv.x * f32(texW);
      let fragY = (1.0 - in.uv.y) * f32(texH);
      let dx = fragX - u.obstaclePos.x;
      let dy = fragY - u.obstaclePos.y;
      if (dx * dx + dy * dy < u.obstacleRad * u.obstacleRad) {
        return vec4f(0.925, 0.0, 0.55, 1.0);
      }

      let val = material[idx];
      let color = vec3f(0.98, 0.93, 0.84) * val;
      return vec4f(color, 1.0);
    }
  `;

  const shaderModule = gpuDevice.createShaderModule({ code: shaderCode });

  renderBindGroupLayout = gpuDevice.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" },
      },
    ],
  });

  renderPipeline = gpuDevice.createRenderPipeline({
    layout: gpuDevice.createPipelineLayout({
      bindGroupLayouts: [renderBindGroupLayout],
    }),
    vertex: { module: shaderModule },
    fragment: { module: shaderModule, targets: [{ format }] },
  });

  uniformBuffer = gpuDevice.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function renderFrame(matBuffer: GPUBuffer) {
  const uniformData = new Float32Array([
    AW,
    AH,
    obstacleX,
    obstacleY,
    obstacleRad,
    0,
    0,
    0,
  ]);
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const bindGroup = gpuDevice.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: matBuffer } },
    ],
  });

  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: canvasContext.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  });
  pass.setPipeline(renderPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
}

// --- Main loop ---
// Optimizations vs upstream:
// 1. withBatch: batches all JIT dispatches into a single queue.submit()
// 2. gpuBufferSync: synchronous GPU buffer access (no async overhead per frame)
async function simulate() {
  while (running) {
    // All simulation dispatches batched into a single queue.submit()
    withBatch(() => {
      // 1) Advect velocity
      {
        using prev = velocity;
        velocity = advectVel(prev, prev);
      }

      // 2) Boundary conditions
      const obsXv = (obstacleX / AW) * W;
      const obsYv = (obstacleY / AH) * H;
      const obsRadV = (obstacleRad / AW) * W;
      {
        using prev = velocity;
        velocity = applyBoundary(prev, obsXv, obsYv, obsRadV);
      }

      // 3) Mouse force
      if (mouseDown && !movingObstacle) {
        const mx = (mouseX / AW) * W;
        const my = (mouseY / AH) * H;
        const fdx = (2 * (mouseX - lastMouseX)) / 3.5;
        const fdy = (2 * (mouseY - lastMouseY)) / 3.5;
        const radiusParam = 0.001 * Math.max(AW, AH);
        {
          using prev = velocity;
          velocity = applyForce(prev, mx, my, fdx, fdy, radiusParam);
        }
        {
          using prev = velocity;
          velocity = applyBoundary(prev, obsXv, obsYv, obsRadV);
        }
      }

      // 4) Pressure projection (warm-start from previous frame)
      {
        using div = computeDivergence(velocity);
        for (let i = 0; i < JACOBI_ITERS; i++) {
          using prev = pressure;
          pressure = jacobiStep(div, prev);
        }
      }

      // 5) Subtract pressure gradient
      {
        using prev = velocity;
        velocity = subtractGradient(prev, pressure);
      }
      {
        using prev = velocity;
        velocity = applyBoundary(prev, obsXv, obsYv, obsRadV);
      }

      // 6) Advect material
      {
        using prev = material;
        material = advectMat(prev, velocity);
      }
      {
        using prev = material;
        material = applyMatBoundary(prev);
      }
    });

    // Synchronous GPU buffer access — no async overhead per frame
    const matBuffer = material.gpuBufferSync();
    renderFrame(matBuffer);

    await new Promise((r) => requestAnimationFrame(r));
  }
}

// --- Public API ---

export async function startSimulation(
  canvas: HTMLCanvasElement,
): Promise<() => void> {
  await init("webgpu");
  defaultDevice("webgpu");

  AW = Math.min(canvas.clientWidth, 1200);
  AH = Math.min(canvas.clientHeight, 800);
  canvas.width = AW;
  canvas.height = AH;

  const maxDim = Math.max(AW, AH);
  const scale = maxDim / 300;
  W = Math.floor(AW / scale);
  H = Math.floor(AH / scale);

  obstacleRad = OBSTACLE_RAD_FRAC * maxDim;
  obstacleX = AW / 5;
  obstacleY = AH / 2;

  // Create jitted functions with baked dimensions
  advectVel = makeAdvect(H, W, 1);
  advectMat = makeAdvect(AH, AW, W / AW);
  applyMatBoundary = makeApplyMatBoundary(AH);

  // Initialize fields
  {
    using ones = np.ones([H, W]);
    using vx = ones.mul(INFLOW_SPEED);
    using vy = np.zeros([H, W]);
    velocity = np.stack([vx, vy], 2);
  }
  pressure = np.zeros([H, W]);
  {
    using initMat = np.zeros([AH, AW]);
    material = applyMatBoundary(initMat);
  }

  initRenderer(canvas);

  // --- Mouse/touch handlers ---
  function getMousePos(e: MouseEvent): [number, number] {
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  function onMouseMove(e: MouseEvent) {
    lastMouseX = mouseX;
    lastMouseY = mouseY;
    [mouseX, mouseY] = getMousePos(e);
    if (movingObstacle) {
      obstacleX = mouseX;
      obstacleY = mouseY;
    }
  }

  function onMouseDown(e: MouseEvent) {
    const [mx, my] = getMousePos(e);
    mouseX = mx;
    mouseY = my;
    lastMouseX = mx;
    lastMouseY = my;
    const dx = mx - obstacleX;
    const dy = my - obstacleY;
    if (dx * dx + dy * dy < obstacleRad * obstacleRad) {
      movingObstacle = true;
      mouseDown = false;
    } else {
      mouseDown = true;
      movingObstacle = false;
    }
  }

  function onMouseUp() {
    mouseDown = false;
    movingObstacle = false;
  }

  function onTouchMove(e: TouchEvent) {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    lastMouseX = mouseX;
    lastMouseY = mouseY;
    mouseX = touch.clientX - rect.left;
    mouseY = touch.clientY - rect.top;
    if (movingObstacle) {
      obstacleX = mouseX;
      obstacleY = mouseY;
    }
  }

  function onTouchStart(e: TouchEvent) {
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const mx = touch.clientX - rect.left;
    const my = touch.clientY - rect.top;
    mouseX = mx;
    mouseY = my;
    lastMouseX = mx;
    lastMouseY = my;
    const dx = mx - obstacleX;
    const dy = my - obstacleY;
    if (dx * dx + dy * dy < obstacleRad * obstacleRad) {
      movingObstacle = true;
    } else {
      mouseDown = true;
    }
  }

  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("mouseleave", onMouseUp);
  canvas.addEventListener("touchmove", onTouchMove, { passive: false });
  canvas.addEventListener("touchstart", onTouchStart);
  canvas.addEventListener("touchend", onMouseUp);
  canvas.addEventListener("touchcancel", onMouseUp);

  // Start simulation loop
  running = true;
  simulate();

  // Return cleanup function
  return () => {
    running = false;
    canvas.removeEventListener("mousemove", onMouseMove);
    canvas.removeEventListener("mousedown", onMouseDown);
    canvas.removeEventListener("mouseup", onMouseUp);
    canvas.removeEventListener("mouseleave", onMouseUp);
    canvas.removeEventListener("touchmove", onTouchMove);
    canvas.removeEventListener("touchstart", onTouchStart);
    canvas.removeEventListener("touchend", onMouseUp);
    canvas.removeEventListener("touchcancel", onMouseUp);
    velocity?.dispose();
    pressure?.dispose();
    material?.dispose();
  };
}
