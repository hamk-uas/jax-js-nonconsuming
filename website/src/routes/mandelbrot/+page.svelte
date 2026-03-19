<script lang="ts">
  import { defaultDevice, init } from "@hamk-uas/jax-js-nonconsuming";
  import { onMount } from "svelte";

  import {
    calculateMandelbrot,
    calculateMandelbrotForiLoop,
    calculateMandelbrotJitLoop,
    calculateMandelbrotScan,
    height,
    width,
  } from "./mandelbrot";

  let milliseconds = $state(0);

  onMount(async () => {
    await init("webgpu");
    defaultDevice("webgpu");
  });

  let canvas: HTMLCanvasElement;

  function renderMandelbrot(result: Int32Array) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;

    for (let i = 0; i < result.length; i++) {
      const value = 255 * (1 - result[i] / 100);
      data[i * 4] = value; // Red
      data[i * 4 + 1] = value; // Green
      data[i * 4 + 2] = value; // Blue
      data[i * 4 + 3] = 255; // Alpha
    }

    ctx.putImageData(imageData, 0, 0);
  }
</script>

<main class="p-4">
  <h1 class="text-2xl mb-2">mandelbrot in jax-js-nonconsuming</h1>

  <p class="mb-4">NumPy + GPU + JIT, in JavaScript!</p>

  <div class="flex flex-wrap gap-2 mb-4">
    <button
      onmousedown={async () => {
        const start = performance.now();
        const arr = calculateMandelbrot(100);
        const result = (await arr.data()) as Int32Array;
        arr.dispose();
        milliseconds = performance.now() - start;
        renderMandelbrot(result);
      }}
    >
      for(jit)
    </button>

    <button
      onmousedown={async () => {
        const start = performance.now();
        const arr = calculateMandelbrotJitLoop(100);
        const result = (await arr.data()) as Int32Array;
        arr.dispose();
        milliseconds = performance.now() - start;
        renderMandelbrot(result);
      }}
    >
      jit(for)
    </button>

    <button
      onmousedown={async () => {
        const start = performance.now();
        const arr = calculateMandelbrotScan(100);
        const result = (await arr.data()) as Int32Array;
        arr.dispose();
        milliseconds = performance.now() - start;
        renderMandelbrot(result);
      }}
    >
      jit(lax.scan)
    </button>

    <button
      onmousedown={async () => {
        const start = performance.now();
        const arr = calculateMandelbrotForiLoop(100);
        const result = (await arr.data()) as Int32Array;
        arr.dispose();
        milliseconds = performance.now() - start;
        renderMandelbrot(result);
      }}
    >
      jit(foriLoop)
    </button>
  </div>

  {#if milliseconds}
    <span class="text-sm">Computed in {milliseconds.toFixed(1)} ms</span>
  {/if}

  <canvas bind:this={canvas} {width} {height} class="my-8"></canvas>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border px-2 hover:bg-gray-100 active:scale-95;
  }
</style>
