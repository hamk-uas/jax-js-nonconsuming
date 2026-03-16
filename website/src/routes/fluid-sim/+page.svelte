<script lang="ts">
  import { resolve } from "$app/paths";
  import { page } from "$app/state";
  import { onMount } from "svelte";
  import { startSimulation } from "./simulation";

  let canvas: HTMLCanvasElement;

  onMount(() => {
    let stop: (() => void) | undefined;
    startSimulation(canvas).then((fn) => (stop = fn));
    return () => stop?.();
  });
</script>

<main class="fluid-sim">
  <div class="pb-2 text-neutral-300 text-center">
    <h1 class="text-xl">
      Vortex Shedding <a
        class="text-lg"
        target="_blank"
        href="https://github.com/hamk-uas/jax-js-nonconsuming/blob/main/website/src/routes{page
          .route.id}/simulation.ts">(source)</a
      >
    </h1>
    <p class="text-sm text-neutral-400">
      Incompressible Navier-Stokes fluid simulation. Click and drag to apply
      force and move the obstacle.
    </p>
    <p class="text-sm text-neutral-400">
      Using <a href={resolve("/")}>jax-js-nonconsuming</a> on WebGPU. Based on
      <a target="_blank" href="https://github.com/amandaghassaei/VortexShedding"
        >VortexShedding</a
      > — all credit to Amanda Ghassaei.
    </p>
  </div>
  <canvas bind:this={canvas}></canvas>
</main>

<svelte:head>
  <style>
    body {
      background: #111;
    }
  </style>
</svelte:head>

<style lang="postcss">
  @reference "$app.css";

  .fluid-sim {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    height: 100vh;
    overflow: hidden;
    color: #eee;
    font-family: system-ui, sans-serif;
    padding: 8px;
  }

  a {
    @apply text-yellow-300/80;
    text-decoration: underline;
  }

  canvas {
    width: 100%;
    max-width: 1200px;
    flex: 1;
    min-height: 0;
    max-height: 800px;
    cursor: crosshair;
  }
</style>
