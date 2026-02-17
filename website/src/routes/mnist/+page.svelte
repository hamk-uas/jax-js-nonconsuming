<script lang="ts">
  import {
    blockUntilReady,
    defaultDevice,
    init,
    numpy as np,
    tree,
  } from "@hamk-uas/jax-js-nonconsuming";
  import pThrottle from "p-throttle";
  import { onMount } from "svelte";

  import LineChart from "$lib/chart/LineChart.svelte";

  import {
    ConvNet,
    MLP,
    type ModelType,
    type Params,
    runInference,
  } from "./models";
  import { train, type TrainMetric, type TestMetric } from "./training";

  let logs = $state<string[]>([]);

  function log(msg: string) {
    logs.push(msg);
    console.log(msg);
  }

  // Training metrics
  let trainMetrics = $state<TrainMetric[]>([]);
  let testMetrics = $state<TestMetric[]>([]);

  // Training and inference state
  let latestParams: Params | null = null;
  let probs: number[] = $state.raw([]);
  let running = $state(false);
  let stopping = false;

  // Settings
  let learningRate = $state(0.005);
  let logLearningRate = $state(Math.log10(0.005));
  let showSettings = $state(false);

  $effect(() => {
    learningRate = parseFloat(Math.pow(10, logLearningRate).toFixed(6));
  });
  $effect(() => {
    logLearningRate = Math.log10(learningRate);
  });

  let Model: ModelType = undefined!; // Initialized below.
  let batchSize: number = undefined!;

  let selectedModel = $state("MLP");
  // svelte-ignore state_referenced_locally
  changeModelType(selectedModel);

  function changeModelType(modelType: string) {
    if (running) return;
    switch (modelType) {
      case "MLP":
        Model = MLP;
        batchSize = 1000;
        break;
      case "ConvNet":
        Model = ConvNet;
        batchSize = 250;
        break;
      default:
        throw new Error(`Unknown model type: ${modelType}`);
    }
    tree.dispose(latestParams);
    latestParams = null;
  }

  async function run() {
    running = true;
    stopping = false;
    logs = [];
    trainMetrics = [];
    testMetrics = [];

    tree.dispose(latestParams);
    latestParams = null;

    try {
      await train(
        { model: Model, learningRate, batchSize },
        {
          log,
          onTrainBatch(metric) {
            trainMetrics.push(metric);
          },
          onTestEval(metric) {
            testMetrics.push(metric);
          },
          onParamsUpdate(params) {
            if (latestParams !== params) tree.dispose(latestParams);
            latestParams = params;
          },
          onEpochEnd() {
            // Retrigger the inference demo if the user has drawn something.
            if (hasDrawn) inferenceDemo();
          },
          shouldStop() {
            return stopping;
          },
        },
      );
    } finally {
      running = false;
    }
  }

  function stop() {
    stopping = true;
  }

  onMount(async () => {
    await init("webgpu");
    defaultDevice("webgpu");

    ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });

  const inferenceDemo = pThrottle({ limit: 0, interval: 30 })(async () => {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // First, construct a 784-dimensional vector from the image data.
    const ar = new Float32Array(784);
    for (let i = 0; i < 28; i++) {
      for (let j = 0; j < 28; j++) {
        for (let l = i * 10; l < (i + 1) * 10; l++) {
          for (let k = j * 10; k < (j + 1) * 10; k++) {
            const idx = (l * 280 + k) * 4;
            const r = imgData.data[idx];
            const g = imgData.data[idx + 1];
            const b = imgData.data[idx + 2];
            ar[i * 28 + j] += (1 - (r + g + b) / 3 / 255) / 100;
          }
        }
      }
    }

    if (latestParams === null) {
      log("No model available for inference. Train the model first.");
      return;
    }
    probs = await runInference(Model, latestParams, ar);
  });

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let hasDrawn = $state(false);
  let drawing = false;
  let lastPos = [0, 0];
  const lineWidth = 28;

  function coords(event: PointerEvent): [number, number] {
    const rect = canvas.getBoundingClientRect();
    return [
      (event.offsetX / rect.width) * canvas.width,
      (event.offsetY / rect.height) * canvas.height,
    ];
  }

  function drawStart(event: PointerEvent) {
    event.preventDefault();
    const [x, y] = coords(event);
    drawing = true;
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.ellipse(x, y, lineWidth / 2, lineWidth / 2, 0, 0, Math.PI * 2);
    ctx.fill();
    lastPos = [x, y];
    hasDrawn = true;
    inferenceDemo();
  }

  function drawMove(event: PointerEvent) {
    if (!drawing) return;
    event.preventDefault();
    const [x, y] = coords(event);
    ctx.beginPath();
    ctx.moveTo(lastPos[0], lastPos[1]);
    ctx.lineTo(x, y);
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.stroke();
    lastPos = [x, y];
    inferenceDemo();
  }

  function drawEnd() {
    drawing = false;
  }

  function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
    probs = [];
  }
</script>

<svelte:head>
  <title>mnist + jax-js-nonconsuming</title>
</svelte:head>

<main class="p-4">
  <section class="max-w-3xl">
    <h1 class="text-2xl mb-4">mnist + jax-js-nonconsuming</h1>

    <p class="mb-4">
      Let's train a neural network to classify MNIST digits, in your browser
      with <code>jax-js-nonconsuming</code>.
    </p>

    <p class="mb-4">
      The model is a 3-layer MLP or 4-layer convolutional neural network trained
      with Adam. Each epoch has 60 (MLP) or 240 (ConvNet) randomized batches,
      with 60,000 images in total in the train set.
    </p>

    <p class="mb-4 text-sm">
      Note: This demo requires a <a
        class="underline"
        target="_blank"
        href="https://browserleaks.com/webgpu">WebGPU</a
      >-enabled browser. Works best on Chrome.
    </p>

    <div class="mb-8">
      <div class="flex gap-2">
        <select
          bind:value={selectedModel}
          onchange={() => changeModelType(selectedModel)}
          disabled={running}
        >
          <option value="MLP">MLP</option>
          <option value="ConvNet">ConvNet</option>
        </select>
        <button
          onclick={() => (showSettings = !showSettings)}
          class="text-sm flex items-center gap-1"
        >
          Settings
          <span
            class="transform transition-transform {showSettings
              ? 'rotate-180'
              : ''}"
            style="font-size: 10px;">▼</span
          >
        </button>
        {#if !running}
          <button onclick={run}>Run</button>
        {:else}
          <button onclick={stop}>Stop</button>
        {/if}
      </div>

      {#if showSettings}
        <div class="mt-2 p-3 border rounded bg-gray-50">
          <div class="flex items-center gap-3 mb-2">
            <label for="learning-rate-slider" class="font-semibold text-sm"
              >Learning rate:</label
            >
            <input
              id="learning-rate-slider"
              type="range"
              min="-3"
              max="-2"
              step="0.01"
              bind:value={logLearningRate}
              class="w-32"
              disabled={running}
            />
            <input
              type="number"
              min="0.001"
              max="0.01"
              step="any"
              bind:value={learningRate}
              class="w-24 px-2 py-1 border rounded text-sm"
              aria-label="Learning rate numerical input"
              disabled={running}
            />
          </div>
          <div class="flex justify-end">
            <button
              onclick={() => {
                learningRate = 0.005;
              }}
              disabled={running}
            >
              Reset to default
            </button>
          </div>
        </div>
      {/if}
    </div>
  </section>

  <div class="grid md:grid-cols-2 xl:grid-cols-3 gap-4 my-6">
    <div class="h-[220px] border border-gray-400 rounded">
      <LineChart
        title="Train Loss"
        data={trainMetrics}
        x="iteration"
        y="loss"
      />
    </div>
    <div class="h-[220px] border border-gray-400 rounded">
      <LineChart
        title="Test Loss & Accuracy"
        data={testMetrics}
        x="epoch"
        y={["loss", "acc"]}
      />
    </div>
    <div class="h-[220px] border border-gray-400 rounded">
      <div class="flex flex-col h-full">
        <p class="shrink-0 text-sm text-center my-1">Inference Demo</p>
        <div class="grow flex px-2 pb-2 min-h-0">
          <div
            class="relative aspect-square h-full border-4 border-gray-200 rounded-md"
          >
            <canvas
              width="280"
              height="280"
              class="w-full h-full"
              onpointerdown={drawStart}
              onpointermove={drawMove}
              onpointerleave={drawEnd}
              onpointerup={drawEnd}
              bind:this={canvas}
            ></canvas>

            {#if hasDrawn}
              <button
                class="absolute bottom-1 right-1"
                onclick={(event) => {
                  event.stopPropagation();
                  clearCanvas();
                }}>Clear</button
              >
            {:else}
              <p
                class="absolute top-18 left-6 -rotate-15 animate-bounce italic text-gray-400 pointer-events-none"
              >
                draw a digit here!
              </p>
            {/if}
          </div>
          <div class="grow ml-2">
            {#if probs.length > 0}
              <p class="text-xs font-bold">Probabilities:</p>
              {#each probs as prob, i}
                <div class="flex items-center text-xs tabular-nums">
                  <span class="w-4 text-right">{i}:</span>
                  <span
                    class="ml-1 bg-gray-400 h-3 rounded-sm"
                    style:width="{72 * prob}%"
                  ></span>
                  <span class="ml-2">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              {/each}
            {/if}
          </div>
        </div>
      </div>
    </div>
  </div>

  <div
    class="font-mono text-sm rounded bg-gray-900 px-4 py-2 h-[600px] overflow-y-scroll mt-8"
  >
    {#each logs as log}
      <div class="text-white whitespace-pre-wrap">{log}</div>
    {/each}
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border rounded px-2 hover:bg-gray-100 active:scale-95;
  }

  select {
    @apply border rounded px-1 text-sm;
  }
</style>
