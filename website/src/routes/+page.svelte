<script lang="ts">
  import { browser } from "$app/environment";
  import { resolve } from "$app/paths";

  import { ArrowUpRight, ExternalLinkIcon } from "@lucide/svelte";

  import logo from "$lib/assets/logo-nonconsuming.svg";
  import MatmulPerfDemo from "$lib/homepage/MatmulPerfDemo.svelte";
  import EmbeddedRepl from "$lib/repl/EmbeddedRepl.svelte";

  const installText = {
    npm: `npm install github:hamk-uas/jax-js-nonconsuming`,
    pin: `npm install github:hamk-uas/jax-js-nonconsuming#v0.2.0`,
  };

  let installMode = $state<"npm" | "pin">("npm");

  const links = [
    {
      title: "GitHub Repository",
      href: "https://github.com/hamk-uas/jax-js-nonconsuming",
      description:
        "Get started with jax-js-nonconsuming and check out the tutorial.",
    },
    {
      title: "REPL",
      href: resolve("/repl"),
      description: "Try out the library in this browser-based REPL.",
    },
    {
      title: "API Reference",
      href: "https://hamk-uas.github.io/jax-js-nonconsuming/docs/",
      description: "View the full API documentation.",
    },
    {
      title: "MobileCLIP2 Inference",
      href: resolve("/mobileclip"),
      description:
        "Generate embeddings for books and search them in real time.",
    },
    {
      title: "Kyutai Pocket TTS",
      href: resolve("/tts"),
      description: "Voice cloning AI model that runs in your browser.",
    },
    {
      title: "MNIST Training",
      href: resolve("/mnist"),
      description: "Demo of training a neural network on MNIST.",
    },
  ];
</script>

<svelte:head>
  <title>jax-js-nonconsuming (Fork) – ML for the web</title>
</svelte:head>

<!-- Header -->
<header
  class="px-6 py-4 flex items-center justify-between max-w-screen-xl mx-auto font-tiktok gap-6"
>
  <div class="flex items-center gap-3 shrink-0">
    <a href={resolve("/")}>
      <img src={logo} alt="jax-js-nonconsuming logo" class="h-8" />
    </a>
  </div>
  <nav class="flex items-center gap-6">
    <a href={resolve("/repl")} class="hidden sm:block hover:text-primary"
      >REPL</a
    >
    <a
      rel="external"
      href="https://hamk-uas.github.io/jax-js-nonconsuming/docs/"
      class="hover:text-primary">Docs</a
    >
    <a
      href="https://github.com/hamk-uas/jax-js-nonconsuming"
      target="_blank"
      class="bg-primary/15 hover:bg-primary/25 px-4 py-1 rounded-full"
    >
      GitHub
      <ExternalLinkIcon size={16} class="inline-block mb-1 ml-0.5 opacity-60" />
    </a>
  </nav>
</header>

<main class="font-tiktok">
  <!-- Hero section -->
  <section class="px-6 py-14 md:py-20 max-w-screen-xl mx-auto">
    <div class="grid md:grid-cols-[5fr_3fr] gap-x-12 gap-y-16">
      <div class="lg:py-8">
        <h1 class="text-3xl sm:text-4xl mb-6 leading-tight max-w-2xl">
          jax-js-nonconsuming (fork) is <span class="hidden sm:inline"
            >a machine learning</span
          ><span class="sm:hidden">an ML</span> library and compiler for the web
        </h1>
        <div
          class="mb-8 p-4 bg-amber-50 border-l-4 border-amber-400 text-amber-900 rounded-r shadow-sm"
        >
          <p class="font-bold mb-1">Fork Notice</p>
          <p class="text-sm">
            This is a <strong>non-consuming ownership fork</strong> of
            <a
              href="https://github.com/ekzhang/jax-js"
              class="underline hover:text-amber-700"
              target="_blank"
              rel="noopener noreferrer">ekzhang/jax-js</a
            >. Operations leave inputs alive (no manual
            <code class="bg-amber-100 px-1 rounded">.ref</code> needed), designed
            for teams familiar with NumPy or MATLAB.
          </p>
        </div>
        <p class="text-lg text-gray-700 leading-snug mb-8 max-w-2xl">
          High-performance WebGPU and WebAssembly kernels in JavaScript. Run
          neural networks, image algorithms, simulations, and numerical code,
          all JIT compiled in your browser.
        </p>

        <!-- Add to project box -->
        <div class="bg-primary/5 rounded-lg p-4">
          <h2 class="text-xl font-medium mb-1.5">
            Add jax-js-nonconsuming to your project
          </h2>
          <p class="text-gray-600 text-sm mb-4">
            Zero dependencies. All major browsers. Install from
            <button
              class="enabled:underline"
              onclick={() => (installMode = "npm")}
              disabled={installMode === "npm"}>GitHub</button
            >
            or
            <button
              class="enabled:underline"
              onclick={() => (installMode = "pin")}
              disabled={installMode === "pin"}>pin a tag</button
            >.
          </p>
          <div
            class="bg-primary/5 border-1 border-primary rounded-lg px-3 py-2 font-mono whitespace-pre-wrap"
          >
            <span class="text-primary/50 select-none">&gt;&nbsp;</span
            >{installText[installMode]}
          </div>
        </div>
      </div>

      <!-- Performance Chart -->
      <MatmulPerfDemo />
    </div>
  </section>

  <!-- Explainer section -->
  <section class="mx-auto max-w-screen-xl my-8 sm:px-6 hidden">
    <div class="sm:rounded-xl bg-primary/5 px-8 py-8">
      <div class="mx-auto max-w-2xl">
        <h2 class="text-xl font-medium text-center mb-6">
          Like JAX and PyTorch in your browser
        </h2>

        <p class="mb-6">
          jax-js-nonconsuming is an end-to-end ML library inspired by JAX, but
          in pure JavaScript:
        </p>

        <ul
          class="space-y-2 pl-4 mb-6 list-disc list-inside marker:text-gray-400"
        >
          <li>Runs completely client-side (Chrome, Firefox, iOS, Android).</li>
          <li>
            Has close <a
              href="https://github.com/hamk-uas/jax-js-nonconsuming/blob/main/FEATURES.md"
              target="_blank"
              class="underline hover:text-primary">API compatibility</a
            > with NumPy/JAX.
          </li>
          <li>Is written from scratch, with zero external dependencies.</li>
        </ul>

        <p class="mb-6">
          jax-js-nonconsuming is likely the most portable GPU ML framework,
          since it runs anywhere a browser can run. It's also simple but
          optimized, including a lightweight compiler that translates your
          high-level operations into WebGPU and WebAssembly kernels.
        </p>

        <p>
          The goal of jax-js-nonconsuming is to make numerical code accessible
          and deployable to everyone, so compute-intensive apps can run fast and
          locally on consumer hardware.
        </p>
      </div>
    </div>
  </section>

  <!-- Live Editor section -->
  <section class="px-6 py-12 max-w-screen-xl mx-auto">
    <h2 class="text-xl mb-2">Try it out!</h2>

    <p class="mb-4 text-sm text-gray-600">
      This is a live editor, the code is running in your browser{browser &&
      navigator.gpu
        ? " with WebGPU"
        : ""}.
    </p>

    <EmbeddedRepl
      initialText={String.raw`import { grad, numpy as np, vmap } from "@jax-js-nonconsuming/jax";

const f = (x: np.Array) => {
  using sq = x.mul(x);
  using s = sq.sum();
  return np.sqrt(s);
};

using x = np.array([1, 2, 3, 4]);

using y0 = f(x);
console.log(y0.js());

using y1 = grad(f)(x);
console.log(y1.js());

using y2 = vmap(grad(np.square))(x);
console.log(y2.js());
`}
    />
  </section>

  <!-- Learn More section -->
  <section class="px-6 py-16 max-w-screen-xl mx-auto">
    <h2 class="text-xl mb-6">Learn more</h2>

    <div class="grid sm:grid-cols-3 gap-x-6 md:gap-x-8 gap-y-4">
      {#each links as { title, href, description }}
        <a
          {href}
          class="bg-primary/5 hover:bg-primary/15 transition-colors p-4 rounded-lg"
        >
          <h3 class="mb-2">
            {title}
            <ArrowUpRight size={18} class="inline-block text-gray-400 mb-px" />
          </h3>
          <p class="text-sm text-gray-600">
            {description}
          </p>
        </a>
      {/each}
    </div>
  </section>
</main>
