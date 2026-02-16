<script lang="ts">
  const {
    title = "jax-js-nonconsuming – Open-source web ML framework",
    description = "Machine learning library and compiler for the web, written in pure JavaScript. Run high-performance WebAssembly and WebGPU kernels.",
  }: { title?: string; description?: string } = $props();
</script>

<svelte:head>
  <meta name="description" content={description} />

  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="jax-js-nonconsuming" />
  <meta property="og:title" content={title} />
  <meta property="og:description" content={description} />
  <!-- Needs to be an absolute URL, so this is hardcoded. -->
  <meta
    property="og:image"
    content="https://hamk-uas.github.io/jax-js-nonconsuming/images/social-image.png"
  />
  <meta property="twitter:card" content="summary_large_image" />
</svelte:head>
