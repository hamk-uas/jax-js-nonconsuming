<script lang="ts">
  import "../../src/polyfills"; // Safari: Symbol.dispose + SuppressedError
  import Seo from "$lib/common/Seo.svelte";
  import "../app.css";

  let { children } = $props();
</script>

<Seo />

{@render children()}
