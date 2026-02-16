<script lang="ts">
  import { resolve } from "$app/paths";
  import { page } from "$app/state";

  const error = $derived(page.error!);
</script>

<svelte:head>
  <title
    >{page.status === 404 ? "Not Found" : "Error"} | jax-js-nonconsuming</title
  >
</svelte:head>

<div class="min-h-screen flex sm:items-center sm:justify-center px-4 py-8">
  <div class="flex flex-col sm:flex-row gap-x-8 gap-y-6">
    <a href={resolve("/")} class="block text-xl font-medium sm:py-1"
      >jax-js-nonconsuming</a
    >
    <div class="sm:border-l sm:border-gray-300 sm:pl-4 sm:py-1">
      <h1 class="text-2xl mb-5">{page.status} {error.message}</h1>
      <p class="text">
        {page.status === 404
          ? "We couldn't find what you were looking for!"
          : "Oops! Something went wrong."}
      </p>
      <p class="text">
        <a href={resolve("/")} class="text-gray-500 underline">Return home</a>
      </p>
    </div>
  </div>
</div>
