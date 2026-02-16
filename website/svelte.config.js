import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

const isDev = process.argv.includes("dev");
const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1];
const inferredBasePath =
  process.env.BASE_PATH ??
  (repoName && !repoName.endsWith(".github.io") ? `/${repoName}` : "");

/** @type {import("@sveltejs/kit").Config} */
const config = {
  preprocess: vitePreprocess(),

  kit: {
    // Fallback is used for dynamic routes that aren't prerendered.
    // https://svelte.dev/docs/kit/adapter-static
    adapter: adapter({ fallback: "200.html" }),
    alias: {
      "$app.css": "src/app.css",
    },
    paths: {
      base: isDev ? "" : inferredBasePath,
    },
  },
};

export default config;
