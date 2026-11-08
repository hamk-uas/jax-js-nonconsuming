import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";
import lucidePreprocess from "vite-plugin-lucide-preprocess";

const isDev = process.argv.includes("dev");
const repoName = process.env.GITHUB_REPOSITORY?.split("/")[1];
const inferredBasePath =
  process.env.BASE_PATH ??
  (repoName && !repoName.endsWith(".github.io") ? `/${repoName}` : "");
const isRootRelative = (path: string): path is `/${string}` =>
  path.startsWith("/");
if (inferredBasePath !== "" && !isRootRelative(inferredBasePath)) {
  throw new Error(
    `BASE_PATH must be empty or start with "/": ${inferredBasePath}`,
  );
}

export default defineConfig({
  plugins: [
    sveltekit({
      preprocess: vitePreprocess(),
      // Fallback is used for dynamic routes that aren't prerendered.
      // https://svelte.dev/docs/kit/adapter-static
      adapter: adapter({ fallback: "200.html" }),
      alias: { "$app.css": "src/app.css" },
      paths: { base: isDev ? "" : inferredBasePath },
    }),
    tailwindcss(),
    lucidePreprocess(),
    process.env.BASIC_SSL ? basicSsl() : null,
  ],
  optimizeDeps: {
    // https://github.com/vitejs/vite/issues/14609
    exclude: ["@rollup/browser", "onnxruntime-web"],
  },
  build: {
    // Increase chunk size warning limit for ML libraries.
    chunkSizeWarningLimit: 4000,
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
