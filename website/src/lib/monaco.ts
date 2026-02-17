import jaxTypesSrc from "@hamk-uas/jax-js-nonconsuming/dist/index.d.ts?raw";
import loadersTypesSrc from "@hamk-uas/jax-js-nonconsuming-loaders/dist/index.d.ts?raw";
import optaxTypesSrc from "@hamk-uas/jax-js-nonconsuming-optax/dist/index.d.ts?raw";
import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
// import cssWorker from "monaco-editor/esm/vs/language/css/css.worker?worker";
// import htmlWorker from "monaco-editor/esm/vs/language/html/html.worker?worker";
// import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";

import replBuiltinsSrc from "../routes/repl/repl-builtins.d.ts?raw";

self.MonacoEnvironment = {
  getWorker: function (_: string, label: string) {
    switch (label) {
      // case "json":
      //   return new jsonWorker();
      // case "css":
      // case "scss":
      // case "less":
      //   return new cssWorker();
      // case "html":
      // case "handlebars":
      // case "razor":
      //   return new htmlWorker();
      case "typescript":
      case "javascript":
        return new tsWorker();
      default:
        return new editorWorker();
    }
  },
};

monaco.typescript.typescriptDefaults.setCompilerOptions({
  target: monaco.typescript.ScriptTarget.ESNext,
  allowNonTsExtensions: true,
  moduleResolution: monaco.typescript.ModuleResolutionKind.NodeJs,
  module: monaco.typescript.ModuleKind.ESNext,
  typeRoots: ["node_modules/@types"],
  allowSyntheticDefaultImports: true,
  esModuleInterop: true,
});

monaco.typescript.typescriptDefaults.addExtraLib(
  jaxTypesSrc,
  "file:///node_modules/@hamk-uas/jax-js-nonconsuming/index.d.ts",
);
monaco.typescript.typescriptDefaults.addExtraLib(
  optaxTypesSrc,
  "file:///node_modules/@hamk-uas/jax-js-nonconsuming-optax/index.d.ts",
);
monaco.typescript.typescriptDefaults.addExtraLib(
  loadersTypesSrc,
  "file:///node_modules/@hamk-uas/jax-js-nonconsuming-loaders/index.d.ts",
);

// Global declarations for _BUILTINS available in the REPL.
monaco.typescript.typescriptDefaults.addExtraLib(
  replBuiltinsSrc,
  "file:///repl-builtins.d.ts",
);

export default monaco;
