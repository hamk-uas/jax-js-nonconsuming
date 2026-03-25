import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import * as eslintImport from "eslint-plugin-import";
import globals from "globals";
import ts from "typescript-eslint";

import jaxJsPlugin from "./packages/eslint-plugin/src/index";

export default defineConfig([
  globalIgnores([
    "**/dist/",
    "docs/",
    "website/",
    "coverage/",
    "tmp/",
    "*.config.mjs",
    "*.cjs",
    "apply_shader_cache.mjs",
  ]),
  js.configs.recommended,
  ts.configs.recommendedTypeChecked,
  {
    files: ["**/*.{js,mjs,cjs,ts}"],
    languageOptions: { globals: globals.browser },
  },
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    files: ["scripts/**/*.mjs"],
    ...ts.configs.disableTypeChecked,
  },
  {
    files: ["scripts/**/*.mjs"],
    rules: {
      "@typescript-eslint/consistent-type-exports": "off",
    },
  },
  {
    plugins: { import: eslintImport, "jax-js": jaxJsPlugin },
    rules: {
      "@typescript-eslint/consistent-type-exports": "error",
      "@typescript-eslint/no-array-constructor": "off",
      "@typescript-eslint/no-empty-object-type": "off",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-floating-promises": "off",
      "@typescript-eslint/no-redundant-type-constituents": "off",
      "@typescript-eslint/no-this-alias": "off",
      "@typescript-eslint/no-unnecessary-type-assertion": "off",
      "@typescript-eslint/no-unsafe-argument": "off",
      "@typescript-eslint/no-unsafe-assignment": "off",
      "@typescript-eslint/no-unsafe-call": "off",
      "@typescript-eslint/no-unsafe-member-access": "off",
      "@typescript-eslint/no-unsafe-return": "off",
      "@typescript-eslint/no-unused-expressions": "off",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          varsIgnorePattern: "^_",
          argsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
      "@typescript-eslint/require-await": "off",
      "@typescript-eslint/restrict-template-expressions": "off",
      "@typescript-eslint/unbound-method": "off",
      "import/newline-after-import": "warn",
      "import/order": [
        "warn",
        {
          alphabetize: {
            order: "asc",
          },
          groups: ["builtin", "external"],
          "newlines-between": "always",
        },
      ],
      "prefer-const": ["warn", { destructuring: "all" }],
      "sort-imports": [
        "warn",
        {
          allowSeparatedGroups: true,
          ignoreCase: true,
          ignoreDeclarationSort: true,
          ignoreMemberSort: false,
          memberSyntaxSortOrder: ["none", "all", "multiple", "single"],
        },
      ],
      "jax-js/no-array-chain": "warn",
      "jax-js/no-nested-array-leak": "off",
      "jax-js/no-dispose-then-reassign-param": "warn",
      "jax-js/no-make-disposable-alias": "warn",
      "jax-js/require-retained-release": "warn",
      "jax-js/require-try-finally-symmetry": "warn",
      "jax-js/require-wrapper-dispose-symmetry": "warn",
      "jax-js/no-unnecessary-ref": "warn",
      "jax-js/no-use-after-dispose": "warn",
      "jax-js/require-using": "warn",
      "jax-js/require-scan-result-dispose": "warn",
    },
  },
  {
    files: ["src/**/*.{js,mjs,cjs,ts}", "packages/**/*.{js,mjs,cjs,ts}"],
    ...(jaxJsPlugin.configs!.invariance as any),
  },
  // Internal framework code manipulates Slots, shapes, Jaxprs — not np.Arrays.
  // Disable require-using and no-nested-array-leak (AluExp false positives).
  {
    files: [
      "src/frontend/**/*.ts",
      "src/library/**/*.ts",
      "src/backend/**/*.ts",
      "src/tuner.ts",
      "src/*.ts",
    ],
    ignores: ["**/*.test.ts"],
    rules: {
      "jax-js/require-using": "off",
      "jax-js/no-nested-array-leak": "off",
    },
  },
  // Also disable for .test.ts files inside src/ (they test internals)
  {
    files: ["src/**/*.test.ts"],
    rules: {
      "jax-js/require-using": "off",
    },
  },
  // ONNX operators always run in compiled context; intermediates are
  // managed by the JIT compiler.
  {
    files: ["packages/onnx/**/*.ts"],
    rules: {
      "jax-js/no-nested-array-leak": "off",
    },
  },
  // Internal framework code must not import zeros()/ones() — they wrap
  // fullInternal() which is the correct primitive for tracing contexts.
  // Use fullInternal() or a local zerosInternal() helper instead.
  {
    files: [
      "src/frontend/**/*.ts",
      "src/library/**/*.ts",
      "src/backend/**/*.ts",
    ],
    ignores: ["**/*.test.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "./array",
              importNames: ["zeros", "ones"],
              message:
                "Use fullInternal() instead. zeros()/ones() are public API wrappers not intended for internal tracing contexts.",
            },
            {
              name: "../frontend/array",
              importNames: ["zeros", "ones"],
              message:
                "Use fullInternal() instead. zeros()/ones() are public API wrappers not intended for internal tracing contexts.",
            },
          ],
        },
      ],
    },
  },
  {
    files: ["test/**/*.{js,mjs,cjs,ts}"],
    ...(jaxJsPlugin.configs!.invariance as any),
  },
  {
    files: ["scripts/**/*.mjs"],
    languageOptions: { globals: globals.node },
  },
  {
    files: ["scripts/**/*.mjs"],
    rules: {
      "@typescript-eslint/consistent-type-exports": "off",
    },
  },
]);
