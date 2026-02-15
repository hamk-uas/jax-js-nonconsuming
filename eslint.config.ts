import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import * as eslintImport from "eslint-plugin-import";
import globals from "globals";
import ts from "typescript-eslint";

import jaxJsPlugin from "./packages/eslint-plugin/src/index";

export default defineConfig([
  globalIgnores(["**/dist/", "docs/", "website/", "coverage/", "test/deno/"]),
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
      "jax-js/no-array-chain": "off",
      "jax-js/no-unnecessary-ref": "off",
      "jax-js/no-use-after-dispose": "off",
      "jax-js/require-using": "off",
    },
  },
  {
    files: [
      "src/**/*.{js,mjs,cjs,ts}",
      "test/**/*.{js,mjs,cjs,ts}",
      "packages/**/*.{js,mjs,cjs,ts}",
    ],
    rules: {
      "jax-js/no-array-chain": "off",
      "jax-js/no-unnecessary-ref": "warn",
      "jax-js/no-use-after-dispose": "warn",
      "jax-js/require-using": "warn",
    },
  },
]);
