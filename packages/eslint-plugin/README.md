# @jax-js/eslint-plugin (in-repo)

This package contains jax-js ownership lint rules for the non-consuming model.

## Rules

- `jax-js/require-using`
- `jax-js/no-use-after-dispose`
- `jax-js/no-unnecessary-ref`
- `jax-js/no-array-chain`

## Workspace wiring

The plugin is wired in root `eslint.config.ts` as plugin name `jax-js`.

Current rollout in this repo:

- `jax-js/require-using`: `warn` on `src/**`, `packages/**`
- `jax-js/no-unnecessary-ref`: `warn` on `src/**`, `packages/**`
- `jax-js/no-use-after-dispose`: `warn` on `src/**`, `packages/**`
- `jax-js/no-array-chain`: `off`
- all `jax-js/*` rules: `off` on `test/**`

Enable them explicitly in local config/CI as needed.
