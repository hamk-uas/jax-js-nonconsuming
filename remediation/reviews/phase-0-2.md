# Review: phase 0, round 2

- **Range:** `12a4909..dbe29aa`; re-review of the three round-1 findings and the implementation
  response. The homepage remains excluded at the human's request.
- **Reviewer:** the separate, non-implementing review session (`c70303a7`).
- **Date:** 2026-11-08.
- **Verdict:** fail -- finding 2 is partially resolved, but module-initialization GPU errors still
  pass silently.
- **Human sign-off:** pending. This is the second failed round; the plan calls for a human decision
  before another implementation/re-review cycle.

No implementation files were changed in the original repository. Temporary reproduction files and a
query-count mutation were confined to a disposable clone. This report is not a human sign-off and
does not mark Phase 0 done.

## Exit criteria

| Criterion                                          | Status                                | Evidence                                                                                                                                                         |
| -------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reviewed working tree clean                        | Verified                              | `git status --short` was empty at `dbe29aa`, before adding this report.                                                                                          |
| Fresh-clone dependency installation and generators | Verified                              | `pnpm install --offline --frozen-lockfile` passed and ran the root build and website `svelte-kit sync` prepare scripts. Dependencies were installed, not linked. |
| Build, type-check and lint                         | Verified                              | `pnpm build && pnpm check && pnpm lint` passed in the fresh installed clone.                                                                                     |
| Default suite green                                | Verified                              | `pnpm test`: 91 files passed; 3314 tests passed, 14 skipped.                                                                                                     |
| Required GPU backends fail loudly                  | Verified for the selected profile     | Forcing the AMD profile on this NVIDIA/Intel machine failed both selected files with `Required backend(s) did not initialize: webgpu, webgl`.                    |
| Invalid GPU work fails the test or file            | Contradicted at module initialization | The same invalid QuerySet fails inside a test, but its real error event before the first test is only logged; the file passes on both GPUs.                      |
| Passing review and human sign-off                  | Not satisfied                         | The remaining error-capture gap keeps this report at fail; no human sign-off has been supplied.                                                                  |

## Round-1 findings

### 1. GPU selection by installed manifest -- resolved

[detectGpu](../../test/gpu-config.ts) now checks hardware PCI vendor IDs as well as driver-manifest
availability. Direct execution with injected AMD-only hardware and both Intel and Radeon manifests
selects `amd`, sets the Radeon driver, and requires `webgpu,webgl`. Intel-only hardware selects
Intel; absent sysfs does not restrict the loader. This server selects NVIDIA.

The missing-adapter negative check was performed with a temporary config calling `gpuConfig("amd")`,
running `test/profile-gpu.test.ts` and `test/devices.test.ts`. Both files failed during setup rather
than passing with backend skips.

### 2. Invalid GPU work accepted -- partially resolved

The profiling fix is correct:

- [webgpu.ts](../../src/backend/webgpu.ts), lines 330-331, limits profiling to 4096 queries and 2048
  passes.
- All seven [profiling tests](../../test/profile-gpu.test.ts) pass on NVIDIA and Intel.
- Changing the query-count constant back to 8192 in the disposable clone fails five of seven tests
  on each GPU, including the new nonzero-duration and truncation tests.
- An invalid QuerySet created inside a test fails through the new `afterEach` check.

**Remaining finding F2.1 -- P2: module-initialization GPU errors are missed.**

In [setup.ts](../../test/setup.ts), lines 51-63, the `uncapturederror` listener is installed by
`beginGpuErrorScopes()`. That function is first called from `beforeEach` at line 80. Test modules
and their top-level awaited initialization execute before that hook, so errors delivered during
module initialization are not added to `uncapturedGpuErrors`.

The implementation response and R21 say errors outside tests fail the next test or the file. The
following actual GPU-error reproduction contradicts that claim:

```ts
import { getWebGPUDevice } from "@hamk-uas/jax-js-nonconsuming";
import { expect, test } from "vitest";

const device = getWebGPUDevice();
const moduleError = new Promise<string>((resolve) => {
  device.addEventListener("uncapturederror", (event) => resolve(event.error.message), {
    once: true,
  });
});
device.createQuerySet({ type: "timestamp", count: 8192 });
device.queue.submit([]);
await device.queue.onSubmittedWorkDone();
const message = await moduleError;

test("module initialization already emitted a GPU validation error", () => {
  expect(message).toMatch(/8192.*4096/);
});
```

The executed fixture also included a five-second timeout guard. Run separately through
`scripts/gpu-test.sh run test/review-module-gpu-error.test.ts`, it returned exit code 0 on both
NVIDIA and Intel: one test passed while the backend logged
`Query count (8192) exceeds the maximum query count (4096)`.

After restoring the unmodified backend, running the fixture alongside the seven profiling tests
again returned exit code 0 with eight tests passed on each GPU. Thus the gap is independent of the
profiling fix. The normal full suites did not log WebGPU errors; the reproduction establishes that
the harness can still miss them, not that the normal suites contain an undiscovered invalid
dispatch.

The listener must cover device use before the first test hook, and a focused negative test must
prove that such an error fails the file. No implementation fix was made during this review.

### 3. Clean-checkout prerequisite -- resolved by the clarified procedure

The implementer's correction is accepted. The previous reproduction used linked dependencies and
therefore bypassed prepare generators. A fresh clone at `dbe29aa` with normal `pnpm install` passes
the listed build/check/test sequence; the explicit installation prerequisite is a valid
clarification, not a test tolerance or backend exception.

The website prepare script's silent `|| echo ''` remains recorded as Phase 1 debt. It is not a new
blocker in this re-review.

## Full-suite results

These runs used the unmodified `dbe29aa` clone after dependency installation, before adding
temporary reproduction fixtures.

| Suite      | Passed | Skipped | Failed |
| ---------- | ------ | ------- | ------ |
| Default    | 3314   | 14      | 0      |
| NVIDIA GPU | 3367   | 14      | 0      |
| Intel GPU  | 3363   | 14      | 4      |

The Intel failures are the same four recorded R24 transcendental-accuracy tests in
[numpy.test.ts](../../test/numpy.test.ts), at lines 2475 and 2505. They are unchanged and remain
Phase 1 work, not a new finding against these fixes.

## Rework and conclusion

The history-churn script's existing collection and measurement functions report three commits in
`12a4909..dbe29aa`, one source-changing commit, six added nonblank source lines, zero recent source
lines redone, corrective share 0, and follow-ups/change 0. Plotting dependencies were unavailable,
so the same metrics were computed without generating the figure. The metric excludes test/config/doc
changes, and newest-commit follow-ups are right-censored; this sample cannot establish a trend.

GPU selection, required-adapter enforcement, the profiling query limit, and the installation
procedure are verified improvements. Phase 0 still needs the human decision required after two
failed rounds: fix F2.1 before sign-off, or explicitly accept/defer that remaining error-capture
limitation. This report makes neither decision on the human's behalf.
