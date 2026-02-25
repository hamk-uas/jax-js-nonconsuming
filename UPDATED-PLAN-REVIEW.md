# Review: `PREENCODED-MULTI-KERNEL-SCAN-PLAN.md`

Date: 2026-02-25  
Reviewer: GitHub Copilot (GPT-5.3-Codex)

## Findings (Ordered by Severity)

1. **High:** Phase 1 fusion expansion still has correctness risk for reduction/internal-dependency cases.
- The plan broadens fused eligibility to include reduction-related dependencies when "same-gidx" appears safe (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:224` to `PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:229`).
- This area needs a stricter formal admission predicate because false positives cause silent numerical corruption.
- Current mitigation is conservative fallback (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:789`), but the criteria are not yet formalized enough to guarantee conservative behavior.
- Recommendation: Add a dedicated "proof obligation" section with explicit structural rules for `AluExp` + `Reduction` consumer patterns and reject-by-default for ambiguous forms.

2. **High:** Nested preencoding is directionally correct, but symbolic/dynamic bounds are not fully specified.
- Recursive encoding examples assume concrete lengths (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:292`).
- The plan later states preencoded rejects symbolic length (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:742`), but this is not integrated into nested-loop examples and decision tables.
- Recommendation: Add an explicit compatibility matrix for outer/inner `{concrete,symbolic}` lengths and mandatory fallback behavior.

3. **Medium:** The document is much better balanced than earlier versions, but future primitive details are too deep for this execution plan.
- Sections for `while_loop`, `cond`, and `switch` are valuable, but very implementation-specific (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:546` onward).
- This dilutes immediate implementation focus (Phases 1–4).
- Recommendation: Keep only interface constraints here; move detailed lowering and AD design for future primitives to a separate RFC.

4. **Medium:** Structural trigger for full LoopPlan unification may be too late.
- The plan defers unification until a third primitive exists (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:688` to `PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:696`).
- But Phase 2 recursive encoding already introduces generalized loop orchestration complexity.
- Recommendation: Introduce minimal shared planner/executor interfaces now (without full type migration) to reduce near-term complexity risk.

5. **Medium:** Performance claims need stricter benchmark framing.
- Example estimate for Kalman smoother (~20ms) is useful but currently aspirational (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:326`).
- Recommendation: Mark such numbers as hypotheses and tie success criteria to explicit benchmark commands, hardware classes, and variance bounds.

6. **Low (Positive):** Housekeeping choices are strong and AEP-aligned.
- `classifyBodySteps()` extraction and moving assocScan execution out of `jit.ts` are clear debt-reduction wins (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:416`, `PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:452`).
- Implementation order is practical and incremental (`PREENCODED-MULTI-KERNEL-SCAN-PLAN.md:842`).

## Balance Assessment

Overall: **Mostly yes, this strikes a good balance**.

What is now strong:
- Keeps no-regression intent for current compiled-loop fast paths.
- Adds a practical Tier-2 route for nested/mixed bodies.
- Includes structural cleanup, not only new feature paths.

What should be tightened before implementation begins:
- Formal correctness constraints for Phase 1 fusion admission.
- Explicit nested symbolic-length policy.
- Scope discipline for future-control-flow sections.

## Open Questions

1. What exact structural predicate defines same-gidx-safe reduction consumption?
2. What is the required fallback behavior for nested symbolic lengths?
3. Which hard limits trigger chunking/fallback for nested preencoded command buffers?
4. Should minimal loop-planner/executor interface unification move into Phase 4 to de-risk recursion now?

## Recommendation

Proceed with this v3 plan after adding:
- A formal Phase 1 fusion safety contract,
- A nested-length compatibility matrix,
- Benchmark protocol language for all headline performance goals.
