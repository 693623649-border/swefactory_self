"""Task-independent contract review; never built from held-out evaluation output."""

CONTRACT_REVIEW_VERSION = "public-contract-v1"
CONTRACT_REVIEW = """

Contract review (use only the supplied issue and current repository):
- Before editing, identify the issue's observable acceptance requirements. For any
  explicitly requested function, member, import path, CLI flag or configuration
  key, preserve its exact name, public location, arguments and defaults. Do not
  replace a requested public API with a differently named private helper.
- Inspect the target implementation, its callers and nearby existing tests.
  Distinguish explicit issue requirements from assumptions. Check return shape,
  None versus empty values, exceptions, side effects and compatibility with
  existing callers. Infer unspecified details from those sources; do not invent
  requirements or look for upstream patches or hidden grading tests.
- Make a minimal implementation covering every issue requirement, including
  supporting exports, CLI wiring and configuration where explicitly requested.
  Keep files unchanged when they are unrelated to those requirements.
- Verify in the actual repository environment. In addition to relevant existing
  tests, run small direct checks derived from the issue for imports/signatures,
  normal inputs and relevant empty/error cases. Prefer shell one-liners rather
  than modifying tracked tests, fixtures or snapshots. Distinguish a failed
  implementation from an unavailable dependency or a test invocation error.
- Before ending, review the diff against the requirements and the tests actually
  run after the last change. If verification fails, use the observed feedback to
  repair the implementation; never report a failing check as passed. Keep notes
  brief and act promptly rather than repeating a plan.
"""
