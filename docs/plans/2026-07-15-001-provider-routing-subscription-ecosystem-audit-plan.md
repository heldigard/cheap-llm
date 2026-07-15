# Provider Routing and Subscription Ecosystem Audit

Date: 2026-07-15
Status: ready for implementation

## Objective

Audit and strengthen `cheap-llm` as the low-cost signal-distillation layer used
by the surrounding CLI ecosystem. Cover current OpenRouter, DeepInfra, ZenMux,
DeepSeek, and subscription-backed routes without weakening the project's
advisory-only or secret-scrubbing boundaries.

## Constraints

- Preserve the public SemVer contract unless a justified additive change is
  required.
- Keep local/private inference first by default.
- Never read, print, persist, or migrate credential values.
- Never introduce an implicit PAYG route; route provenance and cost class must
  be inspectable.
- Do not use weak models for code writing, security decisions, migrations, or
  other authority-bearing work.
- Preserve existing user changes in project and home configuration.

## Plan

1. Inventory package architecture, tests, model/provider registry, environment
   contracts, benchmark behavior, ecosystem shims, and known memory decisions.
2. Verify provider APIs, current model identifiers, pricing/capability metadata,
   and subscription routes against authoritative current documentation.
3. Define explicit routing policy for local, subscription-seat, direct-provider,
   and aggregator PAYG
   providers, including timeouts, fallbacks, structured output, and context
   limits.
4. Implement the smallest coherent set of code, documentation, test, and
   surrounding-configuration changes.
5. Review for contract drift, accidental spend, secret leakage, cache isolation,
   unsafe output handling, and cross-CLI inconsistencies.
6. Run targeted and full tests, Ruff, types, security scans, `diff --check`,
   ecosystem shim checks, and offline probe/route diagnostics.
7. Update project memory with durable routing decisions and remaining live-test
   steps; do not perform billable live calls merely to validate wiring.

## Acceptance Criteria

- Provider and model selection is current, deterministic, observable, and
  covered by offline tests.
- Subscription-backed capacity is preferred where policy says it is free within
  an existing subscription; PAYG requires an explicit route or existing
  documented fallback policy.
- A provider outage, unavailable credential, malformed response, or model
  retirement fails over safely without exposing secret material.
- Every cloud request has bounded input/output, timeout, response size, and
  sanitized error reporting.
- Existing consumers and the global compatibility shim continue to pass their
  contract tests.

## Deepened Findings

1. `DeepInfra` is registered and usable only for selected pinned models, but
   it has no probe URL and is absent from the default cascade. The CLI can show
   a configured key while omitting provider health entirely.
2. Provider probes issue unauthenticated `HEAD` requests. Several OpenAI-style
   model-list endpoints require a bearer token or do not implement `HEAD`, so a
   healthy provider can be reported as unreachable.
3. Direct DeepSeek V4 defaults to thinking mode. `cheap-llm` currently sends no
   toggle, paying reasoning latency/tokens for short signal-distillation work.
   Official API guidance supports `{"thinking":{"type":"disabled"}}`.
4. Direct DeepSeek cache accounting assumes `fresh_input / 10`. Current official
   prices are model-specific: V4 Flash is $0.0028 cached versus $0.14 fresh;
   V4 Pro is $0.003625 cached versus $0.435 fresh. V4 Pro is missing from the
   local price table entirely.
5. JSON validity is requested only through prompt text. Direct DeepSeek and
   DeepInfra officially support `response_format={"type":"json_object"}`.
6. OpenRouter officially supports provider sorting by price. The current
   request delegates to default load balancing and therefore does not enforce
   the project's low-cost intent within a model slug.
7. Documentation labels a DeepSeek API key as “BYOK $0”/“truly free”. API keys
   consume topped-up or granted balance; this is PAYG capacity, not a CLI-seat
   subscription. Subscription workers already belong to `cli-orchestration`
   and `fusion-local` and must remain a separate authority lane.
8. The surrounding `model-drift-check.py` still points at the deleted
   `~/cheap-llm/cheap_llm.py` monolith, so its cheap-llm drift checks are stale.

## Implementation Decisions

- Add an additive `cloud_provider` public option and `--cloud-provider` CLI
  flag. When explicit, try only that provider for the pinned cloud model; do
  not silently cross a billing/trust boundary. Bump the package minor version.
- Add `--route-plan` as a no-completion diagnostic that prints ordered routes,
  credential availability booleans, and billing class without exposing values.
- Authenticate bounded `GET /models` probes and add DeepInfra model-list
  health. Sanitize probe errors through the existing public-error boundary.
- Propagate the private `require_json` bit into transports. Use native JSON mode
  where officially supported, while retaining prompt validation and cascade
  fallback.
- Disable DeepSeek thinking for this advisory preprocessor and use exact
  model-specific fresh/cache/output prices.
- Sort OpenRouter endpoints by price. Keep the benchmark-selected model order;
  model replacement still requires the existing five-call stability protocol.
- Correct cost/subscription terminology in project and active surrounding docs.
- Fix the drift checker path only after inspecting its current diff; do not
  overwrite unrelated home-config work.

## Validation Matrix

- Unit: cascade shape, explicit provider isolation, DeepSeek payload/cost,
  response-format propagation, authenticated probe, route-plan output.
- Contract: SemVer, additive signature, uniform result shape, shim exports.
- Integration: installed console, compatibility shim, cli-orchestration doctor
  probe parser, fusion-local judge preflight.
- Static/security: Ruff, Mypy, Pyright, Gitleaks, Semgrep/codescan, build,
  `git diff --check`.
- Network: provider catalog/probe only; no billable completion unless explicitly
  requested by the user.
