# Roadmap: Named Production Gaps

Nine selected product and measurement gaps stand between this prototype and operator-grade tooling for an LLM-in-marketing stack. They are numbered by conceptual validity dependency, not chronological implementation order, and grouped by what each one actually requires.

**The dependency claim, stated once:** prompt and run provenance should be recorded from the first generation; an unlogged run cannot be reconstructed. Stable segments are not prerequisites for logging or per-output validation. They are prerequisites for interpreting comparisons across prompt versions. If "high intent" changes definition between run A and run B, a difference in output cannot be attributed to the prompt — the comparison is uninterpretable. The semantic layer provides the versioned cohort definition that makes comparative evaluation valid.

Everything below is **designed, not built**. What is implemented today is described in [architecture-vision.md](architecture-vision.md).

---

## Tier 1 — buildable on synthetic data

### 1. Unified semantic layer addresses uninterpretable comparison

Categorize each retrieved profile into a **named segment** with a stored, human-readable definition — for example "high intent, professional, desktop, price-checking" rather than an opaque cluster id.

Two things follow, and the distinction matters:

- **Generation stays per person.** The model already tailors to whichever individual profile is retrieved; segments do not constrain that, and they should not.
- **Comparative evaluation happens per segment.** Checking a single output does not need one: a deterministic validator and a per-output review can run from the first generation. But a single output is an anecdote, so to know whether a *change* helped you need a stable population to aggregate over. The segment is that population.

The definitions are **marketer-adjustable by design**. A marketer reads the segment, judges whether it holds, and splits or merges it — the most common trigger being a prompt that works for half a segment and fails for the other half. Segmentation stays a human judgement rather than a property of an opaque clusterer.

Optional, and a genuine choice rather than a requirement: **routing**, where a segment deliberately receives a different prompt strategy. Aggregation for evaluation is not optional; routing is.

### 2. Prompt versioning addresses stochastic-output drift

**This starts first in implementation order.** Store prompts as versioned artifacts rather than inline strings, identified by an immutable prompt-version ID or content hash — not by the situation they were used in. Everything situational belongs in the run record instead. Each generation should preserve a run ID, the immutable prompt-version ID or content hash, the model and its parameters, an input/context snapshot or version reference, the constraint-set version, the output, and — when available — the segment-definition version and assigned segment. Recording the input snapshot is what makes later re-grouping possible: prompt provenance alone is not enough to re-segment a past run. Route a share of generations through an experimental version and log outputs for comparison. A marketing team can then iterate on tone and structure without redeploying, and silent quality regressions become visible.

#### What an A/B test looks like here

**Family 1 — within-segment prompt variants.** Within a stable segment, assign variants randomly — or run both against the same frozen profiles — while holding model, retrieval and constraint settings constant. This makes differences more credibly attributable to the prompt.

| Test | Variant A | Variant B | Measured by |
|---|---|---|---|
| Angle | Lead with specification and price justification | Lead with the time saved | Usefulness score, within segment |
| Evidence width | Inject the top-1 retrieved profile | Inject the top-3 | Whether more context sharpens or dilutes |
| Grounding rule | No constraint | "Use no number that is absent from the retrieved context" | Rate of unsupported numeric claims |
| Self-attribution | Copy only | Copy plus the behavioural signal used | Grounding score; reviewer trust |
| Voice encoding | Brand rules as instructions | Brand rules as few-shot examples | Brand-voice adherence (also pre-tests item 8) |

**Family 2 — cross-segment diagnostics.** One prompt, every segment, looking for where it breaks. A prompt tuned for high-intent professionals may produce pushy calls to action for early-exploration browsers. If performance differs between coherent segments, the human can approve segment-specific routing or revise the shared prompt. If results split inside one segment, that is evidence to revisit its boundary.

**Family 3 — regression tests across versions.** Freeze a golden set: N profiles per segment with approved outputs. When a prompt changes, re-run and compare; a drop rejects the version. This is the test that cannot exist without stable segment definitions, because the golden set would have no membership rule.

**Honest limit.** On synthetic data these tests compare **quality proxies** — validator pass rate, judge scores, human review. They do not measure persuasion. Click-through and conversion require real outcomes, which is Tier 2.

### 3. Evaluation loop plus constraint validation addresses missing regression testing and unsafe output

Two layers that must stay separate, because they fail differently.

**A deterministic constraint validator** blocks: no competitor names; audience size within a stated range; no numeric claim absent from the retrieved context; a required call to action present. Rules are cheap, testable, and produce auditable pass/fail results, although their coverage can still be incomplete.

**An LLM-as-judge scorer** rates what rules cannot: brand-voice adherence, grounding, usefulness.

**The independence rule.** The judge receives the task specification, the output constraints and brand rules, the retrieved context as its evidence set, and the generated output as the object under review, scored against a separately versioned rubric. It does not receive the generator's reasoning. Withholding the requirements would be the opposite error: a judge that cannot see what the output was supposed to do cannot assess whether it did it. Record identity and context provenance are verified before judging rather than inherited silently from generation state. A judge that merely replays the generator's assumptions has the shape of a check and the content of a mirror. Any verification step deserves the same question: what does this compare against, and which inputs were independently verified?

### 4. Per-match generation and input transparency address output-trust erosion

Retrieval transparency is already partly in place: every retrieved match is shown with its similarity score and the context that matched. Two gaps remain.

- **Generation runs for the top match only.** Matches two and three display context but produce no copy, so a viewer cannot see retrieval changing the output side by side.
- **Brand-context inputs are not surfaced** next to the generated copy, only the user profile.

Closing both makes the causal chain visible: this profile plus this brand context produced this copy.

---

## Tier 2 — requires real data or real users

### 5. Live CDP ingestion plus two-store architecture addresses stale audience data and recompute cost

Replace the synthetic generator with a Segment- or RudderStack-shaped event stream consumer, paired with a persistent vector store such as Supabase, Pinecone, or pgvector. Split into a static demographic store (refreshed quarterly, updated on detectable life events) and a rolling 90-day clickstream store (refreshed daily), so each updates at its natural cadence and a behavioural change does not force re-vectorizing the whole profile.

The 90-day window is a starting heuristic, not a derived constant. A publicly reported finding from Meta's internal analytics agent — 88% of internal analyst queries hit tables from the preceding 90 days — is suggestive, but analyst query recency is not customer-behaviour retention: it describes how people query a warehouse, not how long a behavioural signal stays predictive. Validate the cutoff against predictive lift, recompute cost, and privacy retention limits, and move it when the evidence says so. Source: [Inside Meta's Home Grown AI Analytics Agent](https://medium.com/@AnalyticsAtMeta/inside-metas-home-grown-ai-analytics-agent-4ea6779acfb3) (Analytics at Meta, 2026).

### 6. Cross-channel graph signal addresses single-channel myopia

Web behaviour is one surface. The richer signal is relational — user to brand to campaign to outcome, across email, paid, and owned channels — compressed into the profile that gets vectorized. A flat text embedding does not preserve those relationships explicitly; a graph can represent them explicitly, and a governed summary of that graph can become another input to the profile store.

### 7. Outcome feedback closes the loop

Delivered copy produces outcomes, and those outcomes are what turn a generator into a system. They should land first in an immutable event and experiment store — what was sent, what was received, what happened — because that is the record an experiment can be read back from. Writing them straight into the profile is the tempting shortcut and the wrong one: a profile that updates itself from the outcomes of copy it previously influenced reinforces its own targeting and drifts toward a narrowing audience. Profile updates belong downstream of that store, governed, versioned, and checked against holdouts. This is the item that most requires production data.

### 8. Reference marketers as few-shot exemplars address cold-start weakness

Curated examples from experienced marketers — approved segment definitions, message choices, and outputs — can serve as few-shot examples for someone new to the brand. This adds a practitioner-derived layer alongside the top-down brand context while keeping exemplar selection explicit and reviewable.

### 9. Role-based interface broadens operator accessibility

Adapt views for growth, lifecycle, product marketing, and marketing operations. Today's interface assumes a single power user.

---

## Operational foundations, assumed rather than enumerated

The nine items above are product and measurement gaps. They are not the whole distance to production, and listing them alone would overstate how close this is. Anything carrying real customer data also needs, at minimum:

- a run and experiment ledger as the system of record for what was generated, under which prompt and model versions;
- privacy, consent, retention and deletion handling for profile data;
- access control and tenant isolation;
- prompt-injection and data-boundary protection, since retrieved profile text reaches a model;
- latency, cost, retry and rate-limit monitoring;
- a model and embedding migration path, because a stored vector does not survive a model change unchanged;
- fallback and failure handling when retrieval or generation is unavailable; and
- incident response and rollback.

None of these are built here, and none are scheduled by this roadmap. They are named so that it is not mistaken for a complete path to production.

## Deliberately not pursued yet

**Channel-specific output variants.** Generating email, SMS, landing-page, and enablement variants from the same retrieved context is a medium-effort change. It is parked on purpose: a variant's value is whether it performs in its channel, and honest testing requires delivering through real channels and reading real outcomes. That depends on items 5 and 7. Building the variants before the measurement would produce more output with no way to judge it.
