# Roadmap: Named Production Gaps

Seven gaps stand between this prototype and operator-grade tooling for an LLM-in-marketing stack. Items 1 through 4 close named production gaps; items 5 and 6 broaden output trust and cold-start behavior; item 7 broadens who can use the system once those gaps are closed.

## 1. Prompt versioning addresses stochastic-output drift

Store prompts as versioned artifacts; route a percentage of generation requests through experimental templates; log outputs for comparison. This lets a marketing team iterate on tone and structure without redeploying, and surfaces when output quality silently degrades after a change.

## 2. Evaluation loop plus constraint-validation addresses lack of regression testing AND unsafe outputs

Two layers, not one. First, LLM-as-judge scoring for output quality, brand-voice adherence, factual grounding, and marketing usefulness, paired with periodic human spot checks. Second, a constraint-validation layer — natural-language rules ("no competitor names in generated copy"; "audience size must be 100 to 5M"; "no restricted terms per brand policy") checked by a separate AI system against every generation before it reaches the marketer. The second layer catches unsafe outputs faster than an eval-loop-only design.

## 3. Channel-specific output variants address format heterogeneity across channels

Use the same retrieved context, then generate variants for email subject lines, SMS, landing pages, lifecycle messages, and sales enablement snippets. Today's single-output design assumes channel-agnostic copy, which is the exception, not the rule.

## 4. Live CDP ingestion plus two-store architecture addresses stale audience data AND recompute cost

Two changes together, not one. First, replace the synthetic generator with a Segment / Rudderstack-shaped event stream consumer, paired with a persistent vector store such as Supabase, Pinecone, or pgvector. Second, split into a static demographic store (refreshed quarterly) plus a rolling 90-day clickstream store (refreshed daily). This decoupling lets each store update at its natural cadence and avoids the recompute cost of re-vectorizing the whole user profile whenever behavior changes.

The 90-day window aligns with a publicly reported finding from Meta's internal analytics agent — 88% of internal analyst queries hit tables from the preceding 90 days — as one supporting data point for the recency bet. Source: [Inside Meta's Home Grown AI Analytics Agent](https://medium.com/@AnalyticsAtMeta/inside-metas-home-grown-ai-analytics-agent-4ea6779acfb3) (Analytics at Meta, 2026).

## 5. Source-signal transparency addresses output-trust erosion

Every generation should surface what shaped it: which retrieved users, which behavioral patterns, and which brand-context inputs went into the prompt. Surfacing the retrieved user profiles and dominant clickstream signals alongside each generated persona or recommendation lets a marketer steer, verify, or reuse rather than take output on faith.

## 6. Reference marketers as few-shot exemplars address cold-start weakness

Naming reference marketers — for example, a senior PMM whose past segmentation choices seed the retrieval for a new marketer working on the same brand — creates a bottom-up personalization layer (individual retrieval history) on top of the top-down brand-context layer already in place. This shortens ramp-up for a new marketer joining an existing brand workflow.

## 7. Role-based interface broadens operator accessibility

Adapt views for growth marketers, lifecycle marketers, product marketing, and marketing operations. Today's UI assumes a single power-user persona.
