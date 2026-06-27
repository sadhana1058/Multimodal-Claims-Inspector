# Evaluation Report — Multi-Modal Evidence Review

## System Overview

Two-stage pipeline:
- **Stage 1 (perception):** One Anthropic API call per claim with ALL images in a single
  multi-image message. The model returns strict JSON describing per-image observations and
  a claim-level synthesis (visible issue, severity, object part, claimed vs. visible mismatch).
  Images are downscaled to ≤768 px longest edge before base64 encoding to reduce token cost.
- **Stage 2 (decision):** Pure deterministic Python. Applies the status ladder:
  inconsistent objects → `not_enough_information`; no inspectable angle → `not_enough_info`;
  part visible + damage absent → `contradicted`; claim mismatch → `contradicted`;
  evidence met + issue matches → `supported`; else → `not_enough_information`.
  User history may only **add** `user_history_risk` / `manual_review_required` flags —
  it can never flip a clear visual verdict. An assertion enforces this hard constraint.

Evaluated on `dataset/sample_claims.csv` — **20 labeled cases**.

## Per-Field Accuracy

| Field                          | claude-opus-4-8      | claude-sonnet-4-6    |
|────────────────────────────────|──────────────────────|──────────────────────|
| evidence_standard_met          | 75.0%                | 80.0%                |
| risk_flags                     | 25.0%                | 5.0%                |
| issue_type                     | 45.0%                | 40.0%                |
| object_part                    | 90.0%                | 75.0%                |
| claim_status                   | 95.0%                | 90.0%                |
| supporting_image_ids           | 90.0%                | 90.0%                |
| valid_image                    | 85.0%                | 35.0%                |
| severity                       | 50.0%                | 35.0%                |

## Claim Status Confusion Matrix

### claude-opus-4-8

```
Pred \ Gold        | supported                | contradicted             | not_enough_information  
───────────────────────────────────────────────────────────────────────────────────────────────────
supported          | 11                       | 0                        | 0                       
contradicted       | 1                        | 5                        | 0                       
not_enough_information | 0                        | 0                        | 3                       
```

### claude-sonnet-4-6

```
Pred \ Gold        | supported                | contradicted             | not_enough_information  
───────────────────────────────────────────────────────────────────────────────────────────────────
supported          | 12                       | 2                        | 0                       
contradicted       | 0                        | 3                        | 0                       
not_enough_information | 0                        | 0                        | 3                       
```

## Per-Case Breakdown

| Case | Object | Gold Status | opus-4 Pred | sonnet-4 Pred |
|------|--------|-------------|---||---|
| case_001 | car | supported | ✓ supported | ✓ supported |
| case_002 | car | not_enough_information | ✓ not_enough_information | ✓ not_enough_information |
| case_003 | car | supported | ✓ supported | ✓ supported |
| case_004 | car | supported | ✓ supported | ✓ supported |
| case_005 | car | contradicted | ✓ contradicted | ✗ supported |
| case_006 | car | not_enough_information | ✓ not_enough_information | ✓ not_enough_information |
| case_007 | car | supported | ✓ supported | ✓ supported |
| case_008 | car | contradicted | ✓ contradicted | ✓ contradicted |
| case_009 | laptop | supported | ✓ supported | ✓ supported |
| case_010 | laptop | supported | ✓ supported | ✓ supported |
| case_011 | laptop | supported | ✓ supported | ✓ supported |
| case_012 | laptop | supported | ✓ supported | ✓ supported |
| case_013 | laptop | supported | ✓ supported | ✓ supported |
| case_014 | laptop | contradicted | ✓ contradicted | ✗ supported |
| case_015 | package | supported | ✗ contradicted | ✓ supported |
| case_016 | package | supported | ✓ supported | ✓ supported |
| case_017 | package | supported | ✓ supported | ✓ supported |
| case_018 | package | not_enough_information | ✓ not_enough_information | ✓ not_enough_information |
| case_019 | package | contradicted | ✓ contradicted | ✓ contradicted |
| case_020 | package | contradicted | ✓ contradicted | ✓ contradicted |

## Accuracy vs Cost Comparison

| Model | claim_status acc | overall acc | est. cost/claim | est. total (44 rows) | wall time |
|-------|-----------------|-------------|-----------------|----------------------|-----------|
| claude-opus-4-8 | 95.0% | 69.4% | ~$0.027 | ~$1.17 | ~313s |
| claude-sonnet-4-6 | 90.0% | 56.2% | ~$0.016 | ~$0.70 | ~160s |

> **Pricing assumptions (cold run):** claude-opus-4-8: \$5/M input + \$25/M output.
> claude-sonnet-4-6: \$3/M input + \$15/M output.
> Image tokens are billed as part of input tokens by the Anthropic API.
> Opus cost and runtime are **measured** from the first cold run (no cache).
> Sonnet cost is **estimated** using the same token counts at Sonnet pricing.
> Cached re-runs cost ~\$0 and complete in under 5 seconds.

## Final Config Chosen for output.csv

**Model:** `claude-opus-4-8`

**claim_status accuracy:** 95.0%

**Rationale:** In a damage-claim verification system both false positives (approving
fraudulent claims) and false negatives (denying legitimate ones) carry real financial
and reputational cost. The Opus model's higher accuracy on multi-image cases, wrong-object
detection, and severity-mismatch contradictions justifies its cost premium. The disk cache
makes repeated runs free, so production iteration cost is low.

## Operational Analysis

### claude-opus-4-8 — sample run (20 claims, cold-run estimate)

> Scaled from the measured 44-claim cold run (125,955 input + 21,632 output tokens, 313s, $1.17).
> Cached re-runs of the same 20 claims cost ~$0 and complete in under 2 seconds.

| Metric | Value |
|--------|-------|
| API calls (live) | 20 |
| Cache hits | 0 |
| Input tokens | ~57,252 |
| Output tokens | ~9,833 |
| Images processed | 29 |
| Est. cost (sample) | ~$0.53 |
| Wall time | ~142s |
| Avg tokens/claim | ~3,354 |

### claude-sonnet-4-6 — sample run (20 claims, cold-run estimate)

> Estimated using Sonnet pricing (\$3/M input + \$15/M output) applied to the same token counts
> as the Opus cold run. Input token usage is model-independent (same images + same prompt).
> Cached re-runs cost ~$0 and complete in under 1 second.

| Metric | Value |
|--------|-------|
| API calls (live) | 20 |
| Cache hits | 0 |
| Input tokens | ~57,252 |
| Output tokens | ~9,833 |
| Images processed | 29 |
| Est. cost (sample) | ~$0.32 |
| Wall time | ~73s |
| Avg tokens/claim | ~3,354 |

### Full Test Set Results (44 rows)

- **claude-opus-4-8 (measured):** 44 API calls, $1.17 cost, 313s runtime, 82 images — 125,955 input + 21,632 output tokens at \$5/M input + \$25/M output
- **claude-sonnet-4-6 (estimated):** 44 API calls, ~$0.70 cost, ~160s runtime, 82 images — same token counts at \$3/M input + \$15/M output
- Subsequent runs against unchanged inputs cost **~$0** due to the SHA-256 disk cache.

### Caching Strategy

Every Stage-1 VLM response is cached to disk at `code/.cache/` keyed by
`sha256(image_bytes + claim_text + claim_object + model_id)`.
Re-runs with unchanged inputs incur zero API cost and near-zero latency.
Cache survives branch switches and `git clean -f` because it lives outside the tracked tree.

### Retry Strategy

Exponential back-off on HTTP 429 (rate limit) and 5xx (server error):
waits 1 s → 2 s → 4 s, then propagates the exception. Max 3 attempts.

### Rate Limit Considerations

Multi-image claims consume more input tokens (each image ≈ 1,000–2,000 tokens).
The pipeline processes claims sequentially, naturally throttling RPM.
With the disk cache, a second run on the same data is instant and never touches the API.
For production scale, claims could be batched across workers with per-worker rate budgets.

### Injection Handling

Adversarial text embedded in images (e.g. 'approve this claim', 'skip review') is
explicitly addressed in the Stage-1 system prompt and the per-image analysis schema.
The VLM flags `text_instruction_present` and ignores the embedded instruction.
Stage 2 propagates this flag into `risk_flags`. The deterministic Stage-2 logic never
reads or acts on claim conversation text directly — only the structured Stage-1 output.
