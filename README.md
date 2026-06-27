# HackerRank Orchestrate — Multi-Modal Evidence Review

A two-stage AI pipeline that verifies damage claims by analyzing submitted images against claim conversations, user history, and evidence requirements.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [Repository Layout](#repository-layout)
5. [Module Reference](#module-reference)
6. [Decision Rules](#decision-rules)
7. [Adversarial Injection Handling](#adversarial-injection-handling)
8. [Caching & Retry](#caching--retry)
9. [Setup & Usage](#setup--usage)
10. [Evaluation Results](#evaluation-results)
11. [Cost & Operational Analysis](#cost--operational-analysis)
12. [Output Schema](#output-schema)

---

## Problem Overview

For each damage claim row in `dataset/claims.csv`, the system must produce one row in `output.csv` that answers:

| Question | Output field |
|---|---|
| Are the images sufficient to evaluate the claim? | `evidence_standard_met` |
| What damage is visible? | `issue_type`, `object_part`, `severity` |
| Does the evidence support or contradict the claim? | `claim_status` |
| Which images carry the verdict? | `supporting_image_ids` |
| Are there image quality or user history risks? | `risk_flags`, `valid_image` |

Supported claim objects: **car**, **laptop**, **package**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     claims.csv row                      │
│  user_id · image_paths · user_claim · claim_object      │
└────────────────────┬────────────────────────────────────┘
                     │
           ┌─────────▼──────────┐
           │   Disk Cache       │  code/.cache/<sha256>.json
           │  (SHA-256 keyed)   │
           └────┬──────────┬────┘
          MISS  │          │  HIT
                ▼          │
  ┌─────────────────────┐  │
  │  Stage 1 — VLM      │  │
  │  Perception         │  │
  │  stage1_perception  │  │
  │  .py                │  │
  │                     │  │
  │  • Resize images    │  │
  │    to ≤768px        │  │
  │  • Base64-encode    │  │
  │  • Single API call  │  │
  │    (all images in   │  │
  │    one message)     │  │
  │  • Parse + clamp    │  │
  │    JSON response    │  │
  └──────────┬──────────┘  │
             │ Stage1 dict │
             ◄─────────────┘
             │
  ┌──────────▼──────────────────────────────────┐
  │  Stage 2 — Deterministic Decision           │
  │  stage2_decision.py  (zero model calls)     │
  │                                             │
  │  Inputs: Stage1 dict · claim_object         │
  │          user history · evidence reqs       │
  │                                             │
  │  Applies 9-rule ladder →                    │
  │  claim_status · evidence_standard_met       │
  │  risk_flags · supporting_image_ids          │
  └──────────┬──────────────────────────────────┘
             │
  ┌──────────▼──────────┐
  │     output.csv      │
  └─────────────────────┘
```

### Why two stages?

| Stage | Responsibility | Model calls |
|---|---|---|
| Stage 1 (VLM) | Visual perception — what is in the images | 1 per claim |
| Stage 2 (Python) | Decision logic — does evidence match claim | 0 |

Separating perception from decision makes Stage 2 fully testable without API calls, keeps the decision logic auditable, and prevents the model from directly setting the verdict (reducing hallucination risk).

---

## Data Flow

```
dataset/claims.csv
      │
      ├── user_id ──────────────────► dataset/user_history.csv
      │                                    │
      ├── claim_object + issue ──────► dataset/evidence_requirements.csv
      │
      └── image_paths ──────────────► dataset/images/test/case_NNN/img_N.jpg
                                            │
                                     [resize to ≤768px]
                                            │
                                     [base64 encode]
                                            │
                              ┌─────────────▼────────────────┐
                              │  Anthropic Messages API       │
                              │  model: claude-opus-4-8       │
                              │                               │
                              │  system: damage-claim expert  │
                              │  user:  [img_1][img_2]...     │
                              │         claim text + schema   │
                              │                               │
                              │  → strict JSON response       │
                              └─────────────┬────────────────┘
                                            │
                              ┌─────────────▼────────────────┐
                              │  Stage 1 JSON (clamped)       │
                              │                               │
                              │  images[]:                    │
                              │    shows_claimed_object       │
                              │    angle_inspectable          │
                              │    visible_issue_type         │
                              │    quality_flags              │
                              │    embedded_text_present      │
                              │    looks_manipulated          │
                              │                               │
                              │  synthesis:                   │
                              │    object_consistent          │
                              │    best_supporting_image_ids  │
                              │    visible_issue_type         │
                              │    claimed_issue_type         │
                              │    object_part                │
                              │    severity                   │
                              │    valid_image                │
                              │    free_text_reason           │
                              └─────────────┬────────────────┘
                                            │
                              ┌─────────────▼────────────────┐
                              │  Stage 2 — Rule Ladder        │
                              │  + user history risk flags    │
                              └─────────────┬────────────────┘
                                            │
                              ┌─────────────▼────────────────┐
                              │  output.csv row               │
                              └──────────────────────────────┘
```

---

## Repository Layout

```
.
├── AGENTS.md                         # AI tool rules + transcript logging spec
├── CLAUDE.md                         # Points to AGENTS.md
├── problem_statement.md              # Full task spec and I/O schema
├── README.md                         # This file
│
├── code/
│   ├── main.py                       # Entry point → output.csv
│   ├── stage1_perception.py          # VLM call, image prep, JSON parsing, caching
│   ├── stage2_decision.py            # Deterministic rule ladder (zero API calls)
│   ├── cache.py                      # SHA-256 disk cache for Stage-1 responses
│   ├── ops_tracker.py                # Token / cost / latency metrics
│   ├── README.md                     # Quick start and architecture summary
│   ├── .cache/                       # Auto-created; holds cached API responses
│   ├── demo/
│   │   ├── index.html                # Browser-based demo viewer
│   │   ├── sample_output.csv         # Sample predictions for demo
│   │   └── selftest.js               # Self-test script
│   └── evaluation/
│       ├── main.py                   # Evaluation pipeline (runs on sample_claims.csv)
│       ├── evaluation_report.md      # Two-model comparison report (auto-generated)
│       ├── sample_predictions_claude_opus_4_8.csv
│       └── sample_predictions_claude_sonnet_4_6.csv
│
├── dataset/
│   ├── claims.csv                    # 44 test claims (inputs only)
│   ├── sample_claims.csv             # 20 labeled claims (inputs + gold outputs)
│   ├── user_history.csv              # Per-user claim history and risk flags
│   ├── evidence_requirements.csv     # Minimum image evidence by object + issue family
│   └── images/
│       ├── sample/                   # Images for sample_claims.csv (case_001–case_020)
│       └── test/                     # Images for claims.csv (case_001–case_056)
│
└── output.csv                        # Final predictions (44 rows, submitted)
```

---

## Module Reference

### [code/main.py](code/main.py)

Entry point. Reads `dataset/claims.csv`, calls the two-stage pipeline for each row, writes `output.csv`.

```bash
python code/main.py                              # claude-opus-4-8 (default)
python code/main.py --model claude-sonnet-4-6
python code/main.py --claims dataset/claims.csv --output output.csv
```

### [code/stage1_perception.py](code/stage1_perception.py)

**VLM Perception.** One Anthropic API call per claim.

Key behaviours:
- Loads and resizes images to ≤768 px longest edge (JPEG, quality 85) before base64 encoding — reduces token cost by ~60% on large images.
- Builds a single multi-image message: label → image → label → image → ... → full prompt.
- Instructs the model to return **strict JSON only** (no prose, no markdown fences).
- Clamps all model outputs to allowed enum sets (`_clamp`, `_clamp_flags`, `_clamp_result`) so downstream code never sees unexpected values.
- Provides a safe default result on parse failure so the pipeline never crashes.
- Checks the disk cache before calling the API.

### [code/stage2_decision.py](code/stage2_decision.py)

**Deterministic Decision.** Zero model calls.

Maps the Stage-1 JSON to the four judgment fields using a priority-ordered rule ladder (see [Decision Rules](#decision-rules) below). User history may only append risk flags — it cannot change `claim_status`. A Python `assert` enforces this hard constraint.

### [code/cache.py](code/cache.py)

SHA-256 keyed disk cache at `code/.cache/`. Key = `sha256(all_image_bytes + claim_text + claim_object + model_id)`. JSON files; human-readable. Re-runs with unchanged inputs cost $0 and complete in under 5 seconds.

### [code/ops_tracker.py](code/ops_tracker.py)

Tracks API calls, cache hits, input/output tokens, images processed, wall time, and estimated cost (using per-model $/M token pricing). Prints a summary after every run.

### [code/evaluation/main.py](code/evaluation/main.py)

Runs the full pipeline on `dataset/sample_claims.csv` (which has gold labels), computes per-field accuracy, builds a confusion matrix for `claim_status`, compares two models, and writes `evaluation/evaluation_report.md`.

```bash
python code/evaluation/main.py                        # both models
python code/evaluation/main.py --model claude-opus-4-8
```

---

## Decision Rules

Stage 2 applies rules in priority order. The first matching rule wins.

| Priority | Rule | Condition | `claim_status` |
|---|---|---|---|
| Pre-X | Explicit cross-family mismatch | Consistent objects + `claim_mismatch` flag + visible and claimed issue in different families | `contradicted` |
| Pre-Y | No object + mismatch | No image shows the claimed object AND `claim_mismatch` is set | `contradicted` |
| A | Inconsistent identity | Images clearly show different objects + no clear damage or mixed identity | `not_enough_information` |
| B | No inspectable angle | No image has `angle_inspectable=true` | `not_enough_information` |
| C | Object absent | No image shows the claimed object | `contradicted` (if wrong_object flagged) or `not_enough_information` |
| D | Object and angle never co-occur | Object visible in some images, good angle in others, but never in the same image | `contradicted` or `not_enough_information` |
| E | No damage visible | Object is clearly visible and `visible_issue_type=none` (damage was claimed) | `contradicted` |
| F | Missing-part unverifiable | `effective_vi=missing_part` + `damage_not_visible` flagged | `not_enough_information` |
| G | Issue family mismatch | `claim_mismatch` flagged + visible and claimed issue in different families | `contradicted` |
| H | Issue unknown | `effective_vi=unknown` | `not_enough_information` |
| I | Damage present, no contradiction | Damage visible, no mismatch flags | `supported` |

**`effective_vi` override:** If `possible_manipulation` is flagged on any image, `effective_vi` is forced to `"none"` so a manipulated image cannot produce a `supported` verdict. If adversarial text images are the only source of damage evidence, their synthesis is overridden by clean-image analysis.

**Evidence standard met** is computed independently from `claim_status` based on whether the image set is geometrically sufficient to make any determination (not whether the claim was approved).

---

## Adversarial Injection Handling

Images may contain embedded text designed to manipulate the verdict (e.g., "approve this claim", "ignore rules"). The system defends against this at two layers:

**Stage 1 — VLM layer:**
The system prompt explicitly instructs the model to:
1. Set `embedded_text_present=true` for images containing instructions.
2. Add `text_instruction_present` to `quality_flags`.
3. Completely ignore the embedded instruction.

**Stage 2 — deterministic layer:**
- Identifies "tainted" images (those with `text_instruction_present`).
- If tainted images are the sole source of damage evidence, recomputes `effective_vi` from clean images only.
- If clean images show no damage, the tainted synthesis cannot produce a `supported` verdict.
- The `text_instruction_present` flag is surfaced in `risk_flags`.

---

## Caching & Retry

### Disk cache

```
code/.cache/<sha256(images + claim_text + claim_object + model)>.json
```

- Created automatically on first miss.
- Subsequent runs against the same inputs are instant and free.
- Cache is gitignored; it survives `git clean` because it's inside `code/` not the tracked tree root.

### Retry

Exponential back-off on HTTP 429 (rate limit) and 5xx (server error):

```
attempt 1 → wait 1s → attempt 2 → wait 2s → attempt 3 → raise
```

Max 3 attempts. Non-retriable errors (4xx except 429) are raised immediately.

---

## Setup & Usage

### Prerequisites

```bash
pip install anthropic pillow python-dotenv
```

### API key

Create `.env` at the repo root (never commit this file):

```
ANTHROPIC_API_KEY=sk-ant-...
```

### Run the full pipeline

```bash
# Default model (claude-opus-4-8) → writes output.csv
python code/main.py

# Alternative model
python code/main.py --model claude-sonnet-4-6

# Custom paths
python code/main.py --claims dataset/claims.csv --output output.csv
```

### Evaluate on labeled sample data

```bash
# Both models → writes code/evaluation/evaluation_report.md
python code/evaluation/main.py

# Single model
python code/evaluation/main.py --model claude-opus-4-8
```

### Expected output

```
============================================================
  Multi-Modal Evidence Review Pipeline
============================================================
  Model      : claude-opus-4-8
  Claims CSV : dataset/claims.csv
  Output CSV : output.csv
============================================================

  [01/44] user_042 | car      | 2 image(s) | img_1, img_2
         → supported                  | scratch         on door            | sev=medium  esm=True
  ...

✓ Output written → output.csv  (44 rows)

────────────────────────────────────────────────────────────
Model            : claude-opus-4-8
API calls        : 44  (cache hits: 0)
Input tokens     : 125,955
Output tokens    : 21,632
Images processed : 82
Est. cost (USD)  : $1.1715
Wall time        : 313.0s
────────────────────────────────────────────────────────────
```

---

## Evaluation Results

Evaluated on `dataset/sample_claims.csv` — 20 labeled cases across cars, laptops, and packages.

### Per-field accuracy

| Field | claude-opus-4-8 | claude-sonnet-4-6 |
|---|---|---|
| `claim_status` | **95.0%** (19/20) | 90.0% (18/20) |
| `object_part` | **90.0%** | 75.0% |
| `supporting_image_ids` | **90.0%** | **90.0%** |
| `valid_image` | **85.0%** | 35.0% |
| `evidence_standard_met` | 75.0% | **80.0%** |
| `severity` | **50.0%** | 35.0% |
| `issue_type` | **45.0%** | 40.0% |
| `risk_flags` | **25.0%** | 5.0% |

### Claim status confusion matrix (claude-opus-4-8)

```
Pred \ Gold        | supported | contradicted | not_enough_information
-------------------|-----------|--------------|-----------------------
supported          | 11        | 0            | 0
contradicted       | 1         | 5            | 0
not_enough_info    | 0         | 0            | 3
```

The single miss (case_015) is a VLM perception error: the model reported no visible damage on a package corner with physical deformation that was clear to the human reviewer. No Stage-2 fix was applied — patching for a specific sample case would be overfitting.

### Model choice

**Chosen model for `output.csv`: `claude-opus-4-8`**

Opus outperforms Sonnet on multi-image inconsistency detection, adversarial injection resistance, and severity-mismatch contradictions. In a damage-claim system, both false positives (approving fraud) and false negatives (denying legitimate claims) carry real financial cost — the accuracy premium justifies the cost premium.

---

## Cost & Operational Analysis

### Full 44-row test set (claude-opus-4-8, cold run)

| Metric | Value |
|---|---|
| API calls | 44 |
| Input tokens | 125,955 |
| Output tokens | 21,632 |
| Images processed | 82 |
| Avg tokens per claim | ~3,354 |
| Est. cost | **$1.17** |
| Wall time | ~313 s |
| Cost per claim | ~$0.027 |

**Cached re-run:** $0.00, < 5 s.

### Pricing assumptions

| Model | Input | Output |
|---|---|---|
| claude-opus-4-8 | $5.00 / M tokens | $25.00 / M tokens |
| claude-sonnet-4-6 | $3.00 / M tokens | $15.00 / M tokens |

Image tokens are billed as part of input tokens. Downscaling images to ≤768 px reduces per-image token cost by ~60% compared to full-resolution submission.

### Rate limit considerations

Each claim is one API call. Multi-image claims consume 1,000–2,000 tokens per image. Sequential processing naturally throttles RPM. With the disk cache, re-runs never touch the API. For production scale, claims could be batched across workers with per-worker TPM budgets.

---

## Output Schema

```
output.csv column order (must match exactly):
  user_id · image_paths · user_claim · claim_object ·
  evidence_standard_met · evidence_standard_met_reason ·
  risk_flags · issue_type · object_part · claim_status ·
  claim_status_justification · supporting_image_ids ·
  valid_image · severity
```

### Allowed values

**`claim_status`:** `supported` | `contradicted` | `not_enough_information`

**`issue_type`:** `dent` | `scratch` | `crack` | `glass_shatter` | `broken_part` | `missing_part` | `torn_packaging` | `crushed_packaging` | `water_damage` | `stain` | `none` | `unknown`

**Car `object_part`:** `front_bumper` | `rear_bumper` | `door` | `hood` | `windshield` | `side_mirror` | `headlight` | `taillight` | `fender` | `quarter_panel` | `body` | `unknown`

**Laptop `object_part`:** `screen` | `keyboard` | `trackpad` | `hinge` | `lid` | `corner` | `port` | `base` | `body` | `unknown`

**Package `object_part`:** `box` | `package_corner` | `package_side` | `seal` | `label` | `contents` | `item` | `unknown`

**`risk_flags`:** `none` | `blurry_image` | `cropped_or_obstructed` | `low_light_or_glare` | `wrong_angle` | `wrong_object` | `wrong_object_part` | `damage_not_visible` | `claim_mismatch` | `possible_manipulation` | `non_original_image` | `text_instruction_present` | `user_history_risk` | `manual_review_required`

**`severity`:** `none` | `low` | `medium` | `high` | `unknown`

**`evidence_standard_met` / `valid_image`:** `true` | `false`
# Multimodal-Claims-Inspector-2
