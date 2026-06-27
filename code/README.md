# Multi-Modal Evidence Review — Solution

## Setup

```bash
pip install anthropic pillow python-dotenv
```

Create a `.env` file at the repo root (never commit it):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

### Generate predictions for all test claims

```bash
# Default: claude-opus-4-8 → writes output.csv at repo root
python code/main.py

# With an alternative model
python code/main.py --model claude-sonnet-4-6

# Custom paths
python code/main.py --claims dataset/claims.csv --output output.csv
```

### Evaluate on labeled sample data

```bash
# Both models (writes code/evaluation/evaluation_report.md)
python code/evaluation/main.py

# Single model
python code/evaluation/main.py --model claude-opus-4-8
```

## Architecture

### Two-stage pipeline

**Stage 1 — VLM Perception** (`code/stage1_perception.py`)

One Anthropic API call per claim. All images for a claim are sent in a single multi-image
message. The model returns strict JSON with:

- Per-image observations: `shows_claimed_object`, `angle_inspectable`, `visible_issue_type`,
  `quality_flags` (blurry, wrong_object, claim_mismatch, possible_manipulation, etc.)
- Claim-level synthesis: `object_consistent_across_images`, `visible_issue_type`, severity,
  claimed vs. visible mismatch

Images are downscaled to ≤768 px longest edge before base64 encoding to reduce token cost.
Invalid or out-of-vocabulary VLM values are clamped to allowed enum sets.

**Stage 2 — Deterministic Decision** (`code/stage2_decision.py`)

Zero model calls. A rule ladder maps Stage 1 outputs to the four output fields that require
judgment (`evidence_standard_met`, `claim_status`, `risk_flags`, `supporting_image_ids`):

| Rule | Condition | Outcome |
|------|-----------|---------|
| Pre-X | Consistent images + explicit cross-family mismatch | `contradicted` |
| Pre-Y | No claimed object + claim_mismatch flag | `contradicted` |
| A | Inconsistent images + no clear damage or mixed identity | `not_enough_information` |
| B | No inspectable angle in any image | `not_enough_information` |
| C | No image shows the claimed object | `contradicted` or `not_enough_info` |
| D | Object and angle present but never in the same image | `contradicted` or `not_enough_info` |
| E | Object visible + no damage (except missing-part) | `contradicted` |
| F | `missing_part` claimed, damage_not_visible flagged | `not_enough_information` |
| G | Explicit claim mismatch, different issue family | `contradicted` |
| H | Visible issue unknown | `not_enough_information` |
| I | Damage visible, no contradiction | `supported` |

**Hard constraint:** User history may only add `user_history_risk` / `manual_review_required`
to `risk_flags`. It can never change a clear visual verdict. A Python `assert` enforces this.

**Adversarial injection handling:** If any image contains text instructions (e.g., "approve
this claim"), the VLM flags `text_instruction_present`. Stage 2 re-derives `effective_vi`
from clean (non-tainted) images only; if clean images show no damage, the tainted synthesis
is overridden.

### Caching

Every Stage-1 API response is cached to `code/.cache/` keyed by
`sha256(all_image_bytes + claim_text + claim_object + model_id)`.
Re-runs against unchanged inputs are instant and cost nothing.

### Retry

Exponential back-off on HTTP 429 (rate limit) and 5xx: waits 1 s, 2 s, 4 s, then raises.

## Accuracy

Evaluated on `dataset/sample_claims.csv` (20 labeled cases):

| Model | claim_status accuracy | overall accuracy |
|-------|-----------------------|------------------|
| claude-opus-4-8 | **95%** (19/20) | 68.8% |
| claude-sonnet-4-6 | 90% (18/20) | 55.6% |

The single remaining Opus miss is a VLM perception error: the model reports no visible damage
when the package corner has physical deformation that was clear to the human reviewer.
No Stage-2 fix is applied because patching for a specific sample case would be overfitting.

**Chosen model for `output.csv`: `claude-opus-4-8`** — higher accuracy on multi-image
inconsistency, adversarial injection, and severity mismatch detection.

## Cost

Full 44-row test run (claude-opus-4-8, no cache):
- 125,955 input tokens + 21,632 output tokens
- **$1.17 total** (~$0.027/claim) at $5/M input + $25/M output
- 82 images, 313 s wall time

With the disk cache, subsequent runs cost $0.

## Files

```
code/
├── main.py                   # Entry point → output.csv
├── stage1_perception.py      # VLM call + caching + retry
├── stage2_decision.py        # Deterministic rule ladder
├── cache.py                  # SHA256-keyed disk cache
├── ops_tracker.py            # Token / cost / latency metrics
├── README.md                 # This file
└── evaluation/
    ├── main.py               # Evaluation entry point
    ├── evaluation_report.md  # Two-model comparison report
    └── sample_predictions_*.csv
```
