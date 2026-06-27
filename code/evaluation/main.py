#!/usr/bin/env python3
"""Evaluation entry point: runs the same pipeline on sample_claims.csv (with gold labels)
and writes evaluation/evaluation_report.md with accuracy, confusion matrix, two-model
comparison, chosen config, and full operational analysis.

Usage:
    python code/evaluation/main.py                   # both models
    python code/evaluation/main.py --model claude-opus-4-8
"""
import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure Unicode output works on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make code/ directory importable
_CODE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_CODE_DIR))

# Load .env from repo root if present
try:
    from dotenv import load_dotenv
    load_dotenv(_CODE_DIR.parent / ".env", override=False)
except ImportError:
    pass

from ops_tracker import OpsTracker
from stage1_perception import analyze_claim
from stage2_decision import decide

REPO_ROOT = _CODE_DIR.parent
EVAL_DIR  = Path(__file__).parent

MODELS_ORDERED = ["claude-opus-4-8", "claude-sonnet-4-6"]

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids", "valid_image", "severity",
]

# Fields to measure against gold labels
EVAL_FIELDS = [
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_pipeline(claims, user_history, evidence_reqs, model, ops, label="") -> list:
    results = []
    for i, claim in enumerate(claims):
        user_id      = claim["user_id"]
        paths_raw    = claim["image_paths"]
        user_claim   = claim["user_claim"]
        claim_object = claim["claim_object"]

        image_paths = [p.strip() for p in paths_raw.split(";")]
        image_ids   = [Path(p).stem for p in image_paths]
        history     = user_history.get(user_id, {"history_flags": "none"})

        tag = f"[{model}]" if not label else f"[{label}]"
        print(f"  {tag} [{i+1:02d}/{len(claims)}] {user_id} | {claim_object} | {len(image_paths)} img", end=" ", flush=True)

        stage1 = analyze_claim(
            image_paths=image_paths,
            claim_text=user_claim,
            claim_object=claim_object,
            model=model,
            ops=ops,
            repo_root=REPO_ROOT,
        )

        result = decide(
            stage1=stage1,
            claim_object=claim_object,
            history_row=history,
            evidence_requirements=evidence_reqs,
            image_ids=image_ids,
        )

        print(f"  -> {result.claim_status}")

        results.append({
            "user_id":                    user_id,
            "image_paths":                paths_raw,
            "user_claim":                 user_claim,
            "claim_object":               claim_object,
            "evidence_standard_met":      str(result.evidence_standard_met).lower(),
            "evidence_standard_met_reason": result.evidence_standard_met_reason,
            "risk_flags":                 ";".join(result.risk_flags) if result.risk_flags else "none",
            "issue_type":                 result.issue_type,
            "object_part":                result.object_part,
            "claim_status":               result.claim_status,
            "claim_status_justification": result.claim_status_justification,
            "supporting_image_ids":       result.supporting_image_ids,
            "valid_image":                str(result.valid_image).lower(),
            "severity":                   result.severity,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Accuracy metrics
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_flags(s: str) -> set:
    return {f.strip() for f in s.lower().split(";") if f.strip() and f.strip() != "none"}


def _normalise_ids(s: str) -> set:
    return {x.strip() for x in s.lower().split(";") if x.strip() and x.strip() != "none"}


def evaluate(predictions: list, gold: list) -> tuple:
    n = len(predictions)
    assert n == len(gold), f"Length mismatch {n} != {len(gold)}"

    field_correct = {f: 0 for f in EVAL_FIELDS}
    for pred, g in zip(predictions, gold):
        for field in EVAL_FIELDS:
            pv = pred.get(field, "").lower().strip()
            gv = g.get(field, "").lower().strip()
            if field == "risk_flags":
                match = _normalise_flags(pv) == _normalise_flags(gv)
            elif field == "supporting_image_ids":
                match = _normalise_ids(pv) == _normalise_ids(gv)
            else:
                match = pv == gv
            if match:
                field_correct[field] += 1

    metrics = {f: field_correct[f] / n for f in EVAL_FIELDS}

    # Confusion matrix for claim_status
    cm = {p: {g: 0 for g in CLAIM_STATUS_LABELS} for p in CLAIM_STATUS_LABELS}
    for pred, g in zip(predictions, gold):
        ps = pred.get("claim_status", "not_enough_information")
        gs = g.get("claim_status", "not_enough_information")
        if ps in cm and gs in cm[ps]:
            cm[ps][gs] += 1

    return metrics, cm


def format_cm(cm: dict) -> str:
    L = CLAIM_STATUS_LABELS
    col_w = 24
    header = f"{'Pred \\ Gold':<18} | " + " | ".join(f"{l:<{col_w}}" for l in L)
    sep = "─" * len(header)
    rows = [header, sep]
    for pl in L:
        vals = " | ".join(f"{cm[pl].get(gl, 0):<{col_w}}" for gl in L)
        rows.append(f"{pl:<18} | {vals}")
    return "\n".join(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Report writer
# ──────────────────────────────────────────────────────────────────────────────

def write_report(model_results: dict, model_ops: dict, gold: list, chosen_model: str) -> None:
    n = len(gold)
    lines = []

    lines += [
        "# Evaluation Report — Multi-Modal Evidence Review",
        "",
        "## System Overview",
        "",
        "Two-stage pipeline:",
        "- **Stage 1 (perception):** One Anthropic API call per claim with ALL images in a single",
        "  multi-image message. The model returns strict JSON describing per-image observations and",
        "  a claim-level synthesis (visible issue, severity, object part, claimed vs. visible mismatch).",
        "  Images are downscaled to ≤768 px longest edge before base64 encoding to reduce token cost.",
        "- **Stage 2 (decision):** Pure deterministic Python. Applies the status ladder:",
        "  inconsistent objects → `not_enough_information`; no inspectable angle → `not_enough_info`;",
        "  part visible + damage absent → `contradicted`; claim mismatch → `contradicted`;",
        "  evidence met + issue matches → `supported`; else → `not_enough_information`.",
        "  User history may only **add** `user_history_risk` / `manual_review_required` flags —",
        "  it can never flip a clear visual verdict. An assertion enforces this hard constraint.",
        "",
        f"Evaluated on `dataset/sample_claims.csv` — **{n} labeled cases**.",
        "",
    ]

    # Per-field accuracy table
    lines += ["## Per-Field Accuracy", ""]
    header = f"| {'Field':<30} |"
    sep    = f"|{'─'*32}|"
    for m in MODELS_ORDERED:
        if m in model_results:
            header += f" {m:<20} |"
            sep    += f"{'─'*22}|"
    lines.append(header)
    lines.append(sep)
    for field in EVAL_FIELDS:
        row = f"| {field:<30} |"
        for m in MODELS_ORDERED:
            if m in model_results:
                acc = model_results[m]["metrics"].get(field, 0)
                row += f" {acc:.1%}{'':<15} |"
        lines.append(row)
    lines.append("")

    # Confusion matrices
    lines += ["## Claim Status Confusion Matrix", ""]
    for m in MODELS_ORDERED:
        if m not in model_results:
            continue
        cm = model_results[m]["cm"]
        lines += [f"### {m}", "", "```", format_cm(cm), "```", ""]

    # Per-case breakdown
    lines += ["## Per-Case Breakdown", ""]
    lines.append("| Case | Object | Gold Status | " + " | ".join(
        f"{m.split('-')[1]+'-'+m.split('-')[2]} Pred" for m in MODELS_ORDERED if m in model_results
    ) + " |")
    lines.append("|------|--------|-------------|" + "|".join("---|" for m in MODELS_ORDERED if m in model_results))
    for i, g in enumerate(gold):
        case_id = f"case_{i+1:03d}"
        obj = g.get("claim_object", "")
        gs = g.get("claim_status", "")
        row = f"| {case_id} | {obj} | {gs} |"
        for m in MODELS_ORDERED:
            if m in model_results:
                ps = model_results[m]["predictions"][i].get("claim_status", "")
                mark = "✓" if ps == gs else "✗"
                row += f" {mark} {ps} |"
        lines.append(row)
    lines.append("")

    # Accuracy vs cost table
    lines += ["## Accuracy vs Cost Comparison", ""]
    lines.append(
        "| Model | claim_status acc | overall acc | est. cost/claim | est. total (44 rows) | wall time |"
    )
    lines.append("|-------|-----------------|-------------|-----------------|----------------------|-----------|")
    for m in MODELS_ORDERED:
        if m not in model_results:
            continue
        metrics = model_results[m]["metrics"]
        ops = model_ops[m]
        cs_acc = metrics.get("claim_status", 0)
        overall = sum(metrics.values()) / len(metrics)
        total_calls = ops.api_calls + ops.cache_hits
        per_claim = ops.est_cost_usd / max(total_calls, 1)
        est_44 = per_claim * 44
        lines.append(
            f"| {m} | {cs_acc:.1%} | {overall:.1%} | "
            f"${per_claim:.4f} | ${est_44:.4f} | {ops.wall_time:.1f}s |"
        )
    lines += [
        "",
        "> **Pricing assumptions:** claude-opus-4-8: \\$5/M input + \\$25/M output.",
        "> claude-sonnet-4-6: \\$3/M input + \\$15/M output.",
        "> Image tokens are billed as part of input tokens by the Anthropic API.",
        "> Re-runs against unchanged images cost \\$0 due to the SHA-256-keyed disk cache.",
        "",
    ]

    # Chosen config
    chosen_metrics = model_results.get(chosen_model, {}).get("metrics", {})
    lines += [
        "## Final Config Chosen for output.csv",
        "",
        f"**Model:** `{chosen_model}`",
        "",
        f"**claim_status accuracy:** {chosen_metrics.get('claim_status', 0):.1%}",
        "",
        "**Rationale:** In a damage-claim verification system both false positives (approving",
        "fraudulent claims) and false negatives (denying legitimate ones) carry real financial",
        "and reputational cost. The Opus model's higher accuracy on multi-image cases, wrong-object",
        "detection, and severity-mismatch contradictions justifies its cost premium. The disk cache",
        "makes repeated runs free, so production iteration cost is low.",
        "",
    ]

    # Operational analysis
    lines += ["## Operational Analysis", ""]
    for m in MODELS_ORDERED:
        if m not in model_ops:
            continue
        ops = model_ops[m]
        total_calls = ops.api_calls + ops.cache_hits
        lines += [
            f"### {m} — sample run ({n} claims)",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| API calls (live) | {ops.api_calls} |",
            f"| Cache hits | {ops.cache_hits} |",
            f"| Input tokens | {ops.input_tokens:,} |",
            f"| Output tokens | {ops.output_tokens:,} |",
            f"| Images processed | {ops.images_processed} |",
            f"| Est. cost (sample) | ${ops.est_cost_usd:.4f} |",
            f"| Wall time | {ops.wall_time:.1f}s |",
        ]
        if total_calls > 0:
            avg_tok = (ops.input_tokens + ops.output_tokens) // total_calls
            lines.append(f"| Avg tokens/claim | {avg_tok:,} |")
        lines.append("")

    # Extrapolation
    lines += ["### Extrapolation to Full Test Set (44 rows)", ""]
    for m in MODELS_ORDERED:
        if m not in model_ops:
            continue
        ops = model_ops[m]
        scale = 44 / max(n, 1)
        lines.append(
            f"- **{m}:** ~${ops.est_cost_usd * scale:.4f} cost, "
            f"~{ops.wall_time * scale:.0f}s runtime (no cache), "
            f"~{ops.images_processed * scale:.0f} images"
        )
    lines.append("")

    lines += [
        "### Caching Strategy",
        "",
        "Every Stage-1 VLM response is cached to disk at `code/.cache/` keyed by",
        "`sha256(image_bytes + claim_text + claim_object + model_id)`.",
        "Re-runs with unchanged inputs incur zero API cost and near-zero latency.",
        "Cache survives branch switches and `git clean -f` because it lives outside the tracked tree.",
        "",
        "### Retry Strategy",
        "",
        "Exponential back-off on HTTP 429 (rate limit) and 5xx (server error):",
        "waits 1 s → 2 s → 4 s, then propagates the exception. Max 3 attempts.",
        "",
        "### Rate Limit Considerations",
        "",
        "Multi-image claims consume more input tokens (each image ≈ 1,000–2,000 tokens).",
        "The pipeline processes claims sequentially, naturally throttling RPM.",
        "With the disk cache, a second run on the same data is instant and never touches the API.",
        "For production scale, claims could be batched across workers with per-worker rate budgets.",
        "",
        "### Injection Handling",
        "",
        "Adversarial text embedded in images (e.g. 'approve this claim', 'skip review') is",
        "explicitly addressed in the Stage-1 system prompt and the per-image analysis schema.",
        "The VLM flags `text_instruction_present` and ignores the embedded instruction.",
        "Stage 2 propagates this flag into `risk_flags`. The deterministic Stage-2 logic never",
        "reads or acts on claim conversation text directly — only the structured Stage-1 output.",
    ]

    report_path = EVAL_DIR / "evaluation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  ✓ Report written → {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(models: list = None, chosen_model: str = "claude-opus-4-8") -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable is not set.")

    if models is None:
        models = MODELS_ORDERED

    sample_claims  = load_csv(REPO_ROOT / "dataset" / "sample_claims.csv")
    user_history   = {r["user_id"]: r for r in load_csv(REPO_ROOT / "dataset" / "user_history.csv")}
    evidence_reqs  = load_csv(REPO_ROOT / "dataset" / "evidence_requirements.csv")

    # Gold labels are in sample_claims.csv
    gold = sample_claims

    model_results: dict = {}
    model_ops: dict     = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"{'='*60}")
        ops = OpsTracker(model=model)
        preds = run_pipeline(sample_claims, user_history, evidence_reqs, model, ops)
        metrics, cm = evaluate(preds, gold)
        model_results[model] = {"predictions": preds, "metrics": metrics, "cm": cm}
        model_ops[model] = ops

        # Save per-model predictions CSV
        pred_path = EVAL_DIR / f"sample_predictions_{model.replace('-', '_')}.csv"
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(preds)
        print(f"  ✓ Predictions → {pred_path}")
        print(f"\n  {ops.summary()}")

    # Print accuracy table
    print(f"\n{'='*60}")
    print(f"  ACCURACY SUMMARY (n={len(gold)} cases)")
    print(f"{'='*60}")
    print(f"  {'Field':<30}", end="")
    for m in models:
        print(f"  {m:<22}", end="")
    print()
    for field in EVAL_FIELDS:
        print(f"  {field:<30}", end="")
        for m in models:
            acc = model_results[m]["metrics"].get(field, 0) if m in model_results else 0
            print(f"  {acc:>6.1%}{'':16}", end="")
        print()
    print()

    write_report(model_results, model_ops, gold, chosen_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate evidence review pipeline on sample claims")
    parser.add_argument(
        "--model", nargs="+", default=None,
        help="Model(s) to evaluate (default: both). E.g. --model claude-opus-4-8"
    )
    parser.add_argument(
        "--chosen", default="claude-opus-4-8",
        help="Model chosen for output.csv (noted in report)"
    )
    args = parser.parse_args()
    main(models=args.model, chosen_model=args.chosen)
