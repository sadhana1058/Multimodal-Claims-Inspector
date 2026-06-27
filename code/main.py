#!/usr/bin/env python3
"""Entry point: reads dataset/claims.csv → writes output.csv at repo root.

Usage:
    python code/main.py                         # default: claude-opus-4-8
    python code/main.py --model claude-sonnet-4-6
    python code/main.py --claims dataset/claims.csv --output output.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure code/ directory is on the Python path regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

# Load .env from repo root if present (ANTHROPIC_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

from ops_tracker import OpsTracker
from stage1_perception import analyze_claim
from stage2_decision import decide

REPO_ROOT = Path(__file__).parent.parent

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids", "valid_image", "severity",
]


def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def process_claims(
    claims: list,
    user_history: dict,
    evidence_reqs: list,
    model: str,
    ops: OpsTracker,
) -> list:
    rows_out = []
    for i, claim in enumerate(claims):
        user_id      = claim["user_id"]
        paths_raw    = claim["image_paths"]
        user_claim   = claim["user_claim"]
        claim_object = claim["claim_object"]

        image_paths = [p.strip() for p in paths_raw.split(";")]
        image_ids   = [Path(p).stem for p in image_paths]
        history     = user_history.get(user_id, {"history_flags": "none", "history_summary": ""})

        print(
            f"  [{i+1:02d}/{len(claims)}] {user_id} | {claim_object} "
            f"| {len(image_paths)} image(s) | {', '.join(image_ids)}"
        )

        # Stage 1 — VLM perception
        stage1 = analyze_claim(
            image_paths=image_paths,
            claim_text=user_claim,
            claim_object=claim_object,
            model=model,
            ops=ops,
            repo_root=REPO_ROOT,
        )

        # Stage 2 — deterministic decision
        result = decide(
            stage1=stage1,
            claim_object=claim_object,
            history_row=history,
            evidence_requirements=evidence_reqs,
            image_ids=image_ids,
        )

        print(
            f"         → {result.claim_status:25s} | "
            f"{result.issue_type:15s} on {result.object_part:15s} | "
            f"sev={result.severity}  esm={result.evidence_standard_met}"
        )

        rows_out.append({
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

    return rows_out


def write_output(rows: list, output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main(
    model: str = "claude-opus-4-8",
    claims_csv: Path = None,
    output_csv: Path = None,
) -> tuple:
    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ERROR: ANTHROPIC_API_KEY environment variable is not set.")

    if claims_csv is None:
        claims_csv = REPO_ROOT / "dataset" / "claims.csv"
    if output_csv is None:
        output_csv = REPO_ROOT / "output.csv"

    print(f"\n{'='*60}")
    print(f"  Multi-Modal Evidence Review Pipeline")
    print(f"{'='*60}")
    print(f"  Model      : {model}")
    print(f"  Claims CSV : {claims_csv}")
    print(f"  Output CSV : {output_csv}")
    print(f"{'='*60}\n")

    claims       = load_csv(claims_csv)
    user_history = {r["user_id"]: r for r in load_csv(REPO_ROOT / "dataset" / "user_history.csv")}
    evidence_reqs = load_csv(REPO_ROOT / "dataset" / "evidence_requirements.csv")

    ops = OpsTracker(model=model)
    rows = process_claims(claims, user_history, evidence_reqs, model, ops)

    write_output(rows, output_csv)

    print(f"\n✓ Output written → {output_csv}  ({len(rows)} rows)")
    print(f"\n{'─'*60}")
    print(ops.summary())
    print(f"{'─'*60}\n")

    return rows, ops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-modal evidence review pipeline")
    parser.add_argument("--model", default="claude-opus-4-8", help="Anthropic model ID")
    parser.add_argument("--claims", type=Path, default=None, help="Path to claims CSV")
    parser.add_argument("--output", type=Path, default=None, help="Path for output CSV")
    args = parser.parse_args()
    main(model=args.model, claims_csv=args.claims, output_csv=args.output)
