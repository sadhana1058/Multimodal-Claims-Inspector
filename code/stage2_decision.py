"""Stage 2: deterministic Python decision logic. Zero model calls."""
from dataclasses import dataclass

VLM_FLAGS = {
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present",
}
HISTORY_FLAGS = {"user_history_risk", "manual_review_required"}

# Semantic groups for "close enough to consider the same family" — used only for
# the object_consistent + claim_mismatch gate, NOT for the supported/contradicted ladder.
_ISSUE_GROUPS = [
    {"dent", "scratch"},
    {"crack", "glass_shatter"},
    {"broken_part", "missing_part"},
    {"torn_packaging", "crushed_packaging"},
    {"water_damage", "stain"},
]

_ISSUE_TO_FAMILY = {
    "dent":              "dent or scratch",
    "scratch":           "dent or scratch",
    "crack":             "crack, broken, or missing part",
    "glass_shatter":     "crack, broken, or missing part",
    "broken_part":       "crack, broken, or missing part",
    "missing_part":      "crack, broken, or missing part",
    "torn_packaging":    "crushed, torn, or seal damage",
    "crushed_packaging": "crushed, torn, or seal damage",
    "water_damage":      "water, stain, or label damage",
    "stain":             "water, stain, or label damage",
    "none":              "general claim review",
    "unknown":           "general claim review",
}


@dataclass
class DecisionResult:
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: str


# ──────────────────────────────────────────────────────────────────────────────

def decide(
    stage1: dict,
    claim_object: str,
    history_row: dict,
    evidence_requirements: list,
    image_ids: list,
) -> DecisionResult:

    images = stage1.get("images", [])
    syn    = stage1.get("synthesis", {})

    # ── Collect VLM quality flags ──
    # Note: text_instruction_present is included in quality_flags by Stage 1 when the VLM
    # detects adversarial embedded text. embedded_text_present=True alone (ordinary labels,
    # barcodes) does NOT add the flag — that would inflate risk_flags with false positives.
    all_flags: set = set()
    for img in images:
        all_flags.update(img.get("quality_flags", []))

    # ── Key synthesis values ──
    object_consistent = syn.get("object_consistent_across_images", True)
    best_ids          = syn.get("best_supporting_image_ids", [])
    visible_issue     = syn.get("visible_issue_type", "unknown")
    claimed_issue     = syn.get("claimed_issue_type", "unknown")
    visible_part      = syn.get("object_part", "unknown")
    claimed_part      = syn.get("claimed_object_part", "unknown")
    severity          = syn.get("severity", "unknown")
    free_text         = syn.get("free_text_reason", "").strip()

    # ── Derived booleans ──
    any_shows_object = any(img.get("shows_claimed_object", False) for img in images)
    all_shows_object = bool(images) and all(img.get("shows_claimed_object", False) for img in images)
    any_inspectable  = any(img.get("angle_inspectable", False) for img in images)
    any_good         = any(
        img.get("shows_claimed_object", False) and img.get("angle_inspectable", False)
        for img in images
    )
    all_blurry = bool(images) and all(
        "blurry_image" in img.get("quality_flags", []) for img in images
    )

    # ── Effective visible issue: if possible_manipulation is flagged the image
    #    can't be trusted → treat visible_issue as "none" for claim determination ──
    effective_vi = visible_issue if "possible_manipulation" not in all_flags else "none"

    # If text_instruction_present only appears on the images that supply the damage evidence,
    # and clean (non-tainted) images show no damage, the synthesis vi is from an adversarial
    # source → override effective_vi so the claim cannot be approved on tainted evidence alone.
    tainted_ids = {
        img["image_id"] for img in images
        if "text_instruction_present" in img.get("quality_flags", [])
    }
    if tainted_ids and len(images) > len(tainted_ids):
        clean_vi_list = [
            img.get("visible_issue_type", "unknown")
            for img in images
            if img["image_id"] not in tainted_ids
        ]
        clean_has_damage = any(v not in ("none", "unknown") for v in clean_vi_list)
        if not clean_has_damage and effective_vi not in ("none", "unknown"):
            effective_vi = "none"

    # ── valid_image ──
    valid_image: bool = bool(syn.get("valid_image", True))
    if "non_original_image" in all_flags:
        valid_image = False
    # All images blocked with no claimed object → not usable
    if images and all(
        not img.get("shows_claimed_object", False)
        and ("cropped_or_obstructed" in img.get("quality_flags", [])
             or "blurry_image" in img.get("quality_flags", []))
        for img in images
    ):
        valid_image = False

    # ── Evidence standard met ──
    # True when the image set is sufficient to make a definitive determination.
    if not object_consistent and len(images) > 1 and (
        effective_vi in ("none", "unknown") or
        ("claim_mismatch" in all_flags and not _issue_family_match(effective_vi, claimed_issue))
    ):
        evidence_standard_met = False
        esm_reason = "Images appear to show different objects; identity cannot be confirmed."
    elif not any_inspectable:
        evidence_standard_met = False
        esm_reason = f"No image provides an inspectable angle for the claimed {claimed_part or claim_object}."
    elif all_blurry:
        evidence_standard_met = False
        esm_reason = "All submitted images are too blurry to evaluate the claim."
    elif "wrong_object" in all_flags and not any_shows_object and any_inspectable:
        # Clear wrong-object but image IS clear → can say contradicted
        evidence_standard_met = True
        esm_reason = (
            "The image is clear enough to determine it does not show the claimed "
            f"{claim_object}; the claim can be evaluated as contradicted."
        )
    elif any_good:
        req_text = _find_requirement(evidence_requirements, claim_object, claimed_issue)
        evidence_standard_met = True
        esm_reason = (
            f"The {visible_part} is visible at an inspectable angle. {req_text}"
        ).strip()
    elif any_inspectable and any_shows_object:
        # Each property is True but not in the same image
        evidence_standard_met = True
        esm_reason = (
            f"The {claim_object} and an inspectable angle are present across images, "
            f"providing sufficient context to evaluate the claim."
        )
    else:
        evidence_standard_met = False
        esm_reason = (
            f"The submitted images do not meet the minimum evidence requirement "
            f"for a {claim_object} {claimed_issue} claim."
        )

    # ── Claim-status ladder ──
    claim_status: str = ""
    supporting: list  = []

    # Pre-Rule X: Consistent images + explicit claim mismatch from a different family
    # → contradicted regardless of angle/visibility (handles case where VLM flags clear mismatch
    #   but angle_inspectable=False blocks the normal path)
    if (
        object_consistent
        and "claim_mismatch" in all_flags
        and not _issue_family_match(effective_vi, claimed_issue)
    ):
        claim_status = "contradicted"
        justification = (
            f"The visible content ({effective_vi} on {visible_part}) clearly does not match "
            f"the claimed {claimed_issue} on {claimed_part}. {free_text}"
        )
        supporting = best_ids or list(image_ids)

    # Pre-Rule Y: No claimed object visible but claim_mismatch is set
    # → the image shows something identifiably wrong → contradicted
    elif (
        not any_shows_object
        and "claim_mismatch" in all_flags
    ):
        claim_status = "contradicted"
        all_flags.add("wrong_object")
        justification = (
            f"The submitted image does not show the claimed {claim_object}; "
            f"the visible object contradicts the claim. {free_text}"
        )
        supporting = best_ids or list(image_ids)

    # Rule A: Inconsistent images — only force not_enough_information when
    # the inconsistency matters (no clear damage, or cross-family issue-type mismatch,
    # or some images don't show the claimed object at all — mixed identity set).
    # Exception: if the claimed object IS visible and effective_vi="none", Rule E will handle
    # it as contradicted (we CAN determine the claim is false even across inconsistent images).
    elif not object_consistent and len(images) > 1 and (
        effective_vi in ("none", "unknown") or
        ("claim_mismatch" in all_flags and not _issue_family_match(effective_vi, claimed_issue)) or
        ("claim_mismatch" in all_flags and not all_shows_object)
    ) and not (any_shows_object and effective_vi == "none"):
        claim_status = "not_enough_information"
        all_flags |= {"wrong_object", "claim_mismatch", "manual_review_required"}
        justification = (
            f"The submitted images appear to show different objects, "
            f"so the {claim_object} identity cannot be confirmed. {free_text}"
        )
        supporting = list(image_ids)

    # Rule B: No inspectable angle at all
    elif not any_inspectable:
        claim_status = "not_enough_information"
        justification = (
            f"No submitted image provides an angle sufficient to inspect the "
            f"claimed {claimed_part or claim_object}. {free_text}"
        )
        supporting = []

    # Rule C: Image is inspectable but does not show claimed object
    elif not any_shows_object:
        if ("wrong_object" in all_flags or "wrong_object_part" in all_flags) and any_inspectable:
            # Image shows something clearly identifiable but it's the wrong thing
            claim_status = "contradicted"
            justification = (
                f"The submitted image does not show the claimed {claim_object}; "
                f"the visible content does not support the claim. {free_text}"
            )
            supporting = best_ids or list(image_ids)
        else:
            claim_status = "not_enough_information"
            justification = (
                f"No submitted image clearly shows the claimed {claim_object}. {free_text}"
            )
            supporting = []

    # Rule D: Shows object, angle is good, but NOT in the same image (edge case)
    # Use claim_mismatch or wrong_object_part to still produce contradicted
    elif not any_good:
        if "claim_mismatch" in all_flags or (
            "wrong_object_part" in all_flags and effective_vi in ("none", "unknown")
        ):
            claim_status = "contradicted"
            justification = (
                f"The submitted images do not show the claimed {claimed_part} "
                f"damage clearly; the visible content contradicts the claim. {free_text}"
            )
            supporting = best_ids or list(image_ids)
        else:
            claim_status = "not_enough_information"
            justification = (
                f"The submitted images do not provide an angle sufficient to "
                f"inspect the claimed {claimed_part or claim_object} damage. {free_text}"
            )
            supporting = []

    # ── From here on any_good = True ──

    # Rule E: No visible damage when damage is claimed → contradicted.
    # Exception: missing_part claims where damage_not_visible is set — the absence of
    # contents cannot be proven from exterior images alone → not_enough_information.
    elif effective_vi == "none" and claimed_issue not in ("none", "unknown"):
        if claimed_issue == "missing_part" and "damage_not_visible" in all_flags:
            claim_status = "not_enough_information"
            justification = (
                f"The images do not show the package contents clearly enough to "
                f"confirm or deny the missing-item claim. {free_text}"
            )
            supporting = []
        else:
            claim_status = "contradicted"
            justification = (
                f"The {visible_part} is clearly visible but shows no damage "
                f"consistent with the claimed {claimed_issue}. {free_text}"
            )
            supporting = best_ids or [img["image_id"] for img in images if img.get("angle_inspectable")]

    # Rule F: Inferred-but-not-visible missing-part claim
    elif effective_vi == "missing_part" and "damage_not_visible" in all_flags:
        claim_status = "not_enough_information"
        justification = (
            f"The contents or missing item cannot be clearly observed from the "
            f"submitted images. {free_text}"
        )
        supporting = []

    # Rule G: Explicit claim mismatch that cannot be reconciled by issue-type family
    elif "claim_mismatch" in all_flags and not _issue_family_match(effective_vi, claimed_issue):
        claim_status = "contradicted"
        justification = (
            f"The visible damage ({effective_vi} on {visible_part}) does not match "
            f"the claimed {claimed_issue} on {claimed_part}. {free_text}"
        )
        supporting = best_ids or [img["image_id"] for img in images if img.get("shows_claimed_object")]

    # Rule H: Effective visible issue is unknown → not enough information
    elif effective_vi == "unknown":
        claim_status = "not_enough_information"
        justification = (
            f"The submitted images do not provide sufficient evidence to verify "
            f"the {claimed_issue} claim on {claimed_part}. {free_text}"
        )
        supporting = []

    # Rule I: Damage IS visible (any damage type), no explicit contradiction → supported
    # NOTE: We do NOT require exact issue-type or part match here. The VLM sets
    # claim_mismatch when the damage clearly doesn't support the claim. Absence of
    # claim_mismatch means the visible damage is plausibly consistent with the claim.
    else:
        claim_status = "supported"
        justification = (
            f"The {visible_part} is visible and the {effective_vi} is consistent "
            f"with the claim. {free_text}"
        )
        supporting = best_ids or [img["image_id"] for img in images if img.get("shows_claimed_object")]

    # ── Risk flags ──
    risk_flags = sorted(all_flags & VLM_FLAGS)

    pre_history_status = claim_status

    history_flags_raw = (history_row.get("history_flags") or "none").strip()
    for hf in history_flags_raw.split(";"):
        hf = hf.strip()
        if hf and hf != "none" and hf in HISTORY_FLAGS and hf not in risk_flags:
            risk_flags.append(hf)

    # HARD CONSTRAINT: user history must not change a clear visual verdict
    assert claim_status == pre_history_status, (
        f"BUG: history changed claim_status from {pre_history_status!r} to {claim_status!r}"
    )

    # ── Severity: demote when no determination possible ──
    if claim_status == "not_enough_information" and effective_vi in ("none", "unknown"):
        severity = "unknown"

    # ── supporting_image_ids ──
    valid_set = set(image_ids)
    supporting_filtered = [x for x in supporting if x in valid_set]
    supporting_str = ";".join(supporting_filtered) if supporting_filtered else "none"

    return DecisionResult(
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=esm_reason.strip(),
        risk_flags=risk_flags,
        issue_type=visible_issue,           # always report what was SEEN (not effective_vi)
        object_part=visible_part,
        claim_status=claim_status,
        claim_status_justification=justification.strip(),
        supporting_image_ids=supporting_str,
        valid_image=valid_image,
        severity=severity,
    )


# ──────────────────────────────────────────────────────────────────────────────

def _issue_family_match(vi: str, ci: str) -> bool:
    """True if vi and ci are in the same semantic family (or equal, or either is unknown)."""
    if vi == ci:
        return True
    if "unknown" in (vi, ci) or "none" in (vi, ci):
        return False
    for group in _ISSUE_GROUPS:
        if vi in group and ci in group:
            return True
    return False


def _find_requirement(requirements: list, claim_object: str, issue_type: str) -> str:
    family = _ISSUE_TO_FAMILY.get(issue_type, "general claim review")
    for req in requirements:
        obj = req.get("claim_object", "all")
        if obj in (claim_object, "all") and family in req.get("applies_to", ""):
            return req.get("minimum_image_evidence", "")
    for req in requirements:
        if req.get("applies_to") == "reviewability":
            return req.get("minimum_image_evidence", "")
    return ""
