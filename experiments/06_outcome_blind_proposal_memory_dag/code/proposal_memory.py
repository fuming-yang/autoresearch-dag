"""Online proposal memory and conservative novelty gate for DAG-C.

The memory contains only proposals observed during the current DAG-C run.
It deliberately stores no DAG-B outcomes and never consults validation loss
when deciding whether a proposal is novel.
"""

from __future__ import annotations

import json
import re
from collections import Counter


PROPOSAL_RETRY_K = 3
_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9_.=:+%→\-]+")


def _text(value) -> str:
    return str(value or "").strip()


def normalize(value) -> str:
    value = _text(value).lower().replace("->", "→")
    value = _WS.sub(" ", value)
    return _NONWORD.sub("", value)


def proposal_record(decision: dict) -> dict:
    """Return a compact, JSON-safe proposal representation from a decision."""
    proposal = decision.get("proposal") or {}
    family = _text(proposal.get("family")) or "UNSPECIFIED"
    mechanism = _text(proposal.get("mechanism")) or family
    transition = _text(proposal.get("effective_transition")) or _text(proposal.get("transition"))
    summary = _text(proposal.get("summary")) or _text(decision.get("hypothesis"))
    justification = _text(proposal.get("mechanistic_justification"))
    return {
        "family": family,
        "mechanism": mechanism,
        "effective_transition": transition,
        "summary": summary,
        "mechanistic_justification": justification,
        "family_key": normalize(family),
        "mechanism_key": normalize(mechanism),
        "transition_key": normalize(transition),
    }


def _meaningful_justification(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]{3,}", text)
    return len(words) >= 5


def classify_proposal(proposal: dict, history: list[dict]) -> dict:
    """Classify a proposal without using results, parents, or future information."""
    current = proposal_record({"proposal": proposal})
    exact = []
    same_mechanism = []
    same_family = []
    for prior in history:
        prior = proposal_record({"proposal": prior.get("proposal", prior)})
        if current["transition_key"] and current["transition_key"] == prior["transition_key"]:
            exact.append(prior)
        if current["mechanism_key"] == prior["mechanism_key"]:
            same_mechanism.append(prior)
        if current["family_key"] == prior["family_key"]:
            same_family.append(prior)
    if exact:
        return {"classification": "EXACT_DUPLICATE", "closest": exact[-1],
                "rejected": True, "reason": "same effective parameter transition"}
    if same_mechanism and current["mechanism_key"]:
        if _meaningful_justification(current["mechanistic_justification"]):
            return {"classification": "REFINEMENT", "closest": same_mechanism[-1],
                    "rejected": False, "reason": "different transition with explicit mechanistic justification"}
        return {"classification": "NEAR_DUPLICATE", "closest": same_mechanism[-1],
                "rejected": True, "reason": "same mechanism without a meaningful mechanistic distinction"}
    if same_family:
        return {"classification": "NEW_VARIANT_WITHIN_EXISTING_FAMILY", "closest": same_family[-1],
                "rejected": False, "reason": "new mechanism within an explored family"}
    return {"classification": "NEW_MECHANISM", "closest": None, "rejected": False,
            "reason": "no prior family or mechanism match"}


def history_summary(state: dict) -> str:
    """Build the compact prompt memory from DAG-C proposals only."""
    history = state.get("proposal_history", [])
    if not history:
        return "SEARCH HISTORY\n(no DAG-C proposals have been attempted yet)"
    counts = Counter(item.get("proposal", {}).get("family", "UNSPECIFIED") for item in history)
    lines = ["SEARCH HISTORY", "Families already explored:"]
    for family, count in sorted(counts.items()):
        members = [item for item in history if item.get("proposal", {}).get("family") == family]
        lines.append(f"- {family}: attempts={count}")
        for item in members[-5:]:
            proposal = item.get("proposal", {})
            lines.append(f"  transition={proposal.get('effective_transition', '')}; "
                         f"classification={item.get('novelty_classification', '')}")
    rejected = [item for item in history if item.get("novelty_rejected")]
    if rejected:
        lines.append("Recent novelty rejections:")
        for item in rejected[-5:]:
            lines.append(f"- {item.get('rejection_reason', 'unspecified')}")
    lines.append("Use this history to propose a genuinely different mechanism. "
                 "Avoid exact or near-equivalent transitions unless a concrete mechanistic reason explains the difference.")
    return "\n".join(lines)


def retry_feedback(rejected_attempts: list[dict], state: dict) -> str:
    """Build operation-local feedback for the next proposal retry."""
    if not rejected_attempts:
        return ""
    lines = ["PREVIOUS REJECTED ATTEMPTS IN THIS OPERATION"]
    for attempt in rejected_attempts:
        proposal = proposal_record({"proposal": attempt.get("proposal", {})})
        lines.extend([
            "",
            f"Attempt {attempt['proposal_attempt_index']}:",
            f"Proposal: {proposal['summary']} (effective transition: {proposal['effective_transition']})",
            f"Mechanism family: {proposal['family']}",
            f"Mechanism: {proposal['mechanism']}",
            f"Novelty classification: {attempt['classification']}",
            f"Rejected because: {attempt['reason']}",
            "Closest previous proposal: " + json.dumps(
                attempt.get("closest_previous_proposal"), sort_keys=True
            ),
        ])

    family_counts = Counter(
        item.get("proposal", {}).get("family", "UNSPECIFIED")
        for item in state.get("proposal_history", [])
    )
    counts = ", ".join(f"{family}={count}" for family, count in sorted(family_counts.items()))
    lines.extend(["", f"Current DAG-C mechanism-family frequencies: {counts or '(none)'}", ""])
    preparation_failures = [
        attempt for attempt in rejected_attempts
        if attempt.get("classification") == "PREPARATION_FAILURE"
    ]
    novelty_failures = len(rejected_attempts) - len(preparation_failures)
    if preparation_failures:
        lines.extend([
            "One or more previous attempts failed proposal preparation or schema validation.",
            "Correct every reported preparation error and return a complete, valid decision and proposal structure.",
            "A formatting correction does not waive novelty requirements; the corrected proposal must still be mechanistically meaningful.",
            "",
        ])
    if novelty_failures == 1 and len(rejected_attempts) == 1:
        lines.extend([
            "Your previous proposal was rejected because it was too similar to an already explored modification.",
            "Do not merely change the numerical value of the same parameter.",
            "For the next proposal, seek a meaningfully different intervention direction or mechanism.",
            "Prefer a different mechanism family when possible.",
            "A proposal in the same broad family is allowed only if it introduces a genuinely different mechanism rather than a trivial parameter variation.",
        ])
    elif novelty_failures:
        lines.extend([
            "The previous retry attempts in this operation were rejected for insufficient novelty.",
            "Do not continue making numerical variations of the same intervention.",
            "Choose a substantially different mechanism/intervention direction.",
            "Prefer mechanism families that (1) have not yet been explored in DAG-C, or (2) have been explored much less frequently.",
            "Do not blindly ban an entire family: a same-family proposal remains valid if it is genuinely mechanistically different.",
        ])
    return "\n".join(lines)


def append_history(state: dict, *, proposal: dict, classification: str, rejected: bool,
                   reason: str, operation_id: str, attempt_index: int, trained: bool = False,
                   candidate_id: str | None = None, beat_parent: bool | None = None) -> dict:
    record = {
        "operation_id": operation_id,
        "proposal_attempt_index": attempt_index,
        "proposal": proposal_record({"proposal": proposal}),
        "novelty_classification": classification,
        "novelty_rejected": bool(rejected),
        "rejection_reason": reason if rejected else None,
        "trained": bool(trained),
        "candidate_id": candidate_id,
        "beat_parent": beat_parent,
    }
    state.setdefault("proposal_history", []).append(record)
    return record


def json_summary(state: dict) -> str:
    return json.dumps({"proposal_history": state.get("proposal_history", [])}, sort_keys=True)
