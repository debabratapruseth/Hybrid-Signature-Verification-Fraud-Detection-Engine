"""OpenAI-assisted reports for the three-branch fusion result."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


def build_report_evidence(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Keep model input focused; omit large image descriptors and file data.

    The report model receives conclusions and supporting measurements, not raw
    image arrays, full Fourier vectors, or local file contents. This reduces
    data exposure and makes invented claims easier to detect.
    """
    evidence = bundle["evidence"]
    branch1, branch2, branch3 = (
        evidence["branch1"],
        evidence["branch2"],
        evidence["branch3"],
    )
    return {
        "case_id": bundle["fusion"]["case_id"],
        "deterministic_fusion": bundle["fusion"],
        "branch1_learned_verification": {
            key: branch1.get(key)
            for key in (
                "decision",
                "overall_risk",
                "reliability",
                "component_risks",
                "reasons",
                "warnings",
            )
        },
        "branch2_structural_ai": {
            "reference_variation": branch2.get("reference_variation", {}),
            "reference_comparisons": branch2.get(
                "reference_comparisons",
                branch2.get("comparisons", []),
            ),
            "warnings": branch2.get("warnings", []),
        },
        "branch3_document_forensics": {
            key: (
                branch3.get("risk", {}).get(key)
                if key in {
                    "decision",
                    "overall_screening_risk",
                    "component_risks",
                    "evidence_availability",
                    "evidence_coverage",
                    "reasons",
                    "warnings",
                }
                else branch3.get(key)
            )
            for key in (
                "decision",
                "overall_screening_risk",
                "component_risks",
                "copy_paste",
                "compression",
                "noise_analysis",
                "evidence_availability",
                "evidence_coverage",
                "reasons",
                "warnings",
            )
        },
    }


def _instructions(audience: str) -> str:
    """Build the audience-specific prompt plus shared evidentiary guardrails."""

    common = """
You are drafting a report for a signature-forensic research prototype.
Use only the supplied JSON evidence. Preserve reported numbers exactly. Do not
invent facts, thresholds, tests, causation, or certainty.

The deterministic fusion conclusion is authoritative: explain it without
replacing or recalculating it. Absence of an elevated indicator is not proof
that editing did not occur. Model outputs are screening evidence, not legal or
forensic conclusions.

Never state that a signature or document is genuine, authentic, forged,
fraudulent, copied, or tampered. State clearly that this is not an authenticity,
forgery, intent, or legal-validity determination and that consequential use
requires qualified human examination.

Write clean Markdown without JSON or a preamble.
""".strip()
    if audience == "senior":
        return common + """

Use these headings exactly:
# Senior Stakeholder Report
## Executive conclusion
## Three-branch summary
## Business implications
## Recommended actions
## Limitations and disclaimer

Write for a senior reader with no background in artificial intelligence,
handwriting examination, statistics, or software.

Use plain everyday English and keep the complete report between 300 and 450
words. Lead with the practical conclusion and what the reader should do next.

Describe the branches using only these business-friendly names:
- Branch 1: signature appearance comparison
- Branch 2: signature shape and stroke comparison
- Branch 3: document editing checks

Do not use technical terms such as Siamese network, embedding, cosine
similarity, ROC, AUC, EER, false-acceptance rate, calibration, ORB, RANSAC,
ELA, residual noise, Hu moments, Fourier descriptors, Shape Context, skeleton
graph, or percentile. Translate their meaning into plain language instead.

Mention at most three numerical values in the entire report. Include the fused
review-risk score, but immediately explain that it is a workflow-priority score
and not the probability that the signature is forged.

In "Three-branch summary", use exactly three short bullet points: one per
branch. Each bullet must state what the branch observed and what that means in
plain language.

In "Business implications", explain the operational consequence without
describing algorithms. In "Recommended actions", provide no more than three
specific action bullets. Keep limitations short and direct.

Prefer phrases such as "the signatures showed differences", "the available
checks did not flag obvious digital reuse", and "a person should review the
originals". Avoid phrases such as "the model generalized weakly", "effective
weights", "feature evidence", or "structural descriptor".
"""
    return common + """

Use these headings exactly:
# Technical Fusion Report
## Scope and evidence
## Branch 1 — Learned verification and quality
## Branch 2 — Structural AI
## Branch 3 — Document forensics
## Fusion method and result
## Evidence gaps
## Validation recommendations
## Technical disclaimer

Write for technical reviewers. Discuss reliability, corroborating or conflicting
indicators, quality limitations, fusion weights, and why the conclusion follows.
Do not claim statistical calibration unless it is supplied.
"""


def _validate_report(text: str, audience: str) -> None:
    """Reject malformed reports and unsupported affirmative verdict language.

    Negated disclaimer language is allowed. For example, "this does not
    determine whether the signature is authentic" contains verdict vocabulary
    but explicitly denies that conclusion.
    """

    headings = (
        [
            "# Senior Stakeholder Report",
            "## Executive conclusion",
            "## Three-branch summary",
            "## Business implications",
            "## Recommended actions",
            "## Limitations and disclaimer",
        ]
        if audience == "senior"
        else [
            "# Technical Fusion Report",
            "## Scope and evidence",
            "## Branch 1 — Learned verification and quality",
            "## Branch 2 — Structural AI",
            "## Branch 3 — Document forensics",
            "## Fusion method and result",
            "## Evidence gaps",
            "## Validation recommendations",
            "## Technical disclaimer",
        ]
    )
    missing = [heading for heading in headings if heading not in text]
    if missing:
        raise ValueError("OpenAI report is missing headings: " + ", ".join(missing))

    verdict_patterns = [
        r"\b(?:signature|document)\s+(?:is|was)\s+"
        r"(?:genuine|authentic|forged|fraudulent|tampered)\b",
        r"\bconfirmed\s+(?:forgery|fraud|tampering|authenticity)\b",
    ]
    negation_patterns = [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdoes not\b",
        r"\bdid not\b",
        r"\bwithout\b",
        r"\bneither\b",
        r"\bnor\b",
        r"\bwhether\b",
        r"\bnot evidence of\b",
        r"\bnot proof of\b",
    ]
    for pattern in verdict_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Examine the surrounding clause. This permits required disclaimers
            # such as "does not determine whether the signature is authentic"
            # while still rejecting a direct affirmative conclusion.
            clause_start = max(
                text.rfind("\n", 0, match.start()),
                text.rfind(".", 0, match.start()),
                text.rfind(";", 0, match.start()),
            )
            prefix = text[max(clause_start + 1, match.start() - 120):match.start()]
            is_negated = any(
                re.search(negation, prefix, re.IGNORECASE)
                for negation in negation_patterns
            )
            if not is_negated:
                raise ValueError(
                    "OpenAI report contains an unsupported affirmative "
                    "forensic verdict."
                )
    lowered = text.lower()
    if "not" not in lowered or "authentic" not in lowered:
        raise ValueError("Report must explicitly say it is not an authenticity determination.")


def _create_report(client: Any, model: str, audience: str, evidence: Dict[str, Any]) -> str:
    """Make one Responses API request, extract Markdown, and validate it."""

    response = client.responses.create(
        model=model,
        instructions=_instructions(audience),
        input="Prepare the requested report from:\n" + json.dumps(evidence, indent=2),
    )
    text = response.output_text.strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty report.")
    _validate_report(text, audience)
    return text


def generate_openai_reports(
    bundle: Dict[str, Any],
    *,
    model: str,
    api_key_environment_variable: str = "OPENAI_API_KEY",
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate both mandatory reports; there is deliberately no local fallback.

    The caller explicitly requested LLM-authored reports. Failing loudly is
    safer than silently replacing them with differently authored local text.
    Numerical fusion remains available before this function is called.
    """
    if client is None:
        api_key = os.environ.get(api_key_environment_variable)
        if not api_key:
            raise EnvironmentError(
                f"{api_key_environment_variable} is not set. "
                "Enter the OpenAI API key securely before generating reports."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI SDK with: pip install openai") from exc
        client = OpenAI(api_key=api_key)

    evidence = build_report_evidence(bundle)
    try:
        senior = _create_report(client, model, "senior", evidence)
        technical = _create_report(client, model, "technical", evidence)
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"OpenAI report generation failed: {exc}") from exc

    return {
        "senior_report": senior,
        "technical_report": technical,
        "report_evidence": evidence,
        "metadata": {
            "provider": "OpenAI",
            "model": model,
            "reports_generated": 2,
            "images_sent_to_model": False,
        },
    }
