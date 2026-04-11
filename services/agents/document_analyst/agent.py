"""Document Analyst — reads financial documents and surfaces hidden risks.

This agent accepts an uploaded document image (lease, insurance policy,
credit-card agreement, medical bill, etc.) and performs deep analysis to
find what the user would never spot themselves: hidden fees, auto-renewal
traps, penalty triggers, coverage gaps, and billing errors.

The analysis is **document-type-specific**.  NODE 3 dispatches to a
tailored Claude prompt for each document type so the questions asked of
the LLM are always domain-relevant.  For example a lease prompt asks
about early-termination penalties and rent-increase clauses, while a
medical-bill prompt looks for billing errors and financial-assistance
eligibility.

Graph pipeline::

    initialise
      → validate_and_classify
      → extract_document_data
      → deep_risk_analysis
      → create_alerts_and_reminders
      → generate_plain_english_summary
      → finalise → END
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph

from services.agents.base_agent import AgentState, BaseAgent

logger = logging.getLogger(__name__)

KNOWN_DOCUMENT_TYPES: set[str] = {
    "lease",
    "insurance_policy",
    "credit_card_agreement",
    "medical_bill",
    "utility_bill",
    "loan_agreement",
    "bank_statement",
    "tax_document",
    "other",
}


def _safe_json_parse(raw: str) -> dict[str, Any]:
    """Parse a JSON string, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ------------------------------------------------------------------
# Document-type-specific prompts for NODE 3.
#
# Each prompt is designed to extract the financial risks most relevant
# to that document type.  The LLM returns structured JSON so downstream
# nodes can create typed alerts and calendar reminders.
# ------------------------------------------------------------------

def _lease_prompt(data: dict[str, Any]) -> str:
    """Prompt for residential / commercial lease analysis.

    Targets: hidden fees, auto-renewal traps, early-termination penalties,
    rent-increase clauses, maintenance responsibilities, security-deposit
    return conditions, and general red flags.
    """
    return (
        f"Analyze this lease document data: {json.dumps(data)}\n\n"
        "Find and return JSON with:\n"
        "- hidden_fees: list of fees not prominently disclosed\n"
        "- auto_renewal_clause: {\"exists\": bool, \"details\": str}\n"
        "- early_termination_penalty: {\"amount\": str, \"conditions\": str}\n"
        "- rent_increase_clause: {\"percentage\": str, \"conditions\": str}\n"
        "- maintenance_responsibilities: {\"tenant\": list, \"landlord\": list}\n"
        "- security_deposit_conditions: exact return conditions\n"
        "- red_flags: [{\"title\": str, \"description\": str, "
        "\"severity\": \"high\"|\"medium\"|\"low\"}]\n"
        "- important_dates: [{\"event\": str, \"date\": str, "
        "\"amount\": float|null}]\n"
        "- key_amounts: [{\"label\": str, \"amount\": float}]"
    )


def _insurance_prompt(data: dict[str, Any]) -> str:
    """Prompt for insurance policy analysis.

    Targets: coverage gaps, exclusion clauses, deductible details,
    premium-increase triggers, cancellation terms, claim-filing deadlines,
    and general red flags.
    """
    return (
        f"Analyze this insurance policy: {json.dumps(data)}\n\n"
        "Find and return JSON with:\n"
        "- coverage_gaps: list of situations NOT covered\n"
        "- exclusions: specific exclusion clauses\n"
        "- deductible_amounts: {\"per_incident\": float, "
        "\"annual_max\": float}\n"
        "- premium_increase_triggers: conditions that raise premiums\n"
        "- cancellation_terms: how and when the policy can be cancelled\n"
        "- claim_filing_deadlines: time limits for filing claims\n"
        "- red_flags: [{\"title\": str, \"description\": str, "
        "\"severity\": \"high\"|\"medium\"|\"low\"}]\n"
        "- important_dates: [{\"event\": str, \"date\": str, "
        "\"amount\": float|null}]\n"
        "- key_amounts: [{\"label\": str, \"amount\": float}]"
    )


def _credit_card_prompt(data: dict[str, Any]) -> str:
    """Prompt for credit-card agreement analysis.

    Targets: APR tiers (purchase / cash-advance / penalty), penalty
    triggers, the full fee schedule, grace-period nuances, the
    minimum-payment trap, reward-programme limitations, and red flags.
    """
    return (
        f"Analyze this credit card agreement: {json.dumps(data)}\n\n"
        "Find and return JSON with:\n"
        "- apr_rates: {\"purchase\": str, \"cash_advance\": str, "
        "\"penalty\": str}\n"
        "- penalty_triggers: conditions that trigger the penalty APR\n"
        "- fee_schedule: {\"annual\": float, \"late\": float, "
        "\"foreign_transaction\": str}\n"
        "- grace_period: exact terms and exceptions\n"
        "- minimum_payment_trap: how minimum payments extend debt\n"
        "- reward_program_limitations: restrictions on earning/redeeming\n"
        "- red_flags: [{\"title\": str, \"description\": str, "
        "\"severity\": \"high\"|\"medium\"|\"low\"}]\n"
        "- important_dates: [{\"event\": str, \"date\": str, "
        "\"amount\": float|null}]\n"
        "- key_amounts: [{\"label\": str, \"amount\": float}]"
    )


def _medical_bill_prompt(data: dict[str, Any]) -> str:
    """Prompt for medical-bill analysis.

    Targets: itemised charge verification, potential billing errors,
    insurance-application accuracy, negotiation opportunities,
    financial-assistance eligibility, and payment-plan options.
    """
    return (
        f"Analyze this medical bill: {json.dumps(data)}\n\n"
        "Find and return JSON with:\n"
        "- itemized_charges: [{\"description\": str, \"amount\": float, "
        "\"seems_reasonable\": bool}]\n"
        "- potential_billing_errors: list of suspicious charges\n"
        "- insurance_applied_correctly: {\"correct\": bool, "
        "\"issues\": list}\n"
        "- negotiation_opportunities: specific items to dispute\n"
        "- financial_assistance_eligibility: programmes to apply for\n"
        "- payment_plan_options: recommended payment strategies\n"
        "- red_flags: [{\"title\": str, \"description\": str, "
        "\"severity\": \"high\"|\"medium\"|\"low\"}]\n"
        "- important_dates: [{\"event\": str, \"date\": str, "
        "\"amount\": float|null}]\n"
        "- key_amounts: [{\"label\": str, \"amount\": float}]"
    )


def _generic_prompt(data: dict[str, Any], doc_type: str) -> str:
    """Fallback prompt for document types without a specialist template."""
    return (
        f"Analyze this {doc_type} document: {json.dumps(data)}\n\n"
        "Find and return JSON with:\n"
        "- key_amounts: [{\"label\": str, \"amount\": float}]\n"
        "- important_dates: [{\"event\": str, \"date\": str, "
        "\"amount\": float|null}]\n"
        "- obligations: list of financial obligations\n"
        "- red_flags: [{\"title\": str, \"description\": str, "
        "\"severity\": \"high\"|\"medium\"|\"low\"}]\n"
        "- summary_points: top 5 things the user needs to know"
    )


_RISK_PROMPTS: dict[str, Any] = {
    "lease": _lease_prompt,
    "insurance_policy": _insurance_prompt,
    "credit_card_agreement": _credit_card_prompt,
    "medical_bill": _medical_bill_prompt,
}


class DocumentAnalystAgent(BaseAgent):
    """Reads uploaded financial documents and surfaces hidden risks.

    Supports specialised analysis for leases, insurance policies,
    credit-card agreements, and medical bills.  All other document types
    receive a generic financial-risk scan.
    """

    def __init__(self) -> None:
        super().__init__(
            name="document_analyst",
            description=(
                "Reads uploaded financial documents — leases, insurance "
                "policies, contracts — and surfaces hidden fees, renewal "
                "traps, and financial risks"
            ),
        )
        self._llm = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Graph wiring
    # ------------------------------------------------------------------

    def define_nodes(self, builder: StateGraph) -> None:
        """Register the five processing nodes."""
        builder.add_node("validate_and_classify", self._validate_and_classify)
        builder.add_node("extract_document_data", self._extract_document_data)
        builder.add_node("deep_risk_analysis", self._deep_risk_analysis)
        builder.add_node(
            "create_alerts_and_reminders",
            self._create_alerts_and_reminders,
        )
        builder.add_node(
            "generate_plain_english_summary",
            self._generate_plain_english_summary,
        )

    def define_edges(self, builder: StateGraph) -> None:
        """Wire the linear pipeline."""
        builder.add_edge("initialise", "validate_and_classify")
        builder.add_edge("validate_and_classify", "extract_document_data")
        builder.add_edge("extract_document_data", "deep_risk_analysis")
        builder.add_edge("deep_risk_analysis", "create_alerts_and_reminders")
        builder.add_edge(
            "create_alerts_and_reminders",
            "generate_plain_english_summary",
        )
        builder.add_edge("generate_plain_english_summary", "finalise")
        builder.add_edge("finalise", END)

    # ------------------------------------------------------------------
    # NODE 1: validate_and_classify
    # ------------------------------------------------------------------

    async def _validate_and_classify(self, state: AgentState) -> dict[str, Any]:
        """Validate the input payload and classify the document type.

        If the caller already provided ``document_type`` it is accepted
        directly.  Otherwise Claude Vision classifies the image.
        """
        inp: dict[str, Any] = state.get("input", {})
        image_b64: str | None = inp.get("image_base64")

        if not image_b64:
            return self.set_error(state, "No document provided")

        doc_type: str | None = inp.get("document_type")

        if not doc_type:
            prompt = (
                "Look at this document image. What type of financial "
                "document is this? Return JSON only: "
                '{{"document_type": str, "confidence": float}} '
                "where document_type is one of: "
                + ", ".join(sorted(KNOWN_DOCUMENT_TYPES))
            )
            try:
                response = self._llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": inp.get(
                                            "image_type", "image/jpeg"
                                        ),
                                        "data": image_b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                parsed = _safe_json_parse(response.content[0].text)
                doc_type = parsed.get("document_type", "other")
            except Exception as exc:
                logger.warning("Document classification failed: %s", exc)
                doc_type = "other"

        if doc_type not in KNOWN_DOCUMENT_TYPES:
            doc_type = "other"

        step = f"Document classified as: {doc_type}"
        logger.info(step)
        return {
            "input": {**inp, "document_type": doc_type},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 2: extract_document_data
    # ------------------------------------------------------------------

    async def _extract_document_data(self, state: AgentState) -> dict[str, Any]:
        """Extract structured data from the document image via Vision MCP."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})

        try:
            extracted: dict[str, Any] = await self.call_tool(
                "vision_mcp",
                "extract_document_info",
                {
                    "image_base64": inp["image_base64"],
                    "document_type": inp["document_type"],
                },
            )
        except RuntimeError as exc:
            logger.error("Document extraction failed: %s", exc)
            return self.set_error(state, f"Could not read document: {exc}")

        key_amounts = extracted.get("key_amounts", [])
        important_dates = extracted.get("important_dates", [])

        step = (
            f"Extracted {len(key_amounts)} amounts, "
            f"{len(important_dates)} dates"
        )
        logger.info(step)
        return {
            "input": {**inp, "extracted_data": extracted},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 3: deep_risk_analysis
    # ------------------------------------------------------------------

    async def _deep_risk_analysis(self, state: AgentState) -> dict[str, Any]:
        """Run a document-type-specific deep analysis via Claude.

        This node dispatches to a **tailored prompt** per document type so
        the questions asked of the LLM are always domain-relevant:

        * **lease** — hidden fees, auto-renewal, early termination,
          rent-increase clause, maintenance split, deposit conditions.
        * **insurance_policy** — coverage gaps, exclusions, deductible
          tiers, premium-increase triggers, claim-filing deadlines.
        * **credit_card_agreement** — APR tiers, penalty triggers, fee
          schedule, grace period, minimum-payment trap, reward limits.
        * **medical_bill** — itemised charges, billing errors,
          insurance accuracy, negotiation opportunities, assistance.
        * **other / fallback** — generic financial obligations, key
          amounts, dates, and red flags.

        Each prompt requests structured JSON so downstream nodes can
        create typed alerts and calendar reminders without extra parsing.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        doc_type: str = inp.get("document_type", "other")
        extracted: dict[str, Any] = inp.get("extracted_data", {})

        prompt_fn = _RISK_PROMPTS.get(doc_type)
        if prompt_fn:
            prompt = prompt_fn(extracted)
        else:
            prompt = _generic_prompt(extracted, doc_type)

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            risk_analysis = _safe_json_parse(response.content[0].text)
        except Exception as exc:
            logger.warning("Risk analysis LLM call failed: %s", exc)
            risk_analysis = {"red_flags": [], "key_amounts": [], "important_dates": []}

        red_flags: list[dict[str, Any]] = risk_analysis.get("red_flags", [])

        step = f"Risk analysis complete — {len(red_flags)} red flags found"
        logger.info(step)
        return {
            "input": {**inp, "risk_analysis": risk_analysis},
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 4: create_alerts_and_reminders
    # ------------------------------------------------------------------

    async def _create_alerts_and_reminders(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """Persist high-severity red flags as alerts and dates as reminders."""
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        risk: dict[str, Any] = inp.get("risk_analysis", {})
        doc_type: str = inp.get("document_type", "document")
        user_id: str = state["user_id"]

        red_flags: list[dict[str, Any]] = risk.get("red_flags", [])
        important_dates: list[dict[str, Any]] = risk.get(
            "important_dates",
            inp.get("extracted_data", {}).get("important_dates", []),
        )

        alerts_created = 0
        reminders_created = 0

        # --- Alerts for red flags ---
        for flag in red_flags:
            severity = str(flag.get("severity", "medium")).lower()
            if severity not in ("high", "medium"):
                continue
            try:
                await self.call_tool(
                    "graph_mcp",
                    "create_alert_node",
                    {
                        "user_id": user_id,
                        "alert_type": "document_risk",
                        "title": (
                            f"{doc_type} red flag: "
                            f"{flag.get('title', 'Unknown')}"
                        ),
                        "message": flag.get("description", ""),
                        "severity": severity,
                    },
                )
                alerts_created += 1
            except RuntimeError as exc:
                logger.warning("Alert creation failed: %s", exc)

        # --- Calendar reminders for important dates ---
        for entry in important_dates:
            date_str = entry.get("date")
            if not date_str:
                continue
            try:
                await self.call_tool(
                    "calendar_mcp",
                    "add_financial_reminder",
                    {
                        "title": f"{doc_type}: {entry.get('event', 'deadline')}",
                        "date": date_str,
                        "description": (
                            f"Action required for your {doc_type}"
                        ),
                        "amount": entry.get("amount"),
                    },
                )
                reminders_created += 1
            except RuntimeError as exc:
                logger.warning("Reminder creation failed: %s", exc)

        step = (
            f"Created {alerts_created} alerts, "
            f"{reminders_created} calendar reminders"
        )
        logger.info(step)
        return {
            "input": {
                **inp,
                "alerts_created": alerts_created,
                "reminders_created": reminders_created,
            },
            **self.add_step(state, step),
        }

    # ------------------------------------------------------------------
    # NODE 5: generate_plain_english_summary
    # ------------------------------------------------------------------

    async def _generate_plain_english_summary(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """Ask Claude for a plain-English summary and finalise output.

        The prompt is structured for people who don't read contracts:
        what the document is, obligations with dollar amounts, red flags,
        and one concrete action for the next 7 days.
        """
        if state.get("status") == "failed":
            return {}

        inp: dict[str, Any] = state.get("input", {})
        doc_type: str = inp.get("document_type", "document")
        risk: dict[str, Any] = inp.get("risk_analysis", {})
        extracted: dict[str, Any] = inp.get("extracted_data", {})
        alerts_created: int = inp.get("alerts_created", 0)
        reminders_created: int = inp.get("reminders_created", 0)

        key_amounts: list[dict[str, Any]] = risk.get(
            "key_amounts", extracted.get("key_amounts", [])
        )
        important_dates: list[dict[str, Any]] = risk.get(
            "important_dates", extracted.get("important_dates", [])
        )
        red_flags: list[dict[str, Any]] = risk.get("red_flags", [])

        prompt = (
            f"I analyzed a {doc_type} document. Here's what I found:\n"
            f"{json.dumps(risk, default=str)}\n"
            f"Key amounts: {json.dumps(key_amounts, default=str)}\n"
            f"Important dates: {json.dumps(important_dates, default=str)}\n\n"
            "Write a plain-English summary for someone who doesn't read "
            "contracts.\n"
            "Structure:\n"
            "1. What this document is (1 sentence)\n"
            "2. The most important financial obligations "
            "(bullet points with dollar amounts)\n"
            "3. RED FLAGS — anything that could cost them money "
            "unexpectedly (if any)\n"
            "4. One specific action they should take in the next 7 days\n\n"
            "Be direct. Use plain language. If something is concerning, "
            "say so clearly."
        )

        try:
            response = self._llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            summary: str = response.content[0].text.strip()
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            flag_count = len(red_flags)
            summary = (
                f"Analyzed your {doc_type}. "
                f"Found {len(key_amounts)} key amounts and "
                f"{flag_count} potential issue{'s' if flag_count != 1 else ''}. "
                f"Check the detailed analysis for specifics."
            )

        output: dict[str, Any] = {
            "document_type": doc_type,
            "extracted_data": extracted,
            "risk_analysis": risk,
            "key_amounts": key_amounts,
            "important_dates": important_dates,
            "red_flags_count": len(red_flags),
            "alerts_created": alerts_created,
            "reminders_created": reminders_created,
            "summary": summary,
        }

        step = "Generated plain-English summary"
        logger.info(step)
        return {
            **self._complete(state, output),
            **self.add_step(state, step),
        }
