"""Vision MCP server — gives Foresight agents eyes.

This server processes photos of receipts, bills, and financial documents
using Claude's vision capabilities and returns structured JSON that
downstream agents can store in the knowledge graph, create alerts from,
or use for cashflow forecasting.

Image handling
--------------
* Images are resized so the longest side is at most 1024 px (saves API
  tokens without losing legibility for OCR).
* JPEG quality is capped at 85 after resize for further compression.
* Payloads larger than 5 MB (after encoding) are rejected before the API
  call is made.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import anthropic
from PIL import Image

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseMCPServer

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
_MAX_SIDE_PX = 1024
_JPEG_QUALITY = 85
_MODEL = "claude-3-haiku-20240307"


def _resize_and_compress(raw_b64: str, media_type: str) -> str:
    """Resize to ≤1024 px longest side and re-encode as JPEG.

    Returns the resulting image as a base64 string.
    """
    img_bytes = base64.b64decode(raw_b64)
    img = Image.open(io.BytesIO(img_bytes))

    w, h = img.size
    if max(w, h) > _MAX_SIDE_PX:
        scale = _MAX_SIDE_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def _validate_image(raw_b64: str) -> None:
    """Raise ``ValueError`` if the payload is too large or undecodable."""
    try:
        raw_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise ValueError("Image data is not valid base64") from exc

    if len(raw_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)} MB limit "
            f"({len(raw_bytes) / (1024 * 1024):.1f} MB received)"
        )

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()
    except Exception as exc:
        raise ValueError(
            "Image is unreadable or corrupted — please provide a clear, "
            "well-lit photo"
        ) from exc


def _parse_json_from_text(text: str) -> dict[str, Any]:
    """Extract the first JSON object from Claude's response text."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {"raw_text": text}
    return json.loads(text[start:end])


class VisionMCPServer(BaseMCPServer):
    """MCP server that processes receipt and document images.

    Tools registered:

    1. **scan_receipt** — extract merchant, items, total from a receipt photo
    2. **analyze_bill** — parse a utility/medical/insurance bill and flag
       anomalies
    3. **extract_document_info** — extract financial data from any document
    """

    def __init__(self, anthropic_api_key: str) -> None:
        super().__init__(name="vision")
        self._claude = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        logger.info("VisionMCPServer created")
        self.setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Register the three vision tools."""

        self.register_tool(
            name="scan_receipt",
            description=(
                "Extract structured data from a photo of a receipt — "
                "merchant, items, total, date, payment method"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_base64": {"type": "string"},
                    "image_type": {"type": "string", "default": "image/jpeg"},
                },
                "required": ["image_base64"],
            },
            handler=self._scan_receipt_handler,
        )

        self.register_tool(
            name="analyze_bill",
            description=(
                "Analyze a photo or PDF of a bill — utility, medical, "
                "insurance — and extract key information and flag anomalies"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_base64": {"type": "string"},
                    "bill_type": {
                        "type": "string",
                        "enum": ["utility", "medical", "insurance", "telecom", "other"],
                    },
                },
                "required": ["image_base64"],
            },
            handler=self._analyze_bill_handler,
        )

        self.register_tool(
            name="extract_document_info",
            description=(
                "Extract financial information from any document — lease, "
                "insurance policy, contract"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_base64": {"type": "string"},
                    "document_type": {"type": "string"},
                },
                "required": ["image_base64"],
            },
            handler=self._extract_document_handler,
        )

    # ------------------------------------------------------------------
    # Claude vision helper
    # ------------------------------------------------------------------

    async def _vision_call(
        self,
        image_b64: str,
        media_type: str,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Send an image + text prompt to Claude and return the text response."""
        response = await self._claude.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Tool 1: scan_receipt
    # ------------------------------------------------------------------

    async def _scan_receipt_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        """Extract structured data from a receipt photo."""
        raw_b64: str = params["image_base64"]
        media_type: str = params.get("image_type", "image/jpeg")

        _validate_image(raw_b64)
        compressed = _resize_and_compress(raw_b64, media_type)

        logger.info("scan_receipt: sending image to Claude vision")

        text = await self._vision_call(
            compressed,
            "image/jpeg",
            "Extract all information from this receipt. Return ONLY a JSON "
            "object with: merchant_name, date (YYYY-MM-DD), total_amount "
            "(number), subtotal, tax, tip, payment_method, items (list of "
            "{name, quantity, price}), currency. If any field is not visible, "
            "use null.",
        )

        data = _parse_json_from_text(text)
        data["raw_text"] = text
        logger.info("scan_receipt: merchant=%s total=%s", data.get("merchant_name"), data.get("total_amount"))
        return data

    # ------------------------------------------------------------------
    # Tool 2: analyze_bill
    # ------------------------------------------------------------------

    async def _analyze_bill_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        """Analyse a bill image and flag anomalies."""
        raw_b64: str = params["image_base64"]
        bill_type: str = params.get("bill_type", "other")

        _validate_image(raw_b64)
        compressed = _resize_and_compress(raw_b64, "image/jpeg")

        logger.info("analyze_bill: type=%s, sending to Claude vision", bill_type)

        extraction_prompt = (
            f"Analyze this {bill_type} bill image. Return ONLY a JSON object "
            "with: biller_name, account_number (last 4 digits only), "
            "billing_period_start (YYYY-MM-DD), billing_period_end, due_date, "
            "total_due (number), previous_balance, new_charges (list of "
            "{description, amount}), payment_methods_accepted (list), "
            "is_past_due (bool), late_fee_if_any. Use null for any field "
            "you cannot determine."
        )
        bill_text = await self._vision_call(compressed, "image/jpeg", extraction_prompt)
        bill_data = _parse_json_from_text(bill_text)

        anomaly_prompt = (
            f"Look at this {bill_type} bill image again. Are there any "
            "unusual charges, fees that seem higher than typical, or hidden "
            "costs? Return ONLY a JSON object with: anomalies (list of "
            "{description, amount, reason_flagged}), risk_level "
            "(none/low/medium/high)."
        )
        anomaly_text = await self._vision_call(compressed, "image/jpeg", anomaly_prompt)
        anomaly_data = _parse_json_from_text(anomaly_text)

        total = bill_data.get("total_due", "unknown")
        biller = bill_data.get("biller_name", "unknown")
        summary = f"{biller} bill for ${total}" if total != "unknown" else f"{biller} bill"

        logger.info("analyze_bill: biller=%s total=%s anomalies=%d", biller, total, len(anomaly_data.get("anomalies", [])))
        return {
            "bill_data": bill_data,
            "anomalies": anomaly_data.get("anomalies", []),
            "summary_text": summary,
        }

    # ------------------------------------------------------------------
    # Tool 3: extract_document_info
    # ------------------------------------------------------------------

    async def _extract_document_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        """Extract financial information from an arbitrary document image."""
        raw_b64: str = params["image_base64"]
        doc_type: str = params.get("document_type", "financial document")

        _validate_image(raw_b64)
        compressed = _resize_and_compress(raw_b64, "image/jpeg")

        logger.info("extract_document_info: type=%s, sending to Claude vision", doc_type)

        extraction_prompt = (
            f"Extract all financially relevant information from this "
            f"{doc_type}. Return ONLY a JSON object with: document_type, "
            "key_amounts (list of {description, amount, frequency}), "
            "important_dates (list of {event, date}), parties_involved "
            "(list of strings), auto_renewal (bool if found, else null), "
            "cancellation_policy (string summary or null), key_terms "
            "(list of short strings). Use null for anything not found."
        )
        extraction_text = await self._vision_call(
            compressed, "image/jpeg", extraction_prompt, max_tokens=1500,
        )
        extracted_data = _parse_json_from_text(extraction_text)

        risk_prompt = (
            "Based on the document you just analyzed, identify any financial "
            "risks, hidden fees, auto-renewal traps, or terms that could cost "
            "the user money. Return ONLY a JSON object with: risks (list of "
            "{risk, severity (low/medium/high), recommendation}), "
            "financial_summary (one-paragraph plain-English summary of the "
            "financial implications)."
        )
        risk_text = await self._vision_call(
            compressed, "image/jpeg", risk_prompt, max_tokens=1024,
        )
        risk_data = _parse_json_from_text(risk_text)

        logger.info(
            "extract_document_info: type=%s amounts=%d risks=%d",
            extracted_data.get("document_type", doc_type),
            len(extracted_data.get("key_amounts", [])),
            len(risk_data.get("risks", [])),
        )
        return {
            "extracted_data": extracted_data,
            "financial_summary": risk_data.get("financial_summary", ""),
            "risks_and_alerts": risk_data.get("risks", []),
        }
