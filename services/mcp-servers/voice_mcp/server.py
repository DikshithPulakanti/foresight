"""Voice MCP server — ears and a voice for Foresight agents.

This server handles the audio layer of the Foresight assistant:

* **Speech-to-text** via OpenAI Whisper (``base`` model for speed).
* **Text-to-speech** via gTTS (Google Text-to-Speech) as a lightweight
  fallback; can be swapped for Kokoro TTS when available.
* **Full voice pipeline** that transcribes, classifies the financial
  intent, and returns a structured result ready for downstream agents.

Privacy
-------
Transcription *length* is logged but the actual transcript text is **never**
written to logs.

Model loading
-------------
The Whisper model is loaded lazily on the first ``transcribe_audio`` call
to avoid slowing down server startup.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from typing import Any

import whisper
from gtts import gTTS

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseMCPServer

logger = logging.getLogger(__name__)

_MAX_DURATION_S = 120
_WHISPER_MODEL = "base"
_WORDS_PER_MINUTE = 150

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "spending_query": ["spent", "spend", "cost", "paid", "charges"],
    "balance_query": ["balance", "account", "how much", "left"],
    "subscription_query": ["subscriptions", "recurring", "cancel"],
    "alert_query": ["alerts", "warnings", "upcoming"],
    "goal_query": ["goal", "saving", "savings"],
}


class VoiceMCPServer(BaseMCPServer):
    """MCP server for audio processing — STT, TTS, and intent detection.

    Tools registered:

    1. **transcribe_audio** — Whisper speech-to-text
    2. **synthesize_speech** — gTTS text-to-speech
    3. **process_voice_query** — full pipeline: transcribe → classify intent
    """

    def __init__(self) -> None:
        super().__init__(name="voice")
        self._whisper_model: whisper.Whisper | None = None
        logger.info("VoiceMCPServer created (Whisper model will load on first use)")
        self.setup()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_whisper(self) -> whisper.Whisper:
        """Return the Whisper model, loading it on first access."""
        if self._whisper_model is None:
            logger.info("Loading Whisper '%s' model …", _WHISPER_MODEL)
            self._whisper_model = whisper.load_model(_WHISPER_MODEL)
            logger.info("Whisper model loaded")
        return self._whisper_model

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Register the three voice tools."""

        self.register_tool(
            name="transcribe_audio",
            description=(
                "Convert spoken audio to text using Whisper — used when "
                "user speaks to Foresight"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "audio_base64": {"type": "string"},
                    "audio_format": {"type": "string", "default": "wav"},
                    "language": {"type": "string", "default": "en"},
                },
                "required": ["audio_base64"],
            },
            handler=self._transcribe_handler,
        )

        self.register_tool(
            name="synthesize_speech",
            description="Convert text to speech for Foresight's voice responses",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "voice_speed": {"type": "number", "default": 1.0},
                    "format": {"type": "string", "default": "mp3"},
                },
                "required": ["text"],
            },
            handler=self._synthesize_handler,
        )

        self.register_tool(
            name="process_voice_query",
            description=(
                "Full pipeline: transcribe audio, understand intent, return "
                "text response ready to synthesize"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "audio_base64": {"type": "string"},
                    "audio_format": {"type": "string", "default": "wav"},
                },
                "required": ["audio_base64"],
            },
            handler=self._process_voice_query_handler,
        )

    # ------------------------------------------------------------------
    # Audio validation
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_audio(raw_b64: str, fmt: str) -> str:
        """Decode base64 audio to a temp file and return its path.

        Raises ``ValueError`` if the payload is empty or too large for the
        configured maximum duration heuristic.
        """
        try:
            audio_bytes = base64.b64decode(raw_b64)
        except Exception as exc:
            raise ValueError("Audio data is not valid base64") from exc

        if len(audio_bytes) == 0:
            raise ValueError("Audio payload is empty")

        # Rough duration estimate: ~16 kB/s for 16-bit 16 kHz mono PCM
        estimated_seconds = len(audio_bytes) / 16_000
        if estimated_seconds > _MAX_DURATION_S:
            raise ValueError(
                f"Audio appears to exceed {_MAX_DURATION_S}s "
                f"(~{estimated_seconds:.0f}s estimated). "
                "Please send a shorter clip."
            )

        suffix = f".{fmt}" if not fmt.startswith(".") else fmt
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name

    # ------------------------------------------------------------------
    # Tool 1: transcribe_audio
    # ------------------------------------------------------------------

    async def _transcribe_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transcribe audio using Whisper."""
        raw_b64: str = params["audio_base64"]
        fmt: str = params.get("audio_format", "wav")
        language: str = params.get("language", "en")

        audio_path = self._decode_audio(raw_b64, fmt)

        try:
            model = self._get_whisper()
            result = model.transcribe(audio_path, language=language, fp16=False)
        finally:
            os.unlink(audio_path)

        transcript: str = result.get("text", "").strip()
        segments = result.get("segments", [])
        duration = segments[-1]["end"] if segments else 0.0

        avg_logprob = 0.0
        if segments:
            avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
        confidence = round(min(1.0, max(0.0, 1.0 + avg_logprob)), 2)

        logger.info(
            "transcribe_audio: length=%d chars, duration=%.1fs, confidence=%.2f",
            len(transcript),
            duration,
            confidence,
        )

        return {
            "transcript": transcript,
            "confidence": confidence,
            "language_detected": result.get("language", language),
            "duration_seconds": round(duration, 2),
        }

    # ------------------------------------------------------------------
    # Tool 2: synthesize_speech
    # ------------------------------------------------------------------

    async def _synthesize_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        """Synthesize speech from text using gTTS."""
        text: str = params["text"]
        voice_speed: float = params.get("voice_speed", 1.0)
        out_format: str = params.get("format", "mp3")

        slow = voice_speed < 0.8

        tts = gTTS(text=text, lang="en", slow=slow)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()

        word_count = len(text.split())
        duration_estimate = round(word_count / _WORDS_PER_MINUTE * 60, 2)

        logger.info(
            "synthesize_speech: %d words, ~%.1fs estimated",
            word_count,
            duration_estimate,
        )

        return {
            "audio_base64": audio_b64,
            "format": out_format,
            "duration_estimate_seconds": duration_estimate,
        }

    # ------------------------------------------------------------------
    # Tool 3: process_voice_query
    # ------------------------------------------------------------------

    async def _process_voice_query_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transcribe audio and classify the financial intent."""
        transcription = await self._transcribe_handler(params)
        transcript: str = transcription["transcript"]

        intent, score = self._classify_intent(transcript)

        logger.info(
            "process_voice_query: intent=%s, confidence=%.2f, transcript_len=%d",
            intent,
            score,
            len(transcript),
        )

        return {
            "transcript": transcript,
            "intent": intent,
            "confidence": score,
            "ready_for_agent": intent != "unknown",
        }

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_intent(transcript: str) -> tuple[str, float]:
        """Match transcript against keyword lists and return (intent, score).

        The score is based on the fraction of keywords that matched, scaled
        to a 0.5–1.0 range (a single keyword match yields 0.5, all keywords
        yield 1.0).
        """
        lower = transcript.lower()
        best_intent = "unknown"
        best_score = 0.0

        for intent, keywords in _INTENT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in lower)
            if hits == 0:
                continue
            score = 0.5 + 0.5 * (hits / len(keywords))
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent, round(best_score, 2)
