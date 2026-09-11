"""Request and validate structured narration through the generic LLM service."""

import json
import re

from loguru import logger
from pydantic import ValidationError

from app.models.narration import MultiNarration, NarrationSegment, Narrator
from app.services import llm

_OUTPUT_INSTRUCTIONS = """# Structured narration output (overrides earlier output-format instructions):
Return ONLY a valid JSON object: {"segments":[{"speaker_id":"...","text":"..."}]}.
No Markdown fences, prose before/after JSON, or additional fields.
Use ONLY the exact supplied speaker IDs; never invent speakers.
Preserve narrative order and split segments whenever the active speaker changes.
Every segment must contain non-empty spoken text. Do not add speaker labels to
spoken text unless they are actually intended to be spoken.
Return no narrator definitions, voice configuration, subtitles, or metadata.
Apply the requested language, length and writing instructions to the spoken text.
"""
_CORRECTION = """
Your previous response was invalid structured output or failed speaker/text validation.
Return ONLY valid JSON matching the required schema and supplied speaker IDs.
"""


_DIAGNOSTICS = {
    "EMPTY_RESPONSE": "empty response",
    "JSON_DECODE_ERROR": "malformed JSON",
    "INVALID_ROOT": "expected a JSON object",
    "INVALID_SCHEMA": "incorrect JSON schema",
    "MISSING_SEGMENTS": "missing segments",
    "INVALID_SEGMENT": "invalid segment fields",
    "UNKNOWN_SPEAKER": "unknown speaker ID",
    "EMPTY_TEXT": "empty segment text",
    "DOMAIN_VALIDATION_ERROR": "invalid narration data",
}


class _StructuredError(ValueError):
    def __init__(self, category):
        self.category = category
        super().__init__(_DIAGNOSTICS[category])


def _parse(response: str, narrators: list[Narrator]) -> MultiNarration:
    if not isinstance(response, str):
        raise _StructuredError("INVALID_ROOT")
    response = response.strip()
    if not response:
        raise _StructuredError("EMPTY_RESPONSE")
    fence = re.fullmatch(r"```json\s*\n(.*?)\n```", response, flags=re.DOTALL)
    if fence:
        response = fence.group(1).strip()
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        raise _StructuredError("JSON_DECODE_ERROR") from None
    if not isinstance(payload, dict):
        raise _StructuredError("INVALID_ROOT")
    if "segments" not in payload:
        raise _StructuredError("MISSING_SEGMENTS")
    if set(payload) != {"segments"}:
        raise _StructuredError("INVALID_SCHEMA")
    segments = payload["segments"]
    if not isinstance(segments, list):
        raise _StructuredError("INVALID_SCHEMA")
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"speaker_id", "text"}:
            raise _StructuredError("INVALID_SEGMENT")
    try:
        validated = [NarrationSegment.model_validate(segment) for segment in segments]
        if any(
            segment.speaker_id not in {n.id for n in narrators} for segment in validated
        ):
            raise _StructuredError("UNKNOWN_SPEAKER")
        return MultiNarration(narrators=narrators, segments=validated)
    except ValidationError as exc:
        category = "DOMAIN_VALIDATION_ERROR"
        if any(
            error["loc"] == ("text",) and error["type"] == "string_too_short"
            for error in exc.errors()
        ):
            category = "EMPTY_TEXT"
        raise _StructuredError(category) from None


def generate(
    video_subject: str,
    narrators: list[Narrator],
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> MultiNarration:
    """Generate ordered segments with at most one malformed-output retry.

    Narrator definitions are application-owned; only IDs/names enter the prompt.
    Existing prompt controls and provider configuration retain their semantics.
    """
    narrators = [Narrator.model_validate(item.model_dump()) for item in narrators]
    if not narrators:
        raise ValueError("at least one narrator is required")
    if len({item.id for item in narrators}) != len(narrators):
        raise ValueError("narrator IDs must be unique")
    prompt = llm.build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    prompt += "\n\n" + _OUTPUT_INSTRUCTIONS
    prompt += "\nSupplied speakers: " + json.dumps(
        [{"id": item.id, "display_name": item.display_name} for item in narrators],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    for attempt in range(2):
        try:
            response = llm.generate_text(
                prompt=prompt if attempt == 0 else prompt + "\n" + _CORRECTION,
                app_config=app_config,
            )
        except Exception:
            logger.warning(
                "Structured narration attempt {} failed: PROVIDER_ERROR", attempt + 1
            )
            raise
        if isinstance(response, str) and response.lstrip().startswith("Error:"):
            logger.warning(
                "Structured narration attempt {} failed: PROVIDER_ERROR", attempt + 1
            )
            # Do not echo provider diagnostics, credentials, or raw responses.
            raise RuntimeError(
                "multi-narrator structured script generation failed: LLM provider error"
            )
        try:
            return _parse(response, narrators)
        except _StructuredError as exc:
            logger.warning(
                "Structured narration attempt {} failed: {}", attempt + 1, exc.category
            )
            if attempt == 1:
                raise ValueError(
                    "multi-narrator structured script generation failed after 2 attempts. "
                    f"Last validation error: {exc}."
                ) from None
