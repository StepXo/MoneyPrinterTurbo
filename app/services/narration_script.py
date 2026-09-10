"""Request and validate structured narration through the generic LLM service."""

import json
import re

from app.models.narration import MultiNarration, Narrator
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


def _parse(response: str, narrators: list[Narrator]) -> MultiNarration:
    response = response.strip()
    fence = re.fullmatch(r"```json\s*\n(.*?)\n```", response, flags=re.DOTALL)
    if fence:
        response = fence.group(1).strip()
    payload = json.loads(response)
    if not isinstance(payload, dict) or set(payload) != {"segments"}:
        raise ValueError("expected an object containing only segments")
    segments = payload["segments"]
    if not isinstance(segments, list):
        raise ValueError("segments must be a list")
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"speaker_id", "text"}:
            raise ValueError("each segment must contain only speaker_id and text")
    return MultiNarration(narrators=narrators, segments=segments)


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
        response = llm.generate_text(
            prompt=prompt if attempt == 0 else prompt + "\n" + _CORRECTION,
            app_config=app_config,
        )
        if isinstance(response, str) and response.lstrip().startswith("Error:"):
            # Do not echo provider diagnostics, credentials, or raw responses.
            raise RuntimeError(
                "multi-narrator structured script generation failed: LLM provider error"
            )
        try:
            if not isinstance(response, str):
                raise ValueError("expected text response")
            return _parse(response, narrators)
        except ValueError:
            if attempt == 1:
                raise ValueError(
                    "multi-narrator structured script generation failed: "
                    "invalid JSON structure or speaker/text validation after 2 attempts"
                ) from None
