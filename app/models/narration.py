"""Provider-independent data for ordered multi-narrator scripts."""

from typing import Annotated, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator


_NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Narrator(BaseModel):
    """An explicitly assigned ID stays independent of the editable display name."""

    id: _NonEmptyText
    display_name: _NonEmptyText
    voice_name: _NonEmptyText


class NarrationSegment(BaseModel):
    speaker_id: _NonEmptyText
    text: _NonEmptyText


class MultiNarration(BaseModel):
    narrators: list[Narrator] = Field(min_length=1)
    segments: list[NarrationSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_speakers(self) -> Self:
        narrator_ids = {narrator.id for narrator in self.narrators}
        if len(narrator_ids) != len(self.narrators):
            raise ValueError("narrator IDs must be unique")
        for segment in self.segments:
            if segment.speaker_id not in narrator_ids:
                raise ValueError(f"unknown speaker ID: {segment.speaker_id}")
        return self

    def flatten_script(self) -> str:
        """Join spoken text in order; pause sanitization belongs at integration."""
        return " ".join(segment.text for segment in self.segments)
