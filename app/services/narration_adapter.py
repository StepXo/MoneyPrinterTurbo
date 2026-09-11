"""Prepare ordinary WAV/SRT artifacts from an ordered multi-speaker script."""

import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from app.config import config
from app.models.narration import MultiNarration
from app.models.schema import VideoParams
from app.services import subtitle, voice
from app.utils import utils

_SAMPLE_RATE = 24000
_TIMESTAMP = re.compile(r"(\d+):(\d{2}):(\d{2}),(\d{3})")


@dataclass(frozen=True)
class NarrationArtifacts:
    audio_file: str
    audio_duration: float
    subtitle_path: str


def _milliseconds(value: str) -> int:
    match = _TIMESTAMP.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid subtitle timestamp: {value}")
    hours, minutes, seconds, millis = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid subtitle timestamp: {value}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _timestamp(value: int) -> str:
    seconds, millis = divmod(value, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _read_cues(source: Path, start_sample: int, end_sample: int):
    """Use SRT as the boundary between incompatible provider timing objects."""
    if not source.is_file():
        raise RuntimeError(f"subtitle generation produced no file: {source.name}")
    blocks = re.split(r"\n\s*\n", source.read_text(encoding="utf-8-sig").strip())
    offset = start_sample * 1000 // _SAMPLE_RATE
    limit = end_sample * 1000 // _SAMPLE_RATE
    previous_end = offset
    cues = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].isdigit():
            raise ValueError(f"invalid subtitle block in {source.name}")
        times = lines[1].split(" --> ")
        if len(times) != 2:
            raise ValueError(f"invalid subtitle timing in {source.name}")
        start, end = (_milliseconds(value) + offset for value in times)
        # Clamp millisecond/provider tail overshoot to actual PCM boundaries.
        start = max(previous_end, min(start, limit))
        end = min(end, limit)
        if end <= start:
            raise ValueError(f"subtitle cue has no usable duration in {source.name}")
        text = "\n".join(lines[2:]).strip()
        if not text:
            raise ValueError(f"empty subtitle text in {source.name}")
        cues.append((start, end, text))
        previous_end = end
    return cues


def _append_pcm(source: Path, normalized: Path, master) -> int:
    if not source.is_file() or not source.stat().st_size:
        raise RuntimeError(f"TTS produced no audio: {source.name}")
    # Same FFmpeg PCM format as voice's pause-aware synthesis; decode once.
    subprocess.run(
        [
            utils.get_ffmpeg_binary(),
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_SAMPLE_RATE),
            "-codec:a",
            "pcm_s16le",
            str(normalized),
        ],
        capture_output=True,
        check=True,
    )
    with wave.open(str(normalized), "rb") as chunk:
        if (chunk.getframerate(), chunk.getnchannels(), chunk.getsampwidth()) != (
            _SAMPLE_RATE,
            1,
            2,
        ):
            raise RuntimeError("normalization did not produce the expected PCM format")
        frames = chunk.getnframes()
        if frames <= 0:
            raise RuntimeError("normalized segment contains no samples")
        remaining = frames
        while remaining:
            data = chunk.readframes(min(remaining, _SAMPLE_RATE))
            if not data or len(data) % 2:
                raise RuntimeError("truncated normalized audio")
            master.writeframesraw(data)
            remaining -= len(data) // 2
    return frames


def _apply_preview_volume(audio_path: Path, volume: float):
    """Apply linear gain once to the preview master, leaving timing unchanged."""
    target = audio_path.with_name("preview.wav")
    subprocess.run(
        [
            utils.get_ffmpeg_binary(),
            "-y",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-af",
            f"volume={float(volume)}",
            "-codec:a",
            "pcm_s16le",
            str(target),
        ],
        capture_output=True,
        check=True,
    )
    with (
        wave.open(str(audio_path), "rb") as source,
        wave.open(str(target), "rb") as result,
    ):
        if source.getparams() != result.getparams():
            raise RuntimeError("preview volume processing changed audio timing/format")
    target.replace(audio_path)


def prepare(
    task_id: str, params: VideoParams, *, apply_volume: bool = False
) -> NarrationArtifacts:
    """Return a WAV master and optional SRT; any failure removes this run only.

    Uses the current subtitle provider and existing TTS pause behavior. Callers
    must keep runtime provider configuration stable, as the existing worker does.
    Only direct audio previews opt into volume; video rendering applies its own.
    """
    if params.narration is None:
        raise ValueError("multi narration is required")
    # Revalidate a snapshot, including models mutated after initial validation.
    narration = MultiNarration.model_validate(params.narration.model_dump())
    if params.custom_audio_file:
        raise ValueError("multi narration cannot use uploaded audio")
    voices = {
        item.id: voice.parse_voice_name(item.voice_name) for item in narration.narrators
    }
    if any(not name or voice.is_no_voice(name) for name in voices.values()):
        raise ValueError("each narrator must have a TTS voice")
    provider = (
        config.app.get("subtitle_provider", "edge").strip().lower()
        if params.subtitle_enabled
        else ""
    )
    if provider not in ("", "edge", "whisper"):
        raise ValueError(f"unsupported subtitle provider: {provider}")
    word_level = params.subtitle_display_mode == "word_by_word"
    output = Path(tempfile.mkdtemp(prefix="narration-", dir=utils.task_dir(task_id)))
    try:
        audio_path = output / "audio.wav"
        subtitle_path = output / "subtitle.srt"
        total_samples = 0
        cues = []
        with tempfile.TemporaryDirectory(prefix="segments-", dir=output) as temporary:
            with wave.open(str(audio_path), "wb") as master:
                master.setparams((1, 2, _SAMPLE_RATE, 0, "NONE", "not compressed"))
                for index, segment in enumerate(narration.segments):
                    try:
                        source = Path(temporary) / f"{index}.mp3"
                        timing = voice.tts(
                            text=segment.text,
                            voice_name=voices[segment.speaker_id],
                            voice_rate=params.voice_rate,
                            voice_file=str(source),
                        )
                        if timing is None:
                            raise RuntimeError("TTS failed")
                        frames = _append_pcm(
                            source, Path(temporary) / f"{index}.wav", master
                        )
                        if (
                            provider == "edge"
                            and utils.remove_pause_tags(segment.text).strip()
                        ):
                            segment_srt = Path(temporary) / f"{index}.srt"
                            voice.create_subtitle(
                                sub_maker=timing,
                                text=segment.text,
                                subtitle_file=str(segment_srt),
                                word_level=word_level,
                            )
                            cues.extend(
                                _read_cues(
                                    segment_srt, total_samples, total_samples + frames
                                )
                            )
                        total_samples += frames
                    except Exception as exc:
                        exc.add_note(
                            f"narration segment {index + 1}, speaker {segment.speaker_id}"
                        )
                        raise
            if provider == "whisper":
                subtitle.create(
                    audio_file=str(audio_path),
                    subtitle_file=str(subtitle_path),
                    word_level=word_level,
                )
                if not word_level:
                    subtitle.correct(
                        subtitle_file=str(subtitle_path),
                        video_script=utils.remove_pause_tags(
                            narration.flatten_script()
                        ),
                    )
                cues = _read_cues(subtitle_path, 0, total_samples)
            if cues:
                subtitle_path.write_text(
                    "".join(
                        f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}\n\n"
                        for index, (start, end, text) in enumerate(cues, 1)
                    ),
                    encoding="utf-8",
                )
        if apply_volume:
            _apply_preview_volume(audio_path, params.voice_volume)
        return NarrationArtifacts(
            str(audio_path),
            total_samples / _SAMPLE_RATE,
            str(subtitle_path) if cues else "",
        )
    except BaseException:
        # This uniquely allocated directory contains only this invocation's work.
        shutil.rmtree(output, ignore_errors=True)
        raise
