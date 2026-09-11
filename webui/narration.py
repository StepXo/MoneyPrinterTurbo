"""Optional Streamlit narration controls; backend services own generation."""

import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from app.config import config
from app.models.narration import MultiNarration, Narrator
from app.services import llm, narration_adapter, narration_script, voice
from app.utils import utils

_DATA = "narration_data"
_ENABLED = "narration_enabled"


def enabled():
    return bool(st.session_state.get(_ENABLED, False))


def _data():
    if _DATA not in st.session_state:
        st.session_state[_DATA] = {
            "narrators": [],
            "segments": [],
            "next_narrator": 1,
            "next_segment": 1,
            "voices": {},
        }
    return st.session_state[_DATA]


def add_narrator():
    data = _data()
    number = data["next_narrator"]
    existing = {item["id"] for item in data["narrators"]}
    while f"narrator_{number}" in existing:
        number += 1
    choices = list(data["voices"])
    data["narrators"].append(
        {
            "id": f"narrator_{number}",
            "display_name": f"Narrator {number}",
            "voice_name": choices[min(len(data["narrators"]), len(choices) - 1)]
            if choices
            else "",
        }
    )
    data["next_narrator"] = number + 1


def _toggle():
    data = _data()
    if enabled():
        data["single_script"] = st.session_state.get("video_script", "")
        data["single_terms"] = st.session_state.get("video_terms", "")
        if not data["narrators"]:
            add_narrator()
            add_narrator()
    else:
        st.session_state["video_script"] = data.get("single_script", "")
        st.session_state["video_terms"] = data.get("single_terms", "")


def render_mode_control(tr):
    # Button events are available before Main reaches their synchronous actions.
    # A temporary widget avoids re-registering the same widget twice in one run.
    busy = any(
        st.session_state.get(key, False)
        for key in ("auto_generate_script", "narration_generate", "auto_generate_terms")
    )
    value = enabled()
    st.session_state[_ENABLED] = value  # retain identity while its widget is absent
    slot = st.empty()

    def unlock():
        slot.empty()
        slot.toggle(
            tr("Use multiple narrators"), key=_ENABLED, value=value, on_change=_toggle
        )

    if busy:
        slot.toggle(
            tr("Use multiple narrators"),
            key="narration_locked_mode",
            value=value,
            disabled=True,
        )
    else:
        unlock()
    if enabled() and not _data()["narrators"]:
        _toggle()
    return unlock if busy else None


def removal_reason(narrator_id):
    data = _data()
    if len(data["narrators"]) <= 2:
        return "Keep at least two narrators."
    if any(item["speaker_id"] == narrator_id for item in data["segments"]):
        return "Reassign this narrator's segments before removing them."
    return ""


def remove_narrator(narrator_id):
    reason = removal_reason(narrator_id)
    if reason:
        raise ValueError(reason)
    data = _data()
    data["narrators"] = [
        item for item in data["narrators"] if item["id"] != narrator_id
    ]
    data.get("sample_previews", {}).pop(narrator_id, None)


def add_segment(revision=None):
    data = _data()
    if revision is not None and revision != data.get("editor_revision", 0):
        return
    data["segments"].append(
        {
            "key": data["next_segment"],
            "speaker_id": data["narrators"][0]["id"],
            "text": "",
        }
    )
    data["next_segment"] += 1
    _refresh_drafts()


def remove_segment(key):
    data = _data()
    if not any(item["key"] == key for item in data["segments"]):
        return
    data["segments"] = [item for item in data["segments"] if item["key"] != key]
    _refresh_drafts()


def _edit(collection, identity_field, identity, field, widget_key):
    for item in _data()[collection]:
        if item[identity_field] == identity:
            item[field] = st.session_state[widget_key]
            if collection == "segments" or field == "display_name":
                _refresh_drafts()
            return


def render_narrators(voice_options, tr, preview_slots=None):
    """Consume Main's existing voice options, retaining earlier assignments."""
    data = _data()
    data["voices"].update(
        {
            key: label
            for key, label in voice_options.items()
            if key and not voice.is_no_voice(key)
        }
    )
    options = list(data["voices"])
    st.caption(
        tr(
            "Choose a voiceover service above to load its voices. Assigned voices are preserved."
        )
    )
    for index, item in enumerate(data["narrators"]):
        if not item["voice_name"] and options:
            item["voice_name"] = options[min(index, len(options) - 1)]
        columns = st.columns(2)
        with columns[0]:
            name_key = f"narration_name_{item['id']}"
            st.text_input(
                tr("Name"),
                value=item["display_name"],
                key=name_key,
                on_change=_edit,
                args=("narrators", "id", item["id"], "display_name", name_key),
            )
        with columns[1]:
            voice_key = f"narration_voice_{item['id']}"
            choices = list(
                dict.fromkeys(
                    options + ([item["voice_name"]] if item["voice_name"] else [])
                )
            )
            st.selectbox(
                tr("Voice"),
                choices,
                index=choices.index(item["voice_name"])
                if item["voice_name"] in choices
                else None,
                format_func=lambda value: data["voices"].get(value, value),
                key=voice_key,
                on_change=_edit,
                args=("narrators", "id", item["id"], "voice_name", voice_key),
            )
        actions = st.columns(2)
        with actions[0]:
            if st.button(
                tr("Play Voice"),
                key=f"narration_sample_{item['id']}",
                icon=":material/graphic_eq:",
                use_container_width=True,
                disabled=not item["voice_name"],
            ):
                data["sample_requested"] = item["id"]
        with actions[1]:
            reason = removal_reason(item["id"])
            st.button(
                tr("Remove narrator"),
                icon=":material/delete:",
                type="primary",
                use_container_width=True,
                key=f"narration_remove_{item['id']}",
                disabled=bool(reason),
                help=tr(reason) if reason else None,
                on_click=remove_narrator,
                args=(item["id"],),
            )
        if preview_slots is not None:
            preview_slots[item["id"]] = st.empty()
    st.button(tr("Add narrator"), key="narration_add_narrator", on_click=add_narrator)
    return data["narrators"][0]["voice_name"] or ""


def _narrators():
    data = _data()
    if len(data["narrators"]) < 2:
        raise ValueError("Select at least two narrators.")
    narrators = [Narrator.model_validate(item) for item in data["narrators"]]
    if len({item.id for item in narrators}) != len(narrators):
        raise ValueError("Narrator IDs must be unique.")
    if len({item.display_name for item in narrators}) != len(narrators):
        raise ValueError("Narrator names must be unique.")
    if any(
        voice.is_no_voice(item.voice_name) or item.voice_name not in data["voices"]
        for item in narrators
    ):
        raise ValueError("Select an available voice for each narrator.")
    return narrators


def _build():
    return MultiNarration(
        narrators=_narrators(),
        segments=[
            {"speaker_id": item["speaker_id"], "text": item["text"]}
            for item in _data()["segments"]
        ],
    )


def _require_used_narrators(narration):
    """Validate completed narration at consumption/application boundaries only."""
    if len({segment.speaker_id for segment in narration.segments}) < 2:
        raise ValueError(
            "Multi-narrator mode requires at least two narrators to be used in the narration."
        )
    return narration


def _set_segments(segments):
    data = _data()
    data["segments"] = []
    for segment in segments:
        data["segments"].append({"key": data["next_segment"], **segment.model_dump()})
        data["next_segment"] += 1


def _representations(narrators, segments):
    """Serialize editor representations without parsing/validating drafts."""
    names = {item["id"]: item["display_name"] for item in narrators}
    blocks = [
        {"speaker": names.get(item["speaker_id"], ""), "text": item["text"]}
        for item in segments
    ]
    return {
        "Tagged text": "\n\n".join(
            f"[{item['speaker']}]:\n{item['text']}" for item in blocks
        ),
        "JSON": json.dumps({"segments": blocks}, ensure_ascii=False, indent=2),
    }


def _replace_narration(candidate):
    """Install a validated candidate; obsolete widget generations cannot write back."""
    data = _data()
    candidate = MultiNarration(
        narrators=_narrators(),
        segments=[item.model_dump() for item in candidate.segments],
    )
    representations = _representations(
        [item.model_dump() for item in candidate.narrators],
        [item.model_dump() for item in candidate.segments],
    )
    _set_segments(candidate.segments)
    data["editor_revision"] = data.get("editor_revision", 0) + 1
    data["drafts"] = representations.copy()
    data["exports"] = representations.copy()
    data["draft_errors"] = {}
    data.pop("full_preview", None)
    data.pop("validated_narration", None)


def _refresh_drafts():
    """Explicit Visual Editor changes refresh derived inputs, retaining rejected drafts."""
    data = _data()
    representations = _representations(data["narrators"], data["segments"])
    drafts = data.setdefault("drafts", {})
    for mode, text in representations.items():
        if mode not in data.get("draft_errors", {}):
            drafts[mode] = text
    data["exports"] = representations
    data["editor_revision"] = data.get("editor_revision", 0) + 1
    data.pop("full_preview", None)


def _display_narration():
    """Cache validation of actual narration edits, independent of draft widgets."""
    data = _data()
    source = (data["narrators"], data["segments"])
    cached = data.get("validated_narration")
    if cached is None or cached[0] != source:
        try:
            narration = _build()
        except ValueError:
            narration = None
        data["validated_narration"] = (copy.deepcopy(source), narration)
    return data["validated_narration"][1]


def generate(params, run_operation):
    narrators = _narrators()
    if not params.video_subject.strip():
        raise ValueError("Please Enter the Video Subject First")

    def operation(app_config):
        result = narration_script.generate(
            video_subject=params.video_subject,
            narrators=narrators,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=(params.video_script_prompt or "")
            + "\nUse at least two distinct supplied speakers according to narrative context; do not mechanically alternate or force equal speaking time.",
            custom_system_prompt=params.custom_system_prompt,
            app_config=app_config,
        )
        _require_used_narrators(result)
        terms = llm.generate_terms(
            params.video_subject,
            utils.remove_pause_tags(result.flatten_script()),
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
            app_config=app_config,
        )
        if not terms:
            raise ValueError(
                "Keyword generation failed. Previous narration was preserved."
            )
        return result, terms

    result, terms = run_operation("generate_narration_and_terms", operation)
    keywords = ", ".join(terms)
    _replace_narration(result)
    st.session_state["video_terms"] = keywords
    return result


def render_script_controls(params, tr, run_operation, backend):
    if backend == "loomloom":
        st.info(
            tr(
                "LoomLoom conversion is unavailable in multiple-narrator mode. Switch to Local LLM Script Generation, or enter segments manually."
            )
        )
    elif st.button(
        tr("Generate Video Script and Keywords"),
        key="narration_generate",
        use_container_width=True,
        type="secondary",
    ):
        try:
            with st.spinner(tr("Generating Video Script and Keywords")):
                generate(params, run_operation)
        except (ValueError, RuntimeError) as exc:
            _show_error(exc, tr)
    mode = st.radio(
        "Narration editor",
        ["Tagged text", "Visual editor", "JSON"],
        key="narration_editor",
        horizontal=True,
    )
    if mode == "Visual editor":
        with st.container(key="advanced_settings_narration"):
            with st.expander(tr("Visual editor"), expanded=True):
                _render_visual_editor(tr)
    # Keep native form widgets mounted, including inactive drafts, so unrelated
    # reruns/editor switches cannot discard the browser's unsubmitted form input.
    for text_mode in ("Tagged text", "JSON"):
        _render_text_editor(text_mode, tr, active=mode == text_mode)
    params.narration = _display_narration()
    params.video_script = params.narration.flatten_script() if params.narration else ""


def _render_visual_editor(tr):
    names = {item["id"]: item["display_name"] for item in _data()["narrators"]}
    for index, item in enumerate(_data()["segments"], 1):
        with st.container(border=True):
            st.caption(f"{tr('Segment')} {index}")
            speaker_key = f"narration_speaker_{item['key']}"
            ids = list(names)
            st.selectbox(
                tr("Speaker"),
                ids,
                index=ids.index(item["speaker_id"])
                if item["speaker_id"] in ids
                else None,
                format_func=lambda value: names[value],
                key=speaker_key,
                on_change=_edit,
                args=("segments", "key", item["key"], "speaker_id", speaker_key),
            )
            text_key = f"narration_text_{item['key']}"
            st.text_area(
                tr("Text"),
                value=item["text"],
                key=text_key,
                on_change=_edit,
                args=("segments", "key", item["key"], "text", text_key),
            )
            st.button(
                tr("Remove segment"),
                key=f"narration_remove_segment_{item['key']}",
                on_click=remove_segment,
                args=(item["key"],),
            )
    st.button(
        tr("Add segment"),
        key=f"narration_add_segment_{_data().get('editor_revision', 0)}",
        on_click=add_segment,
        args=(_data().get("editor_revision", 0),),
    )


def parse_text(text, mode):
    """Import only documented display-name formats; never infer speakers."""
    narrators = _narrators()
    names = {item.display_name: item.id for item in narrators}
    if mode == "JSON":
        payload = json.loads(text.strip())
        if (
            not isinstance(payload, dict)
            or set(payload) != {"segments"}
            or not isinstance(payload["segments"], list)
        ):
            raise ValueError('JSON must contain only a "segments" array.')
        blocks = payload["segments"]
    else:
        blocks = []
        current = None
        for line in text.splitlines():
            header = re.fullmatch(r"\[([^\[\]\r\n]+)\]:\s*(.*)", line.strip())
            if header:
                current = {
                    "speaker": header[1].strip(),
                    "text": header[2] + "\n" if header[2] else "",
                }
                blocks.append(current)
            elif line.strip().startswith("[") and not re.fullmatch(
                r"\[pause:\s*[^\]]+\]", line.strip()
            ):
                raise ValueError(
                    "Malformed speaker header. Use [Display Name]: followed by text."
                )
            elif current is None:
                if line.strip():
                    raise ValueError("Text must follow a speaker header.")
            else:
                current["text"] += line + "\n"
    segments = []
    for block in blocks:
        if (
            not isinstance(block, dict)
            or set(block) != {"speaker", "text"}
            or not isinstance(block["speaker"], str)
        ):
            raise ValueError("Each segment requires speaker and text.")
        speaker = block["speaker"].strip()
        if speaker not in names:
            raise ValueError(f'Unknown narrator: "{block["speaker"]}".')
        segments.append({"speaker_id": names[speaker], "text": block["text"]})
    return MultiNarration(narrators=narrators, segments=segments)


def export_text(mode):
    narration = _build()
    names = {item.id: item.display_name for item in narration.narrators}
    blocks = [
        {"speaker": names[item.speaker_id], "text": item.text}
        for item in narration.segments
    ]
    if mode == "JSON":
        return json.dumps({"segments": blocks}, ensure_ascii=False, indent=2)
    result = "\n\n".join(f"[{item['speaker']}]:\n{item['text']}" for item in blocks)
    # Reserved header syntax in spoken text/names cannot round-trip unambiguously.
    if parse_text(result, mode) != narration:
        raise ValueError(
            "This narration requires JSON to preserve literal speaker headers."
        )
    return result


def _commit_draft(mode, key, revision):
    """Only the native form submission may validate/apply a draft."""
    data = _data()
    if revision != data.get("editor_revision", 0):
        return
    draft = st.session_state[key]
    data.setdefault("drafts", {})[mode] = draft
    errors = data.setdefault("draft_errors", {})
    try:
        result = parse_text(draft, mode)
        _require_used_narrators(result)
        _replace_narration(result)
    except ValueError as exc:
        errors[mode] = (
            "Each segment needs non-empty text and a configured speaker."
            if isinstance(exc, ValidationError)
            else str(exc)
        )


def _render_text_editor(mode, tr, active=True):
    data = _data()
    drafts = data.setdefault("drafts", {})
    exports = data.setdefault("exports", {})
    errors = data.setdefault("draft_errors", {})
    current = _representations(data["narrators"], data["segments"])[mode]
    if mode not in drafts:
        drafts[mode] = current
    exports[mode] = current
    revision = data.get("editor_revision", 0)
    slug = "tagged" if mode == "Tagged text" else "json"
    container_key = f"narration_input_{slug}"
    key = f"narration_draft_{slug}_{revision}"
    # Native forms prevent focus loss/typing from applying input. The submit
    # button stays mounted for Ctrl+Enter but is not a visible extra action.
    st.html(
        f"<style>.st-key-{container_key} {{display: {'block' if active else 'none'};}} "
        f".st-key-{container_key} [data-testid='stFormSubmitButton'] {{display:none;}}</style>"
    )
    with st.container(key=container_key):
        with st.form(
            f"narration_form_{slug}_{revision}", border=False, enter_to_submit=True
        ):
            st.text_area(tr(mode), key=key, value=drafts[mode], height=320)
            st.form_submit_button(
                tr("Submit narration"),
                key=f"narration_submit_{slug}_{revision}",
                on_click=_commit_draft,
                args=(mode, key, revision),
            )
        st.caption(tr("Press Ctrl+Enter to validate and apply narration."))
        if active and mode in errors:
            st.error(tr(errors[mode]))


def prepare_preview(params):
    """Return in-memory audio; the temporary directory belongs only to this preview."""
    snapshot = params.model_copy(deep=True)
    snapshot.narration = _require_used_narrators(_build())
    snapshot.video_script = snapshot.narration.flatten_script()
    if snapshot.custom_audio_file or voice.is_no_voice(snapshot.voice_name):
        raise ValueError(
            "Multiple narrators require automatic voiceover without uploaded audio."
        )
    snapshot.subtitle_enabled = False
    with config.try_runtime_config_lock() as acquired:
        if not acquired:
            raise ValueError("Generation is busy. Try preview again shortly.")
        with tempfile.TemporaryDirectory(
            prefix="narration-preview-", dir=utils.storage_dir("temp", create=True)
        ) as directory:
            # task_dir accepts an absolute directory; no task record is created.
            result = narration_adapter.prepare(directory, snapshot, apply_volume=True)
            audio = Path(result.audio_file).read_bytes()
            if not audio:
                raise RuntimeError("Narration preview produced no audio.")
            return {
                "audio_bytes": audio,
                "mime_type": "audio/wav",
                "duration": result.audio_duration,
            }


def render_audio_controls(
    params, tr, synthesize, sample_text, estimate, selected_server, preview_slots=None
):
    data = _data()
    request = data.pop("sample_requested", None)
    previews = data.setdefault("sample_previews", {})
    for item in data["narrators"]:
        sample = previews.get(item["id"])
        if sample and sample[:3] != (
            item["voice_name"],
            params.voice_rate,
            params.voice_volume,
        ):
            previews.pop(item["id"])
    try:
        if request is not None:
            previews.pop(request, None)
            narrator = next(item for item in _narrators() if item.id == request)
            result = synthesize(
                content=sample_text(narrator.voice_name),
                preview_type="sample",
                selected_tts_server=selected_server,
                voice_name=narrator.voice_name,
                voice_rate=params.voice_rate,
                voice_volume=params.voice_volume,
            )
            if not result or result.get("busy"):
                raise ValueError("Voice preview unavailable. Please try again.")
            previews[request] = (
                narrator.voice_name,
                params.voice_rate,
                params.voice_volume,
                result,
            )
    except ValueError as exc:
        _show_error(exc, tr)
    except Exception:
        st.error(
            tr("Voice preview failed. Check the selected voice and provider settings.")
        )
    try:
        for narrator_id, slot in (preview_slots or {}).items():
            sample = previews.get(narrator_id)
            if sample:
                slot.audio(sample[3]["audio_bytes"], format=sample[3]["mime_type"])
        narration = _display_narration()
        duration = (
            estimate(narration.flatten_script(), params.voice_rate)
            if narration
            else None
        )
        if duration:
            st.caption(
                f"{tr('Estimated narration duration')}: {duration[0]:.0f}–{duration[1]:.0f} s"
            )
        fingerprint = hashlib.sha256(
            repr(
                (
                    narration.model_dump() if narration else None,
                    params.voice_rate,
                    params.voice_volume,
                    config.app,
                )
            ).encode()
        ).hexdigest()
        if st.button(tr("Full audio"), key="narration_full_audio"):
            data.pop("full_preview", None)
            with st.spinner(tr("Generating audio")):
                data["full_preview"] = (fingerprint, prepare_preview(params))
        cached = data.get("full_preview")
        if cached and cached[0] == fingerprint:
            st.audio(cached[1]["audio_bytes"], format=cached[1]["mime_type"])
            st.caption(
                f"{tr('Actual preview duration')}: {cached[1]['duration']:.1f} s"
            )
    except ValueError as exc:
        _show_error(exc, tr)
    except Exception:
        st.error(
            tr("Audio preview failed. Check the selected voice and provider settings.")
        )


def _show_error(exc, tr):
    message = (
        "Complete every narrator name, voice, speaker assignment and segment text."
        if isinstance(exc, ValidationError)
        else str(exc)
    )
    st.error(tr(message))


def prepare_submission(params, voice_mode, uploaded_audio_file, tr):
    if not enabled():
        params.narration = None
        return True
    try:
        if (
            voice_mode != "tts"
            or uploaded_audio_file is not None
            or params.custom_audio_file
            or voice.is_no_voice(params.voice_name)
        ):
            raise ValueError(
                "Multiple narrators require automatic voiceover without uploaded audio."
            )
        narration = _require_used_narrators(_build())
        params.narration = narration
        params.video_script = narration.flatten_script()
        return True
    except ValueError as exc:
        _show_error(exc, tr)
        return False


def restore(params):
    """Called before widgets render, for history and settings preset loading."""
    payload = params.get("narration")
    st.session_state[_ENABLED] = payload is not None
    if payload is None:
        return
    narration = MultiNarration.model_validate(payload)
    for key in list(st.session_state):
        if key.startswith(
            (
                "narration_name_",
                "narration_voice_",
                "narration_speaker_",
                "narration_text_",
            )
        ):
            del st.session_state[key]
    data = _data()
    for key in (
        "drafts",
        "draft_errors",
        "exports",
        "full_preview",
        "sample_preview",
        "sample_previews",
    ):
        data.pop(key, None)
    data["narrators"] = copy.deepcopy(
        [item.model_dump() for item in narration.narrators]
    )
    data["next_narrator"] = (
        max(
            [
                int(item.id.removeprefix("narrator_"))
                for item in narration.narrators
                if item.id.startswith("narrator_")
                and item.id.removeprefix("narrator_").isdigit()
            ]
            + [0]
        )
        + 1
    )
    data["voices"].update(
        {item.voice_name: item.voice_name for item in narration.narrators}
    )
    _replace_narration(narration)
