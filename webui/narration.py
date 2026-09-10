"""Optional Streamlit narration controls; backend services own generation."""

import copy

import streamlit as st
from pydantic import ValidationError

from app.models.narration import MultiNarration, Narrator
from app.services import llm, narration_script, voice
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
    st.toggle(
        tr("Use multiple narrators"), key=_ENABLED, value=False, on_change=_toggle
    )
    if enabled() and not _data()["narrators"]:
        _toggle()


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


def add_segment():
    data = _data()
    data["segments"].append(
        {
            "key": data["next_segment"],
            "speaker_id": data["narrators"][0]["id"],
            "text": "",
        }
    )
    data["next_segment"] += 1


def remove_segment(key):
    data = _data()
    data["segments"] = [item for item in data["segments"] if item["key"] != key]


def _edit(collection, identity_field, identity, field, widget_key):
    for item in _data()[collection]:
        if item[identity_field] == identity:
            item[field] = st.session_state[widget_key]
            return


def render_narrators(voice_options, tr):
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
        with st.container(border=True):
            name_key = f"narration_name_{item['id']}"
            st.text_input(
                tr("Name"),
                value=item["display_name"],
                key=name_key,
                on_change=_edit,
                args=("narrators", "id", item["id"], "display_name", name_key),
            )
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
            reason = removal_reason(item["id"])
            st.button(
                tr("Remove narrator"),
                key=f"narration_remove_{item['id']}",
                disabled=bool(reason),
                help=tr(reason) if reason else None,
                on_click=remove_narrator,
                args=(item["id"],),
            )
    st.button(tr("Add narrator"), key="narration_add_narrator", on_click=add_narrator)
    return data["narrators"][0]["voice_name"] or ""


def _narrators():
    data = _data()
    if len(data["narrators"]) < 2:
        raise ValueError("Select at least two narrators.")
    narrators = [Narrator.model_validate(item) for item in data["narrators"]]
    if len({item.id for item in narrators}) != len(narrators):
        raise ValueError("Narrator IDs must be unique.")
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


def _set_segments(segments):
    data = _data()
    data["segments"] = []
    for segment in segments:
        data["segments"].append({"key": data["next_segment"], **segment.model_dump()})
        data["next_segment"] += 1


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
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
            app_config=app_config,
        )
        terms = llm.generate_terms(
            params.video_subject,
            utils.remove_pause_tags(result.flatten_script()),
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
            app_config=app_config,
        )
        return result, terms

    result, terms = run_operation("generate_narration_and_terms", operation)
    _set_segments(result.segments)
    st.session_state["video_terms"] = ", ".join(terms)
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
    st.button(tr("Add segment"), key="narration_add_segment", on_click=add_segment)
    try:
        params.narration = _build()
        params.video_script = params.narration.flatten_script()
    except ValueError:
        params.narration = None
        params.video_script = ""


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
        narration = _build()
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
    _set_segments(narration.segments)
