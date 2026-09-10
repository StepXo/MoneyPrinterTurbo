import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.models.narration import MultiNarration
from app.models.schema import VideoParams
from app.services import voice, webui_task
from webui import narration as ui

VOICES = {"en-US-JennyNeural-Female": "Jenny", "en-US-GuyNeural-Male": "Guy"}


@pytest.fixture
def state():
    state = {"video_script": "Single script.", "video_terms": "single"}
    with patch.object(ui, "st", SimpleNamespace(session_state=state, error=Mock())):
        yield state


def activate(state):
    state["narration_enabled"] = True
    ui._data()["voices"].update(VOICES)
    ui._toggle()


def valid_segments():
    ui.add_segment()
    ui._data()["segments"][0]["text"] = "First."
    ui.add_segment()
    ui._data()["segments"][1].update(speaker_id="narrator_2", text="Second.")


def test_initial_state_stable_ids_add_remove_and_rename(state):
    assert not ui.enabled()
    activate(state)
    data = ui._data()
    assert [item["id"] for item in data["narrators"]] == ["narrator_1", "narrator_2"]
    data["narrators"][0]["display_name"] = "Changed"
    ui._toggle()
    assert data["narrators"][0]["id"] == "narrator_1"
    assert len(data["narrators"]) == 2
    with pytest.raises(ValueError, match="two"):
        ui.remove_narrator("narrator_1")
    ui.add_narrator()
    ui.remove_narrator("narrator_2")
    ui.add_narrator()
    assert [item["id"] for item in data["narrators"]] == [
        "narrator_1",
        "narrator_3",
        "narrator_4",
    ]


def test_referenced_narrator_cannot_be_removed(state):
    activate(state)
    ui.add_narrator()
    valid_segments()
    before = copy.deepcopy(ui._data())
    with pytest.raises(ValueError, match="Reassign"):
        ui.remove_narrator("narrator_1")
    assert ui._data() == before


def test_segment_edit_assignment_add_remove_and_submission(state):
    activate(state)
    valid_segments()
    first = ui._data()["segments"][0]["key"]
    state["edit"] = "Edited."
    ui._edit("segments", "key", first, "text", "edit")
    state["edit"] = "narrator_2"
    ui._edit("segments", "key", first, "speaker_id", "edit")
    ui.remove_segment(ui._data()["segments"][1]["key"])
    ui.add_segment()
    ui._data()["segments"][-1]["text"] = "Last."
    params = VideoParams(video_subject="Topic")
    assert ui.prepare_submission(params, "tts", None, str)
    assert params.video_script == params.narration.flatten_script() == "Edited. Last."
    assert params.narration.segments[0].speaker_id == "narrator_2"
    assert params.narration.narrators[1].voice_name == list(VOICES)[1]


@pytest.mark.parametrize(
    "failure",
    [
        "empty_text",
        "unknown_speaker",
        "no_segments",
        "empty_name",
        "empty_voice",
        "unknown_voice",
        "duplicate_id",
        "one_narrator",
        "upload",
        "no_voice",
        "upload_mode",
    ],
)
def test_invalid_submission_is_rejected(state, failure):
    activate(state)
    valid_segments()
    data = ui._data()
    params = VideoParams(video_subject="Topic")
    mode, upload = "tts", None
    if failure == "empty_text":
        data["segments"][0]["text"] = " "
    elif failure == "unknown_speaker":
        data["segments"][0]["speaker_id"] = "missing"
    elif failure == "no_segments":
        data["segments"] = []
    elif failure == "empty_name":
        data["narrators"][0]["display_name"] = " "
    elif failure == "empty_voice":
        data["narrators"][0]["voice_name"] = ""
    elif failure == "unknown_voice":
        data["narrators"][0]["voice_name"] = "unknown"
    elif failure == "duplicate_id":
        data["narrators"][1]["id"] = "narrator_1"
    elif failure == "one_narrator":
        data["narrators"].pop()
    elif failure == "upload":
        params.custom_audio_file = "uploaded.wav"
    elif failure == "no_voice":
        params.voice_name = voice.NO_VOICE_NAME
    elif failure == "upload_mode":
        mode, upload = "upload", object()
    assert not ui.prepare_submission(params, mode, upload, str)
    ui.st.error.assert_called_once()


def test_toggle_off_preserves_work_without_leaking_multi_params(state):
    activate(state)
    valid_segments()
    params = VideoParams(video_subject="Topic")
    assert ui.prepare_submission(params, "tts", None, str)
    before = copy.deepcopy(ui._data()["segments"])
    state["narration_enabled"] = False
    ui._toggle()
    params.video_script = state["video_script"]
    assert ui.prepare_submission(params, "upload", object(), str)
    assert params.narration is None
    assert params.video_script == "Single script."
    state["narration_enabled"] = True
    ui._toggle()
    assert ui._data()["segments"] == before


def test_generation_uses_current_definitions_context_and_keeps_order(state):
    activate(state)
    ui._data()["narrators"][0]["display_name"] = "Host"
    valid_segments()
    expected = ui._build()
    params = VideoParams(
        video_subject="Coffee",
        video_language="es",
        paragraph_number=3,
        custom_system_prompt="Warm",
        video_script_prompt="Brief",
    )
    snapshot = {"llm_provider": "ollama"}
    with (
        patch.object(
            ui.narration_script, "generate", return_value=expected
        ) as generate,
        patch.object(ui.llm, "generate_terms", return_value=["coffee"]) as terms,
    ):
        result = ui.generate(params, lambda name, operation: operation(snapshot))
    assert result == expected
    assert ui._build() == expected
    kwargs = generate.call_args.kwargs
    assert kwargs["narrators"][0].display_name == "Host"
    assert kwargs["narrators"][0].voice_name == list(VOICES)[0]
    assert kwargs["video_subject"] == "Coffee"
    assert kwargs["custom_system_prompt"] == "Warm"
    assert kwargs["video_script_prompt"] == "Brief"
    assert kwargs["language"] == "es"
    assert kwargs["paragraph_number"] == 3
    assert kwargs["app_config"] is snapshot
    assert terms.call_args.args[1] == expected.flatten_script()


def test_restoration_preserves_every_field_and_safe_counter(state):
    activate(state)
    valid_segments()
    payload = ui._build().model_dump()
    payload["narrators"][1]["id"] = "narrator_20"
    payload["segments"][1]["speaker_id"] = "narrator_20"
    ui.restore({"narration": payload})
    assert ui.enabled()
    assert ui._build().model_dump() == payload
    ui.add_narrator()
    assert ui._data()["narrators"][-1]["id"] == "narrator_21"
    ui.restore({"video_script": "Legacy"})
    assert not ui.enabled()


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app_environment():
    with (
        patch.object(
            config,
            "app",
            dict(config.app, video_source="pexels", script_generation_backend="local"),
        ),
        patch.object(
            config,
            "ui",
            dict(
                config.ui,
                language="en",
                voice_mode="tts",
                tts_server="azure-tts-v1",
                voice_name=list(VOICES)[0],
            ),
        ),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(voice, "get_all_azure_voices", return_value=list(VOICES)),
        patch.object(webui_task, "submit_generation") as submit,
    ):
        yield submit


def widget(elements, key):
    exact = [item for item in elements if item.key == key]
    if exact:
        return exact[0]
    return next(
        item
        for item in elements
        if item.key == key or str(item.key).startswith(key + "_")
    )


def app():
    result = AppTest.from_file(str(ROOT / "webui/Main.py"), default_timeout=60)
    result.session_state["ui_language"] = "en"
    result.run()
    assert not result.exception
    return result


def test_real_gui_manual_edit_toggle_and_submission(app_environment):
    page = app()
    assert not page.toggle(key="narration_enabled").value
    widget(page.text_area, "video_script").set_value("Legacy text.").run()
    page.toggle(key="narration_enabled").set_value(True).run()
    assert not page.exception
    assert len(page.session_state["narration_data"]["narrators"]) == 2
    assert not any(
        str(item.key).startswith("voice_mode_control")
        for item in page.get("segmented_control")
    )
    assert not any("voice_preview" in str(item.key) for item in page.button)
    page.button(key="narration_add_segment").click().run()
    page.text_area(key="narration_text_1").set_value("First.").run()
    page.selectbox(key="narration_speaker_1").set_value("narrator_2").run()
    page.text_input(key="narration_name_narrator_2").set_value("Guest").run()
    page.selectbox(key="narration_voice_narrator_2").set_value(list(VOICES)[0]).run()
    page.button(key="generate_video_button").click().run()
    assert not page.exception
    params = app_environment.call_args.kwargs["params"]
    assert params.video_script == "First."
    assert params.narration.segments[0].speaker_id == "narrator_2"
    assert params.narration.narrators[1].display_name == "Guest"
    assert params.narration.narrators[1].voice_name == list(VOICES)[0]
    assert app_environment.call_args.kwargs["voice_preview"] is None
    page.toggle(key="narration_enabled").set_value(False).run()
    assert widget(page.text_area, "video_script").value == "Legacy text."
    page.toggle(key="narration_enabled").set_value(True).run()
    assert page.text_area(key="narration_text_1").value == "First."
    assert not page.exception


def test_real_gui_generation_and_invalid_submission(app_environment):
    page = app()
    page.toggle(key="narration_enabled").set_value(True).run()
    page.button(key="generate_video_button").click().run()
    app_environment.assert_not_called()
    widget(page.text_area, "video_subject").set_value("Coffee").run()
    data = page.session_state["narration_data"]
    result = MultiNarration(
        narrators=data["narrators"],
        segments=[{"speaker_id": "narrator_2", "text": "Generated."}],
    )
    with (
        patch.object(ui.narration_script, "generate", return_value=result) as generate,
        patch.object(ui.llm, "generate_terms", return_value=["coffee"]),
    ):
        page.button(key="narration_generate").click().run()
    generate.assert_called_once()
    assert page.text_area(key="narration_text_1").value == "Generated."
    assert not page.exception


def test_real_gui_restoration_uses_existing_preset_restore_boundary(app_environment):
    page = app()
    payload = {
        "narrators": [
            {"id": "narrator_7", "display_name": "Host", "voice_name": list(VOICES)[0]},
            {
                "id": "narrator_12",
                "display_name": "Guest",
                "voice_name": list(VOICES)[1],
            },
        ],
        "segments": [{"speaker_id": "narrator_12", "text": "Restored."}],
    }
    page.session_state["settings_preset_payload"] = {
        "video_subject": "Restored topic",
        "narration": payload,
        "video_script": "Restored.",
        "voice_name": list(VOICES)[0],
    }
    page.run()
    assert not page.exception
    assert page.toggle(key="narration_enabled").value
    assert page.text_input(key="narration_name_narrator_12").value == "Guest"
    assert page.selectbox(key="narration_speaker_1").value == "narrator_12"
    page.button(key="narration_add_narrator").click().run()
    assert page.session_state["narration_data"]["narrators"][-1]["id"] == "narrator_13"
    page.session_state["settings_preset_payload"] = {
        "video_subject": "Legacy",
        "video_script": "Legacy restored.",
        "voice_name": list(VOICES)[0],
    }
    page.run()
    assert not page.exception
    assert not page.toggle(key="narration_enabled").value
    assert widget(page.text_area, "video_script").value == "Legacy restored."


@pytest.mark.parametrize("mode", ["upload", "none"])
def test_real_gui_incompatible_modes_are_unavailable_and_restored_when_off(
    app_environment, mode
):
    with patch.dict(config.ui, {"voice_mode": mode}):
        page = app()
        assert page.session_state["voice_mode_control_en"] == mode
        page.toggle(key="narration_enabled").set_value(True).run()
        assert not page.exception
        assert not any(
            str(item.key).startswith("voice_mode_control")
            for item in page.get("segmented_control")
        )
        assert not any(
            item.key == "custom_audio_file_uploader"
            for item in page.get("file_uploader")
        )
        page.toggle(key="narration_enabled").set_value(False).run()
        assert page.session_state["voice_mode_control_en"] == mode


def test_real_gui_loomloom_multi_conversion_is_explicitly_unavailable(app_environment):
    page = app()
    widget(page.selectbox, "script_generation_backend_select").set_value(
        "loomloom"
    ).run()
    page.toggle(key="narration_enabled").set_value(True).run()
    assert not page.exception
    assert any("LoomLoom conversion is unavailable" in item.value for item in page.info)
    assert not any(
        item.key in ("narration_generate", "loomloom_apply_candidate")
        for item in page.button
    )
    assert any(item.key == "narration_add_segment" for item in page.button)
