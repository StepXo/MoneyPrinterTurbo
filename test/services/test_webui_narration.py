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
USED_NARRATORS_ERROR = (
    "Multi-narrator mode requires at least two narrators to be used in the narration."
)


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


@pytest.mark.parametrize("operation", ["preview", "submission"])
def test_one_used_speaker_blocks_consumption_without_changing_editor_state(
    state, operation
):
    activate(state)
    valid_segments()
    second = ui._data()["segments"][1]["key"]
    state["speaker_edit"] = "narrator_1"
    ui._edit("segments", "key", second, "speaker_id", "speaker_edit")
    assert ui._display_narration().flatten_script() == "First. Second."
    ui.st.error.assert_not_called()
    ui._data()["drafts"] = {"Tagged text": "unfinished draft", "JSON": "{"}
    original = copy.deepcopy(state)
    params = VideoParams(video_subject="Topic", video_script="Previous script.")
    original_params = params.model_copy(deep=True)
    with patch.object(ui.narration_adapter, "prepare") as adapter:
        if operation == "preview":
            with pytest.raises(ValueError) as failure:
                ui.prepare_preview(params)
            assert str(failure.value) == USED_NARRATORS_ERROR
        else:
            assert not ui.prepare_submission(params, "tts", None, str)
            ui.st.error.assert_called_once_with(USED_NARRATORS_ERROR)
        adapter.assert_not_called()
    assert state == original
    assert params == original_params


def test_single_mode_does_not_require_two_used_speakers(state):
    activate(state)
    ui.add_segment()
    ui._data()["segments"][0]["text"] = "Only speaker."
    state["narration_enabled"] = False
    params = VideoParams(video_subject="Topic", video_script="Legacy script.")
    assert ui.prepare_submission(params, "tts", None, str)
    assert params.narration is None
    assert params.video_script == "Legacy script."
    ui.st.error.assert_not_called()


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
    assert kwargs["video_script_prompt"].startswith("Brief\n")
    assert "at least two distinct" in kwargs["video_script_prompt"]
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
    # Tests exercising segment controls select Visual editor explicitly; the UI
    # may default to the paste-oriented Tagged text mode.
    result.session_state["narration_editor"] = "Visual editor"
    return result


def draft_widget(page, mode="Tagged text"):
    slug = "tagged" if mode == "Tagged text" else "json"
    revision = page.session_state["narration_data"].get("editor_revision", 0)
    return page.text_area(key=f"narration_draft_{slug}_{revision}")


def submit_draft(page, mode="Tagged text"):
    slug = "tagged" if mode == "Tagged text" else "json"
    revision = page.session_state["narration_data"].get("editor_revision", 0)
    # The native form's first submit button is invoked by Ctrl+Enter in-browser.
    return page.button(key=f"narration_submit_{slug}_{revision}").click().run()


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
    widget(page.button, "narration_add_segment").click().run()
    page.text_area(key="narration_text_1").set_value("First.").run()
    page.selectbox(key="narration_speaker_1").set_value("narrator_2").run()
    page.text_input(key="narration_name_narrator_2").set_value("Guest").run()
    page.selectbox(key="narration_voice_narrator_2").set_value(list(VOICES)[0]).run()
    assert not page.error
    with patch.object(ui.narration_adapter, "prepare") as adapter:
        page.button(key="narration_full_audio").click().run()
        adapter.assert_not_called()
    assert [item.value for item in page.error] == [USED_NARRATORS_ERROR]
    assert page.text_area(key="narration_text_1").value == "First."
    assert page.selectbox(key="narration_speaker_1").value == "narrator_2"
    page.button(key="generate_video_button").click().run()
    app_environment.assert_not_called()
    assert [item.value for item in page.error] == [USED_NARRATORS_ERROR]
    widget(page.button, "narration_add_segment").click().run()
    page.text_area(key="narration_text_2").set_value("Second.").run()
    page.button(key="generate_video_button").click().run()
    assert not page.exception
    params = app_environment.call_args.kwargs["params"]
    assert params.video_script == "First. Second."
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
        segments=[
            {"speaker_id": "narrator_2", "text": "Generated."},
            {"speaker_id": "narrator_1", "text": "Reply."},
        ],
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
    assert any(
        str(item.key).startswith("narration_add_segment_") for item in page.button
    )


@pytest.mark.parametrize("mode", ["Tagged text", "JSON"])
def test_text_round_trip_n_speakers_multiline_and_rename(state, mode):
    activate(state)
    valid_segments()
    ui.add_narrator()
    ui.add_segment()
    ui._data()["segments"][-1].update(
        speaker_id="narrator_3", text="Third.\n\nParagraph two."
    )
    ui.add_segment()
    ui._data()["segments"][-1]["text"] = "Again."
    before = ui._build()
    encoded = ui.export_text(mode)
    assert "narrator_1" not in encoded
    assert ui.parse_text(encoded, mode) == before
    ui._data()["narrators"][0]["display_name"] = "Host"
    encoded = ui.export_text(mode)
    assert "Host" in encoded
    restored = ui.parse_text(encoded, mode)
    assert restored.segments == before.segments
    assert restored.segments[2].text == "Third.\n\nParagraph two."


@pytest.mark.parametrize(
    "mode,text",
    [
        ("Tagged text", "[Unknown]:\nHello"),
        ("Tagged text", "[Narrator 1]:\n\n[Narrator 2]:\nHello"),
        ("Tagged text", "Before\n[Narrator 1]:\nHello"),
        ("Tagged text", "[Narrator 1]\nHello"),
        ("Tagged text", "[]:\nHello"),
        ("JSON", "bad"),
        ("JSON", "[]"),
        ("JSON", "{}"),
        ("JSON", '{"segments": []}'),
        ("JSON", '{"segments": [{"speaker": "Unknown", "text": "Hello"}]}'),
        ("JSON", '{"segments": [{"speaker": "Narrator 1", "text": " "}]}'),
        ("JSON", '{"segments": [{"speaker_id": "narrator_1", "text": "Hello"}]}'),
        ("JSON", '{"segments": [{"speaker": [], "text": "Hello"}]}'),
        ("JSON", 'Prose {"segments": []}'),
    ],
)
def test_invalid_text_import_never_changes_state(state, mode, text):
    activate(state)
    valid_segments()
    before = copy.deepcopy(ui._data())
    with pytest.raises(ValueError):
        ui.parse_text(text, mode)
    assert ui._data() == before


@pytest.mark.parametrize("mode", ["JSON", "Tagged text"])
def test_duplicate_names_rejected(state, mode):
    activate(state)
    ui._data()["narrators"][1]["display_name"] = "Narrator 1"
    with pytest.raises(ValueError, match="unique"):
        ui.parse_text("", mode)
    with patch.object(ui.narration_script, "generate") as generate:
        with pytest.raises(ValueError, match="unique"):
            ui.generate(VideoParams(video_subject="Topic"), lambda name, op: op({}))
    generate.assert_not_called()


def test_generated_one_speaker_rejected_without_extra_retry_or_keywords(state):
    activate(state)
    valid_segments()
    before = ui._build()
    result = before.model_copy(deep=True)
    result.segments[1].speaker_id = result.segments[0].speaker_id
    with (
        patch.object(ui.narration_script, "generate", return_value=result) as generate,
        patch.object(ui.llm, "generate_terms") as terms,
    ):
        with pytest.raises(ValueError, match="at least two"):
            ui.generate(VideoParams(video_subject="Topic"), lambda name, op: op({}))
    generate.assert_called_once()
    terms.assert_not_called()
    assert ui._build() == before


def test_sample_preview_success_and_complete_duration_input(state):
    activate(state)
    valid_segments()
    before = ui._build()
    ui._data()["sample_requested"] = "narrator_2"
    params = VideoParams(video_subject="Topic", voice_volume=0.8, voice_rate=1.5)
    synthesize = Mock(return_value={"audio_bytes": b"sample", "mime_type": "audio/wav"})
    sample_text = Mock(return_value="Sample sentence.")
    estimate = Mock(return_value=(3.0, 4.0))
    slot = Mock()
    with patch.object(
        ui,
        "st",
        SimpleNamespace(
            session_state=state,
            audio=Mock(),
            caption=Mock(),
            button=Mock(return_value=False),
            error=Mock(),
        ),
    ):
        ui.render_audio_controls(
            params,
            str,
            synthesize,
            sample_text,
            estimate,
            "azure-tts-v1",
            {"narrator_2": slot},
        )
        ui.st.error.assert_not_called()
        slot.audio.assert_called_once_with(b"sample", format="audio/wav")
        ui.st.caption.assert_called_once_with("Estimated narration duration: 3–4 s")
    synthesize.assert_called_once_with(
        content="Sample sentence.",
        preview_type="sample",
        selected_tts_server="azure-tts-v1",
        voice_name=list(VOICES)[1],
        voice_rate=1.5,
        voice_volume=0.8,
    )
    estimate.assert_called_once_with(before.flatten_script(), 1.5)
    assert ui._build() == before


def test_duration_estimation_scales_with_entire_script_and_rate():
    import ast
    import re

    tree = ast.parse((ROOT / "webui/Main.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_estimate_voiceover_duration_range"
    )
    namespace = {"re": re}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "Main.py", "exec"),
        namespace,
    )
    estimate = namespace[function.name]
    text = "This narration contains several spoken words. " * 10
    normal = estimate(text, 1.0)
    assert normal[0] > 0
    assert estimate(text * 2, 1.0)[0] > normal[0]
    assert estimate(text, 0.8)[0] > normal[0] > estimate(text, 1.5)[0]
    assert estimate("", 1.0) is None


def test_independent_sample_players_invalidate_only_changed_settings(state):
    activate(state)
    valid_segments()
    params = VideoParams(video_subject="Topic")
    slots = {item["id"]: Mock() for item in ui._data()["narrators"]}
    synthesize = Mock(
        side_effect=lambda **kw: {
            "audio_bytes": kw["voice_name"].encode(),
            "mime_type": "audio/wav",
        }
    )
    fake = SimpleNamespace(
        session_state=state,
        caption=Mock(),
        button=Mock(return_value=False),
        error=Mock(),
    )
    with patch.object(ui, "st", fake):

        def render():
            ui.render_audio_controls(
                params,
                str,
                synthesize,
                str,
                Mock(return_value=None),
                "azure-tts-v1",
                slots,
            )

        for narrator_id in slots:
            ui._data()["sample_requested"] = narrator_id
            render()
        assert set(ui._data()["sample_previews"]) == set(slots)
        for index, slot in enumerate(slots.values()):
            slot.audio.assert_called_with(
                list(VOICES)[index].encode(), format="audio/wav"
            )
        ui._data()["narrators"][0]["display_name"] = "Renamed"
        render()
        assert len(ui._data()["sample_previews"]) == 2
        ui._data()["narrators"][0]["voice_name"] = list(VOICES)[1]
        render()
        assert set(ui._data()["sample_previews"]) == {"narrator_2"}
        params.voice_rate = 1.5
        render()
        assert not ui._data()["sample_previews"]
        ui._data()["sample_requested"] = "narrator_2"
        render()
        params.voice_volume = 0.6
        render()
        assert not ui._data()["sample_previews"]
        fake.error.assert_not_called()


@pytest.mark.parametrize("button_key", ["narration_generate", "auto_generate_terms"])
@pytest.mark.parametrize("failure", [False, True])
def test_real_ai_operation_locks_mode_and_always_unlocks(
    app_environment, button_key, failure
):
    from streamlit.delta_generator import DeltaGenerator

    page = app()
    assert not page.toggle(key="narration_enabled").disabled
    page.toggle(key="narration_enabled").set_value(True).run()
    result = MultiNarration(
        narrators=page.session_state["narration_data"]["narrators"],
        segments=[
            {"speaker_id": "narrator_1", "text": "First."},
            {"speaker_id": "narrator_2", "text": "Second."},
        ],
    )
    page.session_state["settings_preset_payload"] = {
        "video_subject": "Topic",
        "video_script": result.flatten_script(),
        "narration": result.model_dump(),
    }
    page.run()
    events = []
    original = DeltaGenerator.toggle

    def toggle(self, *args, **kwargs):
        if kwargs.get("key") in ("narration_enabled", "narration_locked_mode"):
            events.append(kwargs.get("disabled", False))
        return original(self, *args, **kwargs)

    def operation(*args, **kwargs):
        assert events[-1] is True
        if failure:
            raise RuntimeError("test provider failure")
        return result if button_key == "narration_generate" else ["keyword"]

    with (
        patch.object(DeltaGenerator, "toggle", toggle),
        patch.object(
            ui.narration_script,
            "generate",
            side_effect=operation if button_key == "narration_generate" else None,
        ),
        patch.object(
            ui.llm,
            "generate_terms",
            side_effect=operation
            if button_key == "auto_generate_terms"
            else lambda *a, **k: ["keyword"],
        ),
    ):
        page.button(key=button_key).click().run()
    assert events == [True, False]
    assert not page.toggle(key="narration_enabled").disabled
    page.run()
    assert not page.exception
    assert not page.toggle(key="narration_enabled").disabled


def test_visual_editor_expander_contains_all_controls_and_preserves_content(
    app_environment,
):
    page = app()
    with patch.object(ui.st, "container", wraps=ui.st.container) as container:
        page.toggle(key="narration_enabled").set_value(True).run()
    assert any(
        call.kwargs == {"key": "advanced_settings_narration"}
        for call in container.call_args_list
    )
    widget(page.button, "narration_add_segment").click().run()
    page.text_area(key="narration_text_1").set_value(
        "Exact text.\n\nSecond paragraph."
    ).run()
    before = copy.deepcopy(page.session_state["narration_data"])
    estimates = [
        c.value for c in page.caption if "Estimated narration duration" in c.value
    ]

    def editor():
        return next(item for item in page.expander if item.label == "Visual editor")

    def exports():
        with patch.object(
            ui,
            "st",
            SimpleNamespace(
                session_state={"narration_data": page.session_state["narration_data"]}
            ),
        ):
            return (
                ui._build().flatten_script(),
                ui.export_text("Tagged text"),
                ui.export_text("JSON"),
            )

    before_exports = exports()
    assert editor().proto.expanded
    assert not any(item.key == "narration_hide_text" for item in page.toggle)
    original = ui.st.expander

    def collapsed(label, *args, **kwargs):
        if label == "Visual editor":
            kwargs["expanded"] = False
        return original(label, *args, **kwargs)

    # AppTest has no browser-side expander click; render the native closed state.
    with patch.object(ui.st, "expander", collapsed):
        page.run()
    assert not editor().proto.expanded
    assert editor().selectbox(key="narration_speaker_1").value == "narrator_1"
    assert (
        editor().text_area(key="narration_text_1").value
        == before["segments"][0]["text"]
    )
    assert editor().button(key="narration_remove_segment_1")
    assert widget(editor().button, "narration_add_segment")
    assert page.session_state["narration_data"]["segments"] == before["segments"]
    assert exports() == before_exports
    assert [
        c.value for c in page.caption if "Estimated narration duration" in c.value
    ] == estimates
    page.run()
    assert editor().proto.expanded
    assert page.text_area(key="narration_text_1").value == before["segments"][0]["text"]
    assert page.selectbox(key="narration_speaker_1").value == "narrator_1"
    assert exports() == before_exports
    assert not page.exception
    sample = page.button(key="narration_sample_narrator_1")
    delete = page.button(key="narration_remove_narrator_1")
    assert sample.label == "Voice Sample"
    assert sample.proto.icon == ":material/graphic_eq:"
    assert delete.proto.icon == ":material/delete:"
    assert delete.proto.type == "primary"
    assert delete.disabled


@pytest.mark.parametrize("fail", [False, True])
def test_full_preview_contract_cleanup_and_failure(state, tmp_path, fail):
    activate(state)
    valid_segments()
    original = ui._build()
    params = VideoParams(video_subject="Topic", voice_volume=1.5, voice_rate=1.2)
    before = params.model_dump()
    folders = []

    def prepare(directory, snapshot, **kwargs):
        folders.append(Path(directory))
        assert snapshot.narration == original
        assert snapshot.video_script == original.flatten_script()
        assert not snapshot.subtitle_enabled
        assert snapshot.voice_volume == 1.5 and snapshot.voice_rate == 1.2
        assert kwargs == {"apply_volume": True}
        output = Path(directory) / "preview.wav"
        output.write_bytes(b"audio")
        if fail:
            raise RuntimeError("preview failed")
        return SimpleNamespace(
            audio_file=str(output), audio_duration=3.25, subtitle_path=""
        )

    unrelated = tmp_path / "keep"
    unrelated.write_text("keep")
    with (
        patch.object(ui.utils, "storage_dir", return_value=str(tmp_path)),
        patch.object(ui.narration_adapter, "prepare", side_effect=prepare) as adapter,
    ):
        if fail:
            with pytest.raises(RuntimeError, match="preview failed"):
                ui.prepare_preview(params)
        else:
            assert ui.prepare_preview(params) == {
                "audio_bytes": b"audio",
                "mime_type": "audio/wav",
                "duration": 3.25,
            }
    adapter.assert_called_once()
    assert all(not folder.exists() for folder in folders)
    assert unrelated.read_text() == "keep"
    assert ui._build() == original
    assert params.model_dump() == before


@pytest.mark.parametrize("mode", ["Tagged text", "JSON"])
def test_textarea_commit_validates_and_preserves_invalid_drafts(app_environment, mode):
    page = app()
    page.toggle(key="narration_enabled").set_value(True).run()
    widget(page.button, "narration_add_segment").click().run()
    page.text_area(key="narration_text_1").set_value("Original.").run()
    page.radio(key="narration_editor").set_value(mode).run()
    assert "Original." in draft_widget(page, mode).value
    before = copy.deepcopy(page.session_state["narration_data"]["segments"])
    # Editing/focus loss is draft-only. Native form submission validates once.
    with patch.object(ui, "parse_text", wraps=ui.parse_text) as parse:
        draft_widget(page, mode).set_value("Incomplete draft").run()
        parse.assert_not_called()
        assert not page.error
        submit_draft(page, mode)
        assert any(
            call.args == ("Incomplete draft", mode) for call in parse.call_args_list
        )
    assert page.session_state["narration_data"]["segments"] == before
    assert len(page.error) == 1
    # Invalid drafts do not poison consumers of the previous valid narration.
    page.radio(key="narration_editor").set_value("Visual editor").run()
    page.text_area(key="narration_text_1").set_value("Changed visually.").run()
    page.radio(key="narration_editor").set_value(mode).run()
    assert draft_widget(page, mode).value == "Incomplete draft"
    with patch.object(ui, "parse_text", wraps=ui.parse_text) as parse:
        page.run()
        assert not any(
            call.args == ("Incomplete draft", mode) for call in parse.call_args_list
        )
    valid = (
        "[Narrator 2]:\nCommitted.\n\n[Narrator 1]: Reply."
        if mode == "Tagged text"
        else '{"segments":[{"speaker":"Narrator 2","text":"Committed."},{"speaker":"Narrator 1","text":"Reply."}]}'
    )
    draft_widget(page, mode).set_value(valid)
    submit_draft(page, mode)
    segments = page.session_state["narration_data"]["segments"]
    assert [(item["speaker_id"], item["text"]) for item in segments] == [
        ("narrator_2", "Committed."),
        ("narrator_1", "Reply."),
    ]
    assert not page.session_state["narration_data"]["draft_errors"]
    page.radio(key="narration_editor").set_value("Visual editor").run()
    page.text_area(key=f"narration_text_{segments[0]['key']}").set_value(
        "Synchronized."
    ).run()
    page.radio(key="narration_editor").set_value(mode).run()
    assert "Synchronized." in draft_widget(page, mode).value
    assert not page.exception


def test_real_gui_drafts_roundtrip_global_controls_and_previews(app_environment):
    page = app()
    page.toggle(key="narration_enabled").set_value(True).run()
    page.radio(key="narration_editor").set_value("Tagged text").run()
    text = (
        "[Narrator 1]:\nFirst paragraph.\n\nSecond paragraph.\n\n[Narrator 2]:\nReply."
    )
    draft_widget(page).set_value(text)
    submit_draft(page)
    page.radio(key="narration_editor").set_value("JSON").run()
    page.radio(key="narration_editor").set_value("Tagged text").run()
    assert draft_widget(page).value == text
    assert not any(
        item.key in ("narration_apply_text", "narration_load_text")
        for item in page.button
    )
    assert not page.exception
    assert len(page.session_state["narration_data"]["segments"]) == 2
    page.radio(key="narration_editor").set_value("Visual editor").run()
    assert (
        page.text_area(key="narration_text_1").value
        == "First paragraph.\n\nSecond paragraph."
    )
    widget(page.selectbox, "voice_volume_select").set_value(1.5).run()
    widget(page.selectbox, "voice_rate_select").set_value(1.2).run()
    before = copy.deepcopy(page.session_state["narration_data"]["segments"])
    with patch.object(
        voice, "tts", side_effect=RuntimeError("fake provider failure")
    ) as tts:
        page.button(key="narration_sample_narrator_2").click().run()
    assert not page.exception
    assert tts.call_args.kwargs["voice_name"] == list(VOICES)[1]
    assert tts.call_args.kwargs["voice_rate"] == 1.2
    assert tts.call_args.kwargs["voice_volume"] == 1.5
    assert page.session_state["narration_data"]["segments"] == before
    with patch.object(
        ui,
        "prepare_preview",
        return_value={
            "audio_bytes": b"audio",
            "mime_type": "audio/wav",
            "duration": 7.3,
        },
    ) as preview:
        page.button(key="narration_full_audio").click().run()
    preview.assert_called_once()
    assert any("Estimated narration duration" in item.value for item in page.caption)
    assert any("Actual preview duration: 7.3" in item.value for item in page.caption)
    app_environment.assert_not_called()
    page.run()
    assert widget(page.selectbox, "voice_volume_select").value == 1.5
    assert widget(page.selectbox, "voice_rate_select").value == 1.2
    assert all(
        "voice_rate" not in item and "voice_volume" not in item
        for item in page.session_state["narration_data"]["narrators"]
    )
    page.text_area(key="narration_text_1").set_value("Changed.").run()
    assert not any("Actual preview duration" in item.value for item in page.caption)
    assert not page.exception


@pytest.mark.parametrize(
    "header",
    ["[camilo]:\nhello", "[camilo]: hello", "[ camilo ]: first\nsecond\nthird"],
)
def test_tagged_display_name_inline_and_multiline_resolution(state, header):
    activate(state)
    ui._data()["narrators"][0]["display_name"] = "camilo"
    result = ui.parse_text(header + "\n[Narrator 2]: other text", "Tagged text")
    assert [item.speaker_id for item in result.segments] == ["narrator_1", "narrator_2"]
    assert result.segments[0].text == (
        "first\nsecond\nthird" if "first" in header else "hello"
    )


@pytest.mark.parametrize("mode", ["Tagged text", "JSON"])
@pytest.mark.parametrize("failure", ["unknown", "empty", "one_speaker", "duplicate"])
def test_import_failure_is_transactional_and_contextual(state, mode, failure):
    activate(state)
    valid_segments()
    original = ui._build()
    if failure == "duplicate":
        ui._data()["narrators"][1]["display_name"] = "Narrator 1"
    speaker = "missing" if failure == "unknown" else "Narrator 1"
    spoken = "" if failure == "empty" else "Only speaker."
    text = (
        f"[{speaker}]: {spoken}"
        if mode == "Tagged text"
        else ui.json.dumps({"segments": [{"speaker": speaker, "text": spoken}]})
    )
    state["submitted"] = text
    ui._commit_draft(mode, "submitted", ui._data().get("editor_revision", 0))
    assert ui._data()["drafts"][mode] == text
    assert [
        {"speaker_id": s["speaker_id"], "text": s["text"]}
        for s in ui._data()["segments"]
    ] == [item.model_dump() for item in original.segments]
    error = ui._data()["draft_errors"][mode]
    if failure == "one_speaker":
        assert error == USED_NARRATORS_ERROR
    elif failure == "duplicate":
        assert "unique" in error
    elif failure == "unknown":
        assert "Unknown narrator" in error
    ui.st.error.assert_not_called()  # emitted once by the active editor, not callback/audio


def test_replacement_rebuilds_both_drafts_and_ignores_obsolete_submit(state):
    activate(state)
    valid_segments()
    before = ui._build()
    old_segment_key = ui._data()["segments"][0]["key"]
    revision = ui._data().get("editor_revision", 0)
    old_key = f"narration_draft_tagged_{revision}"
    state[old_key] = "[Narrator 1]: Stale\n[Narrator 2]: Stale"
    ui._data()["full_preview"] = ("old", {"duration": 15, "audio_bytes": b"old"})
    candidate = before.model_copy(deep=True)
    candidate.segments[0].text = "Replacement."
    ui._replace_narration(candidate)
    installed = copy.deepcopy(ui._data())
    ui._commit_draft("Tagged text", old_key, revision)
    ui.add_segment(revision)
    ui.remove_segment(old_segment_key)
    assert ui._data() == installed
    assert ui._build() == candidate
    assert "full_preview" not in ui._data()
    for mode in ("Tagged text", "JSON"):
        assert ui._data()["drafts"][mode] == ui.export_text(mode)
        assert ui._data()["exports"][mode] == ui.export_text(mode)
    assert [item.model_dump() for item in ui._build().narrators] == [
        item.model_dump() for item in before.narrators
    ]


@pytest.mark.parametrize("failure", ["script", "keywords"])
def test_failed_ai_preserves_every_script_state(state, failure):
    activate(state)
    valid_segments()
    result = ui._build()
    data = ui._data()
    data["drafts"] = {"Tagged text": "unfinished tagged", "JSON": "unfinished JSON"}
    data["full_preview"] = ("old", {"duration": 7})
    state["narration_text_1"] = "widget text"
    before = copy.deepcopy(state)
    with (
        patch.object(
            ui.narration_script,
            "generate",
            side_effect=RuntimeError("provider failed")
            if failure == "script"
            else None,
            return_value=result,
        ),
        patch.object(ui.llm, "generate_terms", return_value=[]),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            ui.generate(VideoParams(video_subject="Topic"), lambda name, op: op({}))
    assert state == before


def test_real_repeated_ai_replaces_all_editors_and_preserves_configuration(
    app_environment,
):
    page = app()
    page.toggle(key="narration_enabled").set_value(True).run()
    widget(page.text_area, "video_subject").set_value("Topic").run()
    widget(page.selectbox, "voice_volume_select").set_value(1.5).run()
    widget(page.selectbox, "voice_rate_select").set_value(1.2).run()
    narrators = copy.deepcopy(page.session_state["narration_data"]["narrators"])
    for index, mode in enumerate(
        ("Visual editor", "Visual editor", "Tagged text", "JSON")
    ):
        page.radio(key="narration_editor").set_value(mode).run()
        draft_widget(page, "Tagged text").set_value("unfinished tagged").run()
        draft_widget(page, "JSON").set_value("unfinished JSON").run()
        page.session_state["narration_data"]["full_preview"] = (
            "stale",
            {"duration": 12},
        )
        previous_revision = page.session_state["narration_data"].get(
            "editor_revision", 0
        )
        result = MultiNarration(
            narrators=narrators,
            segments=[
                {"speaker_id": "narrator_1", "text": f"Generation {index} first."},
                {"speaker_id": "narrator_2", "text": f"Generation {index} second."},
            ],
        )
        with (
            patch.object(ui.narration_script, "generate", return_value=result),
            patch.object(ui.llm, "generate_terms", return_value=["keyword"]),
        ):
            page.button(key="narration_generate").click().run()
        assert not page.exception
        data = page.session_state["narration_data"]
        assert data["editor_revision"] > previous_revision
        assert data["narrators"] == narrators
        assert "full_preview" not in data
        assert not any("Actual preview duration" in c.value for c in page.caption)
        assert [s["text"] for s in data["segments"]] == [
            s.text for s in result.segments
        ]
        for text_mode in ("Tagged text", "JSON"):
            assert "unfinished" not in draft_widget(page, text_mode).value
            assert f"Generation {index}" in draft_widget(page, text_mode).value
        page.run()
        assert [
            s["text"] for s in page.session_state["narration_data"]["segments"]
        ] == [s.text for s in result.segments]
        assert widget(page.selectbox, "voice_volume_select").value == 1.5
        assert widget(page.selectbox, "voice_rate_select").value == 1.2


@pytest.mark.parametrize("mode", ["Tagged text", "JSON"])
def test_real_failed_ai_preserves_unsubmitted_form_drafts(app_environment, mode):
    page = app()
    page.toggle(key="narration_enabled").set_value(True).run()
    widget(page.text_area, "video_subject").set_value("Topic").run()
    page.radio(key="narration_editor").set_value(mode).run()
    draft_widget(page, mode).set_value("Unsubmitted draft").run()
    before = copy.deepcopy(page.session_state["narration_data"])
    with patch.object(
        ui.narration_script, "generate", side_effect=RuntimeError("provider failed")
    ):
        page.button(key="narration_generate").click().run()
    assert draft_widget(page, mode).value == "Unsubmitted draft"
    assert page.session_state["narration_data"] == before
    assert len(page.error) == 1
    assert not page.exception
