import unittest
from contextlib import ExitStack
from unittest.mock import PropertyMock, patch

from app.models.schema import VideoParams
from app.services import task as tm
from app.services.narration_adapter import NarrationArtifacts
from app.services.state import MemoryState


class TestTaskNarration(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.params = VideoParams(video_subject="Coffee", video_script="Legacy script.")
        self.state = MemoryState()
        stack.enter_context(patch.object(tm.sm, "state", self.state))
        stack.enter_context(
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=True)
        )
        stack.enter_context(
            patch.object(
                tm.upload_post.upload_post_service, "is_configured", return_value=False
            )
        )
        self.llm = stack.enter_context(
            patch.object(
                tm.llm,
                "generate_script",
                side_effect=AssertionError("unexpected LLM call"),
            )
        )
        self.terms = stack.enter_context(
            patch.object(tm, "generate_terms", return_value=["coffee"])
        )
        self.save = stack.enter_context(patch.object(tm, "save_script_data"))
        self.audio = stack.enter_context(
            patch.object(tm, "generate_audio", return_value=("legacy.mp3", 5, "timing"))
        )
        self.subtitles = stack.enter_context(
            patch.object(tm, "generate_subtitle", return_value="legacy.srt")
        )
        self.materials = stack.enter_context(
            patch.object(tm, "get_video_materials", return_value=["clip.mp4"])
        )
        self.final = stack.enter_context(
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            )
        )
        self.adapter = stack.enter_context(
            patch.object(
                tm.narration_adapter,
                "prepare",
                return_value=NarrationArtifacts("multi.wav", 3.125, "multi.srt"),
            )
        )

    def enable_multi(self):
        self.params = VideoParams.model_validate(
            {
                **self.params.model_dump(),
                "narration": {
                    "narrators": [
                        {"id": "A", "display_name": "Host", "voice_name": "voice-A"},
                        {"id": "B", "display_name": "Guest", "voice_name": "voice-B"},
                    ],
                    "segments": [
                        {"speaker_id": "A", "text": "First. [pause: 1s]"},
                        {"speaker_id": "B", "text": "Second."},
                    ],
                },
            }
        )

    def test_single_uses_legacy_stages_and_progress(self):
        with patch.object(
            self.state, "update_task", wraps=self.state.update_task
        ) as update:
            tm.start("single", self.params)
        self.adapter.assert_not_called()
        self.audio.assert_called_once_with(
            "single",
            self.params,
            "Legacy script.",
            voice_preview=None,
            allow_server_file_input=False,
        )
        self.subtitles.assert_called_once_with(
            "single", self.params, "Legacy script.", "timing", "legacy.mp3"
        )
        self.final.assert_called_once_with(
            "single", self.params, ["clip.mp4"], "legacy.mp3", "legacy.srt", 5
        )
        progress = [call.kwargs.get("progress") for call in update.call_args_list]
        self.assertEqual(progress, [5, 10, 20, 30, 40, 50, 100])

    def test_multi_uses_adapter_artifacts_and_plain_script_everywhere(self):
        self.enable_multi()
        expected = tm.utils.remove_pause_tags(self.params.narration.flatten_script())
        with (
            patch.object(
                tm.upload_post.upload_post_service, "is_configured", return_value=True
            ),
            patch.object(
                type(tm.upload_post.upload_post_service),
                "auto_upload",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(
                type(tm.upload_post.upload_post_service),
                "platforms",
                new_callable=PropertyMock,
                return_value=["youtube"],
            ),
            patch.object(tm, "_schedule_cross_post", return_value=None) as post,
        ):
            tm.start("multi", self.params)
        self.adapter.assert_called_once_with("multi", self.params)
        self.audio.assert_not_called()
        self.subtitles.assert_not_called()
        self.llm.assert_not_called()
        self.assertEqual(self.params.video_script, expected)
        self.terms.assert_called_once_with("multi", self.params, expected)
        self.save.assert_called_once_with("multi", expected, ["coffee"], self.params)
        self.materials.assert_called_once_with(
            "multi", self.params, ["coffee"], 3.125, loomloom_video_request=None
        )
        self.final.assert_called_once_with(
            "multi", self.params, ["clip.mp4"], "multi.wav", "multi.srt", 3.125
        )
        self.assertEqual(post.call_args.kwargs["video_script"], expected)

    def test_adapter_failure_uses_existing_task_failure(self):
        self.enable_multi()
        self.adapter.side_effect = RuntimeError("segment failed")
        tm.start("failed", self.params)
        self.adapter.assert_called_once()
        self.audio.assert_not_called()
        self.subtitles.assert_not_called()
        self.materials.assert_not_called()
        self.final.assert_not_called()
        failure = self.state.get_task("failed")
        self.assertEqual(failure["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failure["failed_stage"], "pipeline")
        self.assertIn("segment failed", failure["error"])

    def test_incompatible_multi_options_fail_before_stages(self):
        cases = [
            ({"custom_audio_file": "uploaded.wav"}, {}, "custom audio"),
            ({"voice_name": tm.voice.NO_VOICE_NAME}, {}, "no-voice"),
            ({}, {"voice_preview": {}}, "voice previews"),
        ] + [
            ({}, {"stop_at": stage}, "full-video")
            for stage in ("script", "terms", "audio", "subtitle", "materials")
        ]
        for fields, arguments, message in cases:
            with self.subTest(fields=fields, arguments=arguments):
                self.enable_multi()
                params = self.params.model_copy(update=fields)
                tm.start("invalid", params, **arguments)
                failure = self.state.get_task("invalid")
                self.assertEqual(failure["failed_stage"], "preflight")
                self.assertIn(message, failure["error"])
                self.adapter.assert_not_called()
                self.audio.assert_not_called()
                self.terms.assert_not_called()
                self.final.assert_not_called()


if __name__ == "__main__":
    unittest.main()
