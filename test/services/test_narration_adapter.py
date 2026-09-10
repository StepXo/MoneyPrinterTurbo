import struct
import subprocess
import tempfile
import unittest
import wave
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from edge_tts import SubMaker
from edge_tts.srt_composer import Subtitle

from app.models.schema import VideoParams
from app.services import narration_adapter as adapter


class TestNarrationAdapter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "unrelated.txt").write_text("keep")
        self.params = VideoParams(
            video_subject="test",
            narration={
                "narrators": [
                    {
                        "id": name,
                        "display_name": f"Speaker {name}",
                        "voice_name": f"voice_{name}",
                    }
                    for name in ("A", "B", "C")
                ],
                "segments": [
                    {"speaker_id": "A", "text": "First."},
                    {"speaker_id": "B", "text": "Second."},
                    {"speaker_id": "A", "text": "Third."},
                ],
            },
        )
        self.calls = []
        self.addCleanup(patch.stopall)
        patch.object(adapter.utils, "task_dir", return_value=str(self.root)).start()
        patch.dict(adapter.config.app, {"subtitle_provider": "edge"}).start()
        self.tts = patch.object(
            adapter.voice, "tts", side_effect=self.synthesize
        ).start()

    def synthesize(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        # Mixed sample rates and channel counts, with distinct constant samples.
        rate = (24000, 48000, 16000)[index % 3]
        channels = 2 if index == 1 else 1
        with wave.open(kwargs["voice_file"], "wb") as audio:
            audio.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
            audio.writeframes(struct.pack("<h", (index + 1) * 1000) * rate * channels)
        if index == 1:
            # A different source format too, regardless of provider file suffix.
            flac = kwargs["voice_file"] + ".flac"
            subprocess.run(
                [
                    adapter.utils.get_ffmpeg_binary(),
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    kwargs["voice_file"],
                    flac,
                ],
                check=True,
                capture_output=True,
            )
            Path(flac).replace(kwargs["voice_file"])
        text = adapter.utils.remove_pause_tags(kwargs["text"])
        if index == 1:
            return SimpleNamespace(subs=[text], offset=[(0, 8_000_000)])
        maker = SubMaker()
        maker.cues.append(
            Subtitle(
                index=1, start=timedelta(0), end=timedelta(seconds=0.8), content=text
            )
        )
        return maker

    def prepare(self):
        return adapter.prepare("test-task", self.params)

    def assert_clean_failure(self):
        self.assertEqual(list(self.root.iterdir()), [self.root / "unrelated.txt"])

    def test_order_voices_pcm_duration_subtitles_and_cleanup(self):
        original = self.params.model_dump()
        result = self.prepare()
        self.assertEqual(
            [call["voice_name"] for call in self.calls],
            ["voice_A", "voice_B", "voice_A"],
        )
        self.assertEqual(
            [call["text"] for call in self.calls], ["First.", "Second.", "Third."]
        )
        self.assertEqual(self.tts.call_count, 3)
        self.assertEqual(len({call["voice_file"] for call in self.calls}), 3)
        with wave.open(result.audio_file, "rb") as master:
            self.assertEqual(
                (master.getnchannels(), master.getsampwidth(), master.getframerate()),
                (1, 2, 24000),
            )
            self.assertEqual(master.getnframes(), 72000)
            for index in range(3):
                master.setpos(index * 24000 + 12000)
                self.assertAlmostEqual(
                    struct.unpack("<h", master.readframes(1))[0],
                    (index + 1) * 1000,
                    delta=2,
                )
        self.assertEqual(result.audio_duration, 3)
        cues = adapter._read_cues(Path(result.subtitle_path), 0, 72000)
        self.assertEqual(
            [(start, end) for start, end, _ in cues],
            [(0, 800), (1000, 1800), (2000, 2800)],
        )
        self.assertEqual(
            [text.rstrip(".") for _, _, text in cues], ["First", "Second", "Third"]
        )
        self.assertEqual(self.params.model_dump(), original)
        self.assertEqual(
            {p.name for p in Path(result.audio_file).parent.iterdir()},
            {"audio.wav", "subtitle.srt"},
        )
        self.assertTrue(
            all(not Path(call["voice_file"]).exists() for call in self.calls)
        )

    def test_more_than_two_narrators(self):
        self.params.narration.segments[2].speaker_id = "C"
        self.prepare()
        self.assertEqual(
            [call["voice_name"] for call in self.calls],
            ["voice_A", "voice_B", "voice_C"],
        )

    def test_failed_middle_segment_never_skipped(self):
        for failure in ("none", "missing", "corrupt", "exception"):
            with self.subTest(failure=failure):
                self.calls.clear()

                def fail(**kwargs):
                    if self.calls:
                        if failure == "exception":
                            raise OSError("provider unavailable")
                        if failure == "corrupt":
                            Path(kwargs["voice_file"]).write_bytes(b"not audio")
                        return None if failure == "none" else SubMaker()
                    return self.synthesize(**kwargs)

                with patch.object(adapter.voice, "tts", side_effect=fail) as tts:
                    with self.assertRaises(Exception):
                        self.prepare()
                    self.assertEqual(tts.call_count, 2)
                self.assert_clean_failure()

    def test_subtitle_failure_removes_audio(self):
        with patch.object(adapter.voice, "create_subtitle", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "no file"):
                self.prepare()
        self.assert_clean_failure()

    def test_normalization_failure_preserves_exception(self):
        error = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"decode failed")
        with patch.object(adapter.subprocess, "run", side_effect=error):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                self.prepare()
        self.assertIs(caught.exception, error)
        self.assert_clean_failure()

    def test_final_write_failure_removes_partial_artifacts(self):
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.prepare()
        self.assert_clean_failure()

    def test_whisper_uses_complete_audio_and_flattened_correction(self):
        self.params.narration.segments[0].text = "First. [pause: 1s]"

        def transcribe(**kwargs):
            with wave.open(kwargs["audio_file"], "rb") as master:
                self.assertEqual(master.getnframes(), 72000)
            Path(kwargs["subtitle_file"]).write_text(
                "1\n00:00:00,000 --> 00:00:03,001\nFirst. Second. Third.\n\n"
            )

        with (
            patch.dict(adapter.config.app, {"subtitle_provider": "whisper"}),
            patch.object(adapter.subtitle, "create", side_effect=transcribe) as create,
            patch.object(adapter.subtitle, "correct") as correct,
            patch.object(adapter.voice, "create_subtitle") as provider,
        ):
            result = self.prepare()
        create.assert_called_once()
        provider.assert_not_called()
        self.assertEqual(
            correct.call_args.kwargs["video_script"],
            adapter.utils.remove_pause_tags(self.params.narration.flatten_script()),
        )
        self.assertIn("00:00:03,000", Path(result.subtitle_path).read_text())
        self.assertEqual(self.calls[0]["text"], "First. [pause: 1s]")

    def test_word_level_whisper_does_not_correct(self):
        self.params.subtitle_display_mode = "word_by_word"

        def transcribe(**kwargs):
            self.assertTrue(kwargs["word_level"])
            Path(kwargs["subtitle_file"]).write_text(
                "1\n00:00:00,000 --> 00:00:00,500\nFirst\n\n"
            )

        with (
            patch.dict(adapter.config.app, {"subtitle_provider": "whisper"}),
            patch.object(adapter.subtitle, "create", side_effect=transcribe),
            patch.object(adapter.subtitle, "correct") as correct,
        ):
            self.prepare()
        correct.assert_not_called()

    def test_disabled_subtitles(self):
        self.params.subtitle_enabled = False
        with patch.object(adapter.voice, "create_subtitle") as create:
            result = self.prepare()
        create.assert_not_called()
        self.assertEqual(result.subtitle_path, "")
        self.assertTrue(Path(result.audio_file).is_file())

    def test_mutated_unknown_speaker_fails_before_tts(self):
        self.params.narration.segments[0].speaker_id = "missing"
        with self.assertRaises(ValueError):
            self.prepare()
        self.tts.assert_not_called()
        self.assert_clean_failure()

    def test_master_assembly_failure_cleans_partial_files(self):
        original = wave.Wave_write.writeframesraw

        def fail_master(writer, data):
            if str(writer._file.name).endswith("audio.wav"):
                raise OSError("master write failed")
            return original(writer, data)

        with patch.object(wave.Wave_write, "writeframesraw", fail_master):
            with self.assertRaisesRegex(OSError, "master write failed"):
                self.prepare()
        self.assert_clean_failure()

    def test_invalid_provider_subtitle_fails(self):
        def invalid(**kwargs):
            Path(kwargs["subtitle_file"]).write_text("not an SRT file")

        with patch.object(adapter.voice, "create_subtitle", side_effect=invalid):
            with self.assertRaisesRegex(ValueError, "invalid subtitle block"):
                self.prepare()
        self.assert_clean_failure()

    def test_whisper_missing_output_fails(self):
        with (
            patch.dict(adapter.config.app, {"subtitle_provider": "whisper"}),
            patch.object(adapter.subtitle, "create", return_value=None),
            patch.object(adapter.subtitle, "correct"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no file"):
                self.prepare()
        self.assert_clean_failure()

    def test_provider_word_subtitles_and_pause_text(self):
        self.params.subtitle_display_mode = "word_by_word"
        self.params.narration.segments[0].text = "First. [pause: 1s]"
        with patch.object(
            adapter.voice, "create_subtitle", wraps=adapter.voice.create_subtitle
        ) as create:
            result = self.prepare()
        self.assertTrue(
            all(call.kwargs["word_level"] for call in create.call_args_list)
        )
        self.assertEqual(self.calls[0]["text"], "First. [pause: 1s]")
        self.assertNotIn("pause", Path(result.subtitle_path).read_text())

    def test_subtitle_rounding_clamps_overlap_and_tail(self):
        source = self.root / "timing.srt"
        source.write_text(
            "1\n00:00:00,000 --> 00:00:00,501\nOne\n\n"
            "2\n00:00:00,500 --> 00:00:01,001\nTwo\n\n"
        )
        self.assertEqual(
            adapter._read_cues(source, 24000, 48000),
            [(1000, 1501, "One"), (1501, 2000, "Two")],
        )


if __name__ == "__main__":
    unittest.main()
