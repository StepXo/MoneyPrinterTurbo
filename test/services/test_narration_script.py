import json
import unittest
from unittest.mock import patch

from app.models.narration import MultiNarration, Narrator
from app.services import llm, narration_script


class TestNarrationScript(unittest.TestCase):
    def setUp(self):
        self.narrators = [
            Narrator(id="host_id", display_name="Host", voice_name="private-voice-A"),
            Narrator(id="guest_id", display_name="Guest", voice_name="private-voice-B"),
        ]
        self.payload = {
            "segments": [
                {"speaker_id": "host_id", "text": "First."},
                {"speaker_id": "guest_id", "text": "Second."},
                {"speaker_id": "host_id", "text": "Third."},
            ]
        }
        self.response = json.dumps(self.payload)

    def test_valid_output_order_ids_flattening_and_definitions(self):
        originals = [item.model_dump() for item in self.narrators]
        with (
            patch.object(llm, "generate_text", return_value=self.response) as generate,
            patch.object(llm, "generate_script") as legacy,
        ):
            result = narration_script.generate("Coffee", self.narrators)
        self.assertIsInstance(result, MultiNarration)
        self.assertEqual(
            [segment.model_dump() for segment in result.segments],
            self.payload["segments"],
        )
        self.assertEqual(result.flatten_script(), "First. Second. Third.")
        self.assertEqual([item.model_dump() for item in result.narrators], originals)
        self.assertEqual([item.model_dump() for item in self.narrators], originals)
        generate.assert_called_once()
        legacy.assert_not_called()

    def test_n_narrators(self):
        self.narrators.append(
            Narrator(id="third_id", display_name="Third", voice_name="private-C")
        )
        self.payload["segments"][2]["speaker_id"] = "third_id"
        with patch.object(llm, "generate_text", return_value=json.dumps(self.payload)):
            result = narration_script.generate("Coffee", self.narrators)
        self.assertEqual(len(result.narrators), 3)
        self.assertEqual(result.segments[2].speaker_id, "third_id")

    def test_prompt_reuses_custom_instructions_and_config_without_voices(self):
        snapshot = {"llm_provider": "ollama", "api_key": "secret-not-for-prompt"}
        with patch.object(llm, "generate_text", return_value=self.response) as generate:
            narration_script.generate(
                "Coffee facts",
                self.narrators,
                language="es",
                paragraph_number=3,
                video_script_prompt="Under 100 words.",
                custom_system_prompt="Write warmly.",
                app_config=snapshot,
            )
        prompt = generate.call_args.kwargs["prompt"]
        for text in (
            "Coffee facts",
            "language: es",
            "number of paragraphs: 3",
            "Under 100 words.",
            "Write warmly.",
            "host_id",
            "guest_id",
            "Host",
            "Guest",
        ):
            self.assertIn(text, prompt)
        for text in ("private-voice-A", "private-voice-B", "secret-not-for-prompt"):
            self.assertNotIn(text, prompt)
        self.assertIs(generate.call_args.kwargs["app_config"], snapshot)

    def test_invalid_output_retries_once_then_succeeds_or_fails(self):
        invalid = [
            "not JSON",
            "",
            "[]",
            "null",
            "{}",
            '{"segments":{}}',
            '{"segments":[]}',
            '{"segments":[{}]}',
            '{"segments":[{"speaker_id":"unknown","text":"Hi"}]}',
            '{"segments":[{"speaker_id":"Host","text":"Hi"}]}',
            '{"segments":[{"speaker_id":"host_id","text":"   "}]}',
            '{"segments":[{"speaker_id":"host_id","text":4}]}',
            '{"segments":["Hi"]}',
            "Here is JSON: " + self.response,
            self.response + " Commentary",
            "```json\n" + self.response + "\n```\nCommentary",
            '{"segments":[],"narrators":[]}',
        ]
        snapshot = {"llm_provider": "ollama"}
        for first in invalid:
            for recover in (True, False):
                with self.subTest(first=first, recover=recover):
                    with patch.object(
                        llm,
                        "generate_text",
                        side_effect=[first, self.response if recover else first],
                    ) as generate:
                        if recover:
                            result = narration_script.generate(
                                "Coffee", self.narrators, app_config=snapshot
                            )
                            self.assertEqual(
                                result.flatten_script(), "First. Second. Third."
                            )
                        else:
                            with self.assertRaisesRegex(
                                ValueError,
                                "multi-narrator structured script generation failed",
                            ):
                                narration_script.generate(
                                    "Coffee", self.narrators, app_config=snapshot
                                )
                    self.assertEqual(generate.call_count, 2)
                    self.assertIn(
                        "previous response was invalid",
                        generate.call_args.kwargs["prompt"],
                    )
                    self.assertTrue(
                        all(
                            call.kwargs["app_config"] is snapshot
                            for call in generate.call_args_list
                        )
                    )

    def test_one_outer_json_fence_accepted(self):
        with patch.object(
            llm,
            "generate_text",
            return_value=" \n```json\n" + self.response + "\n```\n ",
        ) as generate:
            result = narration_script.generate("Coffee", self.narrators)
        self.assertEqual(result.flatten_script(), "First. Second. Third.")
        generate.assert_called_once()

    def test_invalid_narrator_input_fails_before_llm(self):
        for narrators in ([], [self.narrators[0], self.narrators[0]]):
            with (
                self.subTest(narrators=narrators),
                patch.object(llm, "generate_text") as generate,
            ):
                with self.assertRaises(ValueError):
                    narration_script.generate("Coffee", narrators)
                generate.assert_not_called()

    def test_provider_error_does_not_echo_credentials_or_retry(self):
        with patch.object(
            llm, "generate_text", return_value="Error: secret-credential"
        ) as generate:
            with self.assertRaisesRegex(RuntimeError, "LLM provider error") as caught:
                narration_script.generate("Coffee", self.narrators)
        self.assertNotIn("secret-credential", str(caught.exception))
        generate.assert_called_once()


class TestRawTextBoundary(unittest.TestCase):
    def test_raw_text_and_configuration_pass_through(self):
        raw = '  {"segments":[{"speaker_id":"A","text":"[brackets] (kept)"}]}\n'
        for snapshot in (None, {"llm_provider": "ollama"}):
            with (
                self.subTest(snapshot=snapshot),
                patch.object(llm, "_generate_response", return_value=raw) as response,
            ):
                self.assertEqual(llm.generate_text("prompt", app_config=snapshot), raw)
                response.assert_called_once_with(prompt="prompt", app_config=snapshot)

    def test_raw_boundary_preserves_error_behavior(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: provider unavailable"
        ):
            self.assertEqual(llm.generate_text("prompt"), "Error: provider unavailable")
        with patch.object(
            llm, "_generate_response", side_effect=RuntimeError("failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                llm.generate_text("prompt")

    def test_legacy_script_cleanup_unchanged(self):
        with patch.object(
            llm, "_generate_response", return_value="[scene]Hello (aside)world."
        ):
            self.assertEqual(llm.generate_script("Coffee"), "Hello world.")


if __name__ == "__main__":
    unittest.main()
