import unittest

from pydantic import ValidationError

from app.models.narration import MultiNarration, NarrationSegment, Narrator
from app.models.schema import VideoParams


def narration_payload(count=2):
    return {
        "narrators": [
            {
                "id": f"narrator_{i}",
                "display_name": f"Speaker {i}",
                "voice_name": f"voice_{i}",
            }
            for i in range(1, count + 1)
        ],
        "segments": [
            {"speaker_id": "narrator_1", "text": "First line."},
            {"speaker_id": f"narrator_{count}", "text": "Second line."},
            {"speaker_id": "narrator_1", "text": "Third line."},
        ],
    }


class TestNarration(unittest.TestCase):
    def test_accepts_one_two_and_more_narrators_preserving_order(self):
        for count in (1, 2, 4):
            with self.subTest(count=count):
                payload = narration_payload(count)
                narration = MultiNarration.model_validate(payload)
                self.assertEqual(narration.model_dump(), payload)
                self.assertEqual(
                    narration.flatten_script(), "First line. Second line. Third line."
                )
                self.assertEqual(narration.flatten_script(), narration.flatten_script())

    def test_rejects_blank_fields(self):
        cases = (
            (Narrator, {"id": "n1", "display_name": "Host", "voice_name": "voice"}),
            (NarrationSegment, {"speaker_id": "n1", "text": "Hello"}),
        )
        for model, fields in cases:
            for field in fields:
                for value in ("", " \t\n "):
                    with self.subTest(model=model.__name__, field=field, value=value):
                        with self.assertRaises(ValidationError):
                            model.model_validate({**fields, field: value})

    def test_trims_fields_without_deriving_ids_or_changing_internal_text(self):
        narrator = Narrator(id=" n1 ", display_name=" Host ", voice_name=" voice ")
        segment = NarrationSegment(speaker_id=" n1 ", text=" Hello\nworld. ")
        narration = MultiNarration(narrators=[narrator], segments=[segment])
        self.assertEqual(
            narrator.model_dump(),
            {"id": "n1", "display_name": "Host", "voice_name": "voice"},
        )
        narrator.display_name = "Renamed"
        self.assertEqual(narrator.id, "n1")
        self.assertEqual(segment.speaker_id, "n1")
        self.assertEqual(narration.flatten_script(), "Hello\nworld.")
        with self.assertRaises(ValidationError):
            Narrator(display_name="Host", voice_name="voice")

    def test_rejects_duplicate_ids_after_trimming(self):
        payload = narration_payload()
        payload["narrators"][1]["id"] = " narrator_1 "
        with self.assertRaisesRegex(ValidationError, "narrator IDs must be unique"):
            MultiNarration.model_validate(payload)

    def test_rejects_unknown_speaker(self):
        payload = narration_payload()
        payload["segments"][1]["speaker_id"] = "missing"
        with self.assertRaisesRegex(ValidationError, "unknown speaker ID: missing"):
            MultiNarration.model_validate(payload)

    def test_rejects_empty_collections(self):
        for field in ("narrators", "segments"):
            with self.subTest(field=field):
                payload = narration_payload()
                payload[field] = []
                with self.assertRaises(ValidationError):
                    MultiNarration.model_validate(payload)

    def test_flattening_leaves_pause_processing_to_integration(self):
        payload = narration_payload()
        payload["segments"][0]["text"] = "First [pause: 1s] line."
        self.assertEqual(
            MultiNarration.model_validate(payload).flatten_script(),
            "First [pause: 1s] line. Second line. Third line.",
        )

    def test_legacy_video_params_default_and_explicit_none(self):
        payload = {
            "video_subject": "Coffee",
            "video_script": " Original script. ",
            "voice_name": "original",
        }
        for extra in ({}, {"narration": None}):
            with self.subTest(extra=extra):
                params = VideoParams.model_validate({**payload, **extra})
                self.assertIsNone(params.narration)
                self.assertEqual(params.model_dump(include=set(payload)), payload)
                self.assertEqual(
                    VideoParams.model_validate_json(params.model_dump_json()), params
                )

    def test_video_params_round_trip_preserves_all_narration_data(self):
        payload = narration_payload(4)
        params = VideoParams(video_subject="Coffee", narration=payload)
        self.assertIsInstance(params.narration, MultiNarration)
        self.assertEqual(params.video_script, "")
        for restored in (
            VideoParams.model_validate(params.model_dump(mode="json")),
            VideoParams.model_validate_json(params.model_dump_json()),
        ):
            self.assertEqual(restored.narration.model_dump(), payload)
            self.assertEqual(restored, params)


if __name__ == "__main__":
    unittest.main()
