from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("atlas_image.py")
SPEC = importlib.util.spec_from_file_location("atlas_image", MODULE_PATH)
assert SPEC and SPEC.loader
atlas_image = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atlas_image)


class AtlasImageTest(unittest.TestCase):
    def test_filters_visible_image_models(self) -> None:
        models = [
            {"model": "visible", "type": "Image", "display_console": True},
            {"model": "hidden", "type": "Image", "display_console": False},
            {"model": "video", "type": "Video", "display_console": True},
        ]
        self.assertEqual(["visible"], [m["model"] for m in atlas_image.visible_image_models(models)])

    def test_build_payload_uses_only_schema_supported_options(self) -> None:
        args = argparse.Namespace(
            model="provider/model",
            prompt="A launch image",
            size="1024x1024",
            quality="high",
            output_format="png",
        )
        schema = {
            "properties": {
                "prompt": {"type": "string"},
                "size": {"enum": ["1024x1024"]},
                "quality": {"enum": ["medium", "high"]},
                "output_format": {"enum": ["jpeg", "png"]},
                "enable_sync_mode": {"type": "boolean"},
            }
        }
        self.assertEqual(
            {
                "model": "provider/model",
                "prompt": "A launch image",
                "size": "1024x1024",
                "quality": "high",
                "output_format": "png",
                "enable_sync_mode": False,
            },
            atlas_image.build_payload(args, schema),
        )

    def test_submit_generation_makes_one_post(self) -> None:
        with patch.object(
            atlas_image,
            "request_json",
            return_value={"code": 200, "data": {"id": "prediction-1", "status": "starting"}},
        ) as request:
            result = atlas_image.submit_generation("https://example.test", "secret", {"prompt": "x"})
        self.assertEqual("prediction-1", result["id"])
        self.assertEqual(1, request.call_count)
        self.assertEqual("POST", request.call_args.args[0])

    def test_polling_is_bounded_and_returns_completed_prediction(self) -> None:
        responses = [
            {"code": 200, "data": {"id": "prediction-1", "status": "processing"}},
            {
                "code": 200,
                "data": {
                    "id": "prediction-1",
                    "status": "completed",
                    "outputs": ["https://cdn.example.test/image.png"],
                },
            },
        ]
        with (
            patch.object(atlas_image, "request_json", side_effect=responses) as request,
            patch.object(atlas_image.time, "sleep"),
        ):
            result = atlas_image.poll_generation(
                "https://example.test", "secret", "prediction-1", attempts=3, interval=0
            )
        self.assertEqual("completed", result["status"])
        self.assertEqual(2, request.call_count)


if __name__ == "__main__":
    unittest.main()
