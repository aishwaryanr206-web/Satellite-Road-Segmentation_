import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app as road_app


class AppResultPreviewTests(unittest.TestCase):
    def test_upload_renders_and_serves_input_and_prediction_previews(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            uploads = root / "uploads"
            outputs = root / "outputs"
            uploads.mkdir()
            outputs.mkdir()

            rgb = np.zeros((64, 80, 3), dtype=np.uint8)
            rgb[:, :, 0] = 220
            rgb[:, :, 1] = np.arange(80, dtype=np.uint8)[None, :]
            encoded, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.assertTrue(encoded)

            mask = np.zeros((64, 80), dtype=np.uint8)
            mask[20:45, 25:55] = 255
            with (
                patch.object(road_app, "UPLOAD", uploads),
                patch.object(road_app, "OUTPUT", outputs),
                patch.object(road_app, "predict", return_value=mask),
            ):
                response = road_app.app.test_client().post(
                    "/",
                    data={"image": (io.BytesIO(buffer.tobytes()), "satellite.png")},
                    content_type="multipart/form-data",
                )

                self.assertEqual(response.status_code, 200)
                page = response.get_data(as_text=True)
                self.assertIn("Image results", page)
                self.assertIn("Road prediction overlay", page)
                job = re.search(r"Job ID: ([0-9a-f]{10})", page).group(1)
                input_url = re.search(r'src="([^"]*input_preview\.png)"', page).group(1)
                overlay_url = re.search(
                    r'src="([^"]*prediction_overlay\.png)"', page
                ).group(1)

                for image_url in (input_url, overlay_url):
                    preview = road_app.app.test_client().get(image_url)
                    try:
                        self.assertEqual(preview.status_code, 200)
                        self.assertTrue(preview.mimetype.startswith("image/"))
                        image = cv2.imdecode(
                            np.frombuffer(preview.get_data(), dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        self.assertEqual(image.shape[:2], (64, 80))
                    finally:
                        preview.close()

                forbidden_preview = road_app.app.test_client().get(
                    f"/preview/{job}/statistics.json"
                )
                self.assertEqual(forbidden_preview.status_code, 404)
                self.assertFalse(any(uploads.iterdir()))


if __name__ == "__main__":
    unittest.main()
