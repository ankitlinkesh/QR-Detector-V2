import unittest
from pathlib import Path

import cv2
import numpy as np

from frame_recovery import FrameRanker, generate_recovery_variants, decode_recovery_variants, prepare_recovery_image


class FrameRecoveryTests(unittest.TestCase):
    def test_grayscale_clahe_sharpen_decodes_qr_det5_frame_94(self):
        video_path = Path("inputs/videos/QR-DET5.mp4")
        if not video_path.exists():
            self.skipTest("QR-DET5 fixture is not available")

        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 94)
        ok, frame = capture.read()
        capture.release()
        self.assertTrue(ok)

        results = decode_recovery_variants(frame)

        self.assertIn("(01)96908453915728", [result["text"] for result in results])
        winning = next(result for result in results if result["text"] == "(01)96908453915728")
        self.assertEqual(winning["variant"], "grayscale_clahe_sharpen")

    def test_frame_ranker_prefers_sharper_frame(self):
        ranker = FrameRanker()
        sharp = np.zeros((180, 180, 3), dtype=np.uint8)
        cv2.rectangle(sharp, (30, 30), (150, 150), (255, 255, 255), 4)
        cv2.line(sharp, (30, 90), (150, 90), (255, 255, 255), 2)
        blurry = cv2.GaussianBlur(sharp, (31, 31), 0)

        ranked = ranker.rank([(1, blurry), (2, sharp)], top_n=2)

        self.assertEqual(ranked[0].frame_index, 2)
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_recovery_variants_include_borrowed_combo(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        names = [variant.name for variant in generate_recovery_variants(image)]

        self.assertIn("grayscale_clahe_sharpen", names)

    def test_recovery_variants_include_full_borrowed_pipeline(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        names = [variant.name for variant in generate_recovery_variants(image)]

        self.assertIn("histogram_equalization", names)
        self.assertIn("rotation_45", names)
        self.assertIn("scale_4x", names)
        self.assertIn("denoise_clahe_adaptive", names)
        self.assertIn("brightness_contrast_sharpen", names)

    def test_recovery_variant_limit(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        results = decode_recovery_variants(image, max_variants=1)

        self.assertEqual(results, [])

    def test_prepare_recovery_image_downscales_large_frame(self):
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        resized, preserves_geometry, scale = prepare_recovery_image(image, max_side=1600)

        self.assertEqual(resized.shape[:2], (900, 1600))
        self.assertFalse(preserves_geometry)
        self.assertAlmostEqual(scale, 1600 / 3840)


if __name__ == "__main__":
    unittest.main()


