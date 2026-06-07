import unittest

import cv2
import numpy as np

from preprocessing import generate_crop_variants, make_white_background, add_quiet_zone, inner_square_crop_variants


class ProcessingBackgroundTests(unittest.TestCase):
    def test_white_background_variant_keeps_dark_modules_and_white_border(self):
        gray = np.full((40, 40), 210, dtype=np.uint8)
        cv2.rectangle(gray, (12, 12), (27, 27), 35, -1)

        cleaned = make_white_background(gray)
        padded = add_quiet_zone(cleaned, min_border=8)

        self.assertLess(cleaned[20, 20], 80)
        self.assertGreater(cleaned[0, 0], 240)
        self.assertEqual(padded.shape[:2], (56, 56))
        self.assertGreater(padded[2, 2], 240)


    def test_white_background_removes_dark_border_background(self):
        gray = np.full((60, 60), 220, dtype=np.uint8)
        cv2.rectangle(gray, (0, 0), (59, 59), 20, 4)
        cv2.rectangle(gray, (24, 24), (36, 36), 20, -1)

        cleaned = make_white_background(gray)

        self.assertGreater(cleaned[1, 1], 240)
        self.assertLess(cleaned[30, 30], 80)
    def test_crop_variants_include_background_cleanup_and_quiet_zone(self):
        crop = np.full((80, 120, 3), 230, dtype=np.uint8)
        cv2.rectangle(crop, (35, 20), (75, 60), (20, 20, 20), 2)

        names = [variant.name for variant in generate_crop_variants(crop, prefix="candidate_1_crop")]

        self.assertIn("candidate_1_crop_white_bg_otsu_quiet", names)
        self.assertIn("candidate_1_crop_adaptive_white_quiet", names)
        self.assertIn("candidate_1_crop_clahe_white_quiet", names)

    def test_inner_square_crop_variants_focus_square_inside_noisy_crop(self):
        crop = np.full((120, 180, 3), 230, dtype=np.uint8)
        cv2.rectangle(crop, (20, 55), (160, 65), (50, 50, 50), -1)
        cv2.rectangle(crop, (65, 35), (105, 75), (20, 20, 20), 2)
        cv2.line(crop, (65, 55), (105, 55), (20, 20, 20), 1)
        cv2.line(crop, (85, 35), (85, 75), (20, 20, 20), 1)

        variants = list(inner_square_crop_variants(crop, prefix="candidate_1_crop"))

        self.assertTrue(variants)
        self.assertTrue(any("inner_square" in variant.name for variant in variants))
        self.assertTrue(any(variant.image.shape[0] < crop.shape[0] for variant in variants))


if __name__ == "__main__":
    unittest.main()


