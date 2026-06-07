import unittest

import cv2
import numpy as np

from output_writer import draw_annotations
from preprocessing import ImageVariant
from qr_detector import QRDetector, QRCandidate, DetectionProfile, candidates_to_dicts, merge_candidate_duplicates


class CandidateReportTests(unittest.TestCase):
    def test_raw_candidates_are_drawn_separately_from_decode_candidates(self):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        raw_candidate = {
            "points": [[10, 80], [35, 80], [35, 105], [10, 105]],
            "center": [22.5, 92.5],
            "source": "opencv_raw_test",
            "score": 100.0,
        }
        selected_candidate = {
            "points": [[70, 80], [100, 80], [100, 110], [70, 110]],
            "center": [85.0, 95.0],
            "source": "selected_test",
            "score": 200.0,
        }

        annotated = draw_annotations(
            image,
            [],
            "candidate report",
            False,
            candidates=[selected_candidate],
            raw_candidates=[raw_candidate],
            failure_reason="candidate_no_decode",
        )

        self.assertTrue(np.any(annotated[80, 10:36] != 0))
        self.assertTrue(np.any(annotated[80, 70:101] != 0))

    def test_raw_candidates_preserve_tiny_candidates_before_merge_cap(self):
        detector = QRDetector(mode="smart", max_candidates=1)
        tiny = QRCandidate(
            points=np.array([[10, 10], [28, 10], [28, 28], [10, 28]], dtype="float32"),
            source="contour_tiny_test",
            score=1.0,
        )
        large = QRCandidate(
            points=np.array([[40, 40], [100, 40], [100, 100], [40, 100]], dtype="float32"),
            source="opencv_large_test",
            score=100.0,
        )

        detector.last_raw_candidates = candidates_to_dicts([tiny, large])
        selected = merge_candidate_duplicates([tiny, large], max_candidates=1)

        self.assertEqual(len(selected), 1)
        self.assertEqual(len(detector.last_raw_candidates), 2)
        self.assertEqual(detector.last_raw_candidates[0]["width"], 18.0)
        self.assertEqual(detector.last_raw_candidates[0]["height"], 18.0)

    def test_large_frames_keep_full_resolution_locator_variants(self):
        detector = QRDetector(mode="smart")
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)

        names = [variant.name for variant in detector._generate_locator_variants(image)]

        self.assertIn("locator_original", names)
        self.assertIn("locator_fullres_gray", names)
        self.assertIn("locator_fullres_adaptive", names)

    def test_tiny_square_candidate_survives_contour_filtering(self):
        detector = QRDetector(mode="smart")
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (63, 63), (255, 255, 255), 1)
        variant = ImageVariant(image=image, name="locator_original", source="locator_original", scale=1.0)

        candidates = detector._find_contour_candidates(variant, DetectionProfile())

        self.assertTrue(any(candidate.source.startswith("contour_quad") for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
