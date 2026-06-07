import unittest

import cv2
import numpy as np

from preprocessing import ImageVariant, generate_fallback_tile_variants
from qr_detector import QRDetector, map_points_to_original


class SmartDetectorTests(unittest.TestCase):
    def test_robust_mode_aliases_to_smart(self):
        detector = QRDetector(mode="robust")

        self.assertEqual(detector.mode, "smart")

    def test_fallback_tiles_cover_4k_frame_extremes(self):
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        tiles = list(generate_fallback_tile_variants(image, max_tiles=12))

        self.assertEqual(len(tiles), 12)
        self.assertTrue(any(tile.offset_x == 0 and tile.offset_y == 0 for tile in tiles))
        self.assertTrue(any(tile.offset_x + tile.image.shape[1] >= 3840 for tile in tiles))
        self.assertTrue(any(tile.offset_y + tile.image.shape[0] >= 2160 for tile in tiles))
        self.assertTrue(
            any(
                tile.offset_x <= 1920 <= tile.offset_x + tile.image.shape[1]
                and tile.offset_y <= 1080 <= tile.offset_y + tile.image.shape[0]
                for tile in tiles
            )
        )

    def test_scaled_locator_points_map_back_to_original_coordinates(self):
        points = np.array([[100, 50], [200, 50], [200, 150], [100, 150]], dtype="float32")
        variant = ImageVariant(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            name="locator",
            source="test",
            scale=0.5,
            offset_x=10,
            offset_y=20,
        )

        mapped = map_points_to_original(points, variant)

        np.testing.assert_allclose(
            mapped,
            np.array([[210, 120], [410, 120], [410, 320], [210, 320]], dtype="float32"),
        )

    def test_zxing_checksum_errors_are_recorded_as_near_misses(self):
        detector = QRDetector(mode="smart")
        detector._record_near_miss(
            source="zxing_candidate_crop",
            variant_name="candidate_1_crop_clahe",
            error="ChecksumError @ QRDecoder.cpp:360",
            points=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype="float32"),
        )

        self.assertEqual(len(detector.last_near_misses), 1)
        self.assertEqual(detector.last_near_misses[0]["reason"], "checksum_near_miss")


if __name__ == "__main__":
    unittest.main()
