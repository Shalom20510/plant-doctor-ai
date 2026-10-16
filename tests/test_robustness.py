"""Smoke tests for the synthetic robustness image transformations.

Run with: py -m unittest discover -s tests -v
"""
import unittest
from PIL import Image, ImageChops

from src.robustness import variants


class RobustnessVariantTests(unittest.TestCase):
    def setUp(self):
        # A deterministic RGB test image; no external dataset is needed.
        self.image = Image.new("RGB", (224, 224), (90, 150, 70))

    def test_all_expected_conditions_are_returned(self):
        result = variants(self.image, seed=42)
        self.assertEqual(
            set(result),
            {
                "original",
                "brightness_dim_0.55",
                "brightness_bright_1.45",
                "rotation_15_degrees",
                "rotation_30_degrees",
                "gaussian_blur_1.5",
                "synthetic_edge_clutter",
            },
        )

    def test_variants_keep_image_dimensions_and_rgb_mode(self):
        for name, transformed in variants(self.image, seed=42).items():
            with self.subTest(condition=name):
                self.assertEqual(transformed.size, self.image.size)
                self.assertEqual(transformed.mode, "RGB")

    def test_clutter_is_reproducible_for_same_seed(self):
        first = variants(self.image, seed=7)["synthetic_edge_clutter"]
        second = variants(self.image, seed=7)["synthetic_edge_clutter"]
        self.assertIsNone(ImageChops.difference(first, second).getbbox())

    def test_brightness_variants_are_distinct(self):
        result = variants(self.image, seed=42)
        self.assertIsNotNone(
            ImageChops.difference(
                result["brightness_dim_0.55"], result["brightness_bright_1.45"]
            ).getbbox()
        )


if __name__ == "__main__":
    unittest.main()
