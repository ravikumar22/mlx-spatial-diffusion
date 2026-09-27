"""Unit tests for spatial region masking and Gaussian feathering."""

import unittest
import mlx.core as mx
from mlx_spatial_diffusion.core.mask import Region, RegionMaskGenerator, build_gaussian_kernel_2d


class TestRegionMask(unittest.TestCase):
    def test_region_validation(self):
        # Valid region
        r = Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Left half")
        self.assertEqual(r.box, [0.0, 0.0, 1.0, 0.5])

        # Invalid bounds
        with self.assertRaises(ValueError):
            Region(box=[0.5, 0.0, 0.2, 0.5], prompt="Inverted y")

        with self.assertRaises(ValueError):
            Region(box=[0.0, 0.0, 1.2, 0.5], prompt="Out of bounds")

        with self.assertRaises(ValueError):
            Region(box=[0.0, 0.0, 1.0], prompt="Not 4 elements")

    def test_gaussian_kernel(self):
        kernel = build_gaussian_kernel_2d(radius=3)
        self.assertEqual(kernel.shape, (1, 7, 7, 1))
        # Total sum of kernel should be 1.0
        total_sum = mx.sum(kernel).item()
        self.assertAlmostEqual(total_sum, 1.0, places=5)

    def test_binary_mask_generation(self):
        gen = RegionMaskGenerator(latent_height=32, latent_width=32)
        # Left half
        region = Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Left half", feather_radius=0)
        mask = gen.generate_binary_mask(region)
        self.assertEqual(mask.shape, (1, 32, 32, 1))

        # Check left side is 1.0 and right side is 0.0
        left_val = mask[0, 16, 8, 0].item()
        right_val = mask[0, 16, 24, 0].item()
        self.assertEqual(left_val, 1.0)
        self.assertEqual(right_val, 0.0)

    def test_feathered_mask_smoothness(self):
        gen = RegionMaskGenerator(latent_height=32, latent_width=32)
        region = Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Left half", feather_radius=4)
        feathered = gen.generate_feathered_mask(region)

        # Center of left half should be close to 1.0
        center_left = feathered[0, 16, 8, 0].item()
        self.assertAlmostEqual(center_left, 1.0, delta=0.05)

        # Far right should be 0.0
        far_right = feathered[0, 16, 28, 0].item()
        self.assertAlmostEqual(far_right, 0.0, delta=0.01)

        # Boundary (around x=16) should have intermediate value between 0 and 1
        boundary_val = feathered[0, 16, 16, 0].item()
        self.assertTrue(0.1 < boundary_val < 0.9)


if __name__ == "__main__":
    unittest.main()
