"""Unit tests for feathered latent blending."""

import unittest
import mlx.core as mx
from mlx_spatial_diffusion.core.mask import Region, RegionMaskGenerator
from mlx_spatial_diffusion.core.blending import FeatheredBlender


class TestFeatheredBlender(unittest.TestCase):
    def setUp(self):
        self.blender = FeatheredBlender()
        self.gen = RegionMaskGenerator(latent_height=16, latent_width=16)

    def test_two_halves_blending(self):
        # Region 1: Left half, prediction = 10.0
        r1 = Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Left", feather_radius=2)
        m1 = self.gen.generate_feathered_mask(r1)
        pred1 = mx.ones((1, 16, 16, 4)) * 10.0

        # Region 2: Right half, prediction = -10.0
        r2 = Region(box=[0.0, 0.5, 1.0, 1.0], prompt="Right", feather_radius=2)
        m2 = self.gen.generate_feathered_mask(r2)
        pred2 = mx.ones((1, 16, 16, 4)) * (-10.0)

        blended = self.blender.blend_spatial(
            regional_predictions=[pred1, pred2],
            regional_masks=[m1, m2],
        )

        self.assertEqual(blended.shape, (1, 16, 16, 4))

        # Deep inside left half, blended should be 10.0
        left_val = blended[0, 8, 2, 0].item()
        self.assertAlmostEqual(left_val, 10.0, places=2)

        # Deep inside right half, blended should be -10.0
        right_val = blended[0, 8, 14, 0].item()
        self.assertAlmostEqual(right_val, -10.0, places=2)

        # Right at the boundary seam, it should smoothly interpolate near 0.0
        seam_val = blended[0, 8, 8, 0].item()
        self.assertTrue(abs(seam_val) < 5.0)

    def test_base_prediction_fallback(self):
        # A small central box
        r = Region(box=[0.25, 0.25, 0.75, 0.75], prompt="Center", feather_radius=0)
        m = self.gen.generate_feathered_mask(r)
        pred_box = mx.ones((1, 16, 16, 4)) * 5.0

        # Base background prediction = 1.0
        base_pred = mx.ones((1, 16, 16, 4)) * 1.0

        blended = self.blender.blend_spatial(
            regional_predictions=[pred_box],
            regional_masks=[m],
            base_prediction=base_pred,
        )

        # Center should be 5.0
        self.assertAlmostEqual(blended[0, 8, 8, 0].item(), 5.0, places=2)

        # Corner should fall back to base prediction 1.0
        self.assertAlmostEqual(blended[0, 0, 0, 0].item(), 1.0, places=2)

    def test_channels_first_blending(self):
        # Shape used by Z-Image: (16, 1, 16, 16)
        r1 = Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Left", feather_radius=2)
        m1 = self.gen.generate_feathered_mask(r1)
        pred1 = mx.ones((16, 1, 16, 16)) * 10.0

        r2 = Region(box=[0.0, 0.5, 1.0, 1.0], prompt="Right", feather_radius=2)
        m2 = self.gen.generate_feathered_mask(r2)
        pred2 = mx.ones((16, 1, 16, 16)) * (-10.0)

        blended = self.blender.blend_spatial(
            regional_predictions=[pred1, pred2],
            regional_masks=[m1, m2],
        )

        self.assertEqual(blended.shape, (16, 1, 16, 16))
        # Deep inside left half
        self.assertAlmostEqual(blended[5, 0, 8, 2].item(), 10.0, places=2)
        # Deep inside right half
        self.assertAlmostEqual(blended[5, 0, 8, 14].item(), -10.0, places=2)


if __name__ == "__main__":
    unittest.main()
