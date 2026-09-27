"""Unit tests for SpatialPipeline end-to-end orchestration."""

import unittest
import mlx.core as mx
from mlx_spatial_diffusion import SpatialPipeline, Region


class TestSpatialPipeline(unittest.TestCase):
    def test_pipeline_generate(self):
        pipeline = SpatialPipeline.from_pretrained("flux-schnell", quantize=4)

        regions = [
            Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Warrior in blue armor"),
            Region(box=[0.0, 0.5, 1.0, 1.0], prompt="King in red robes"),
        ]

        latent = pipeline.generate(
            regions=regions,
            base_prompt="Castle hall",
            height=64,
            width=64,
            steps=4,
            seed=42,
        )

        # Expected latent spatial size: 64 // 8 = 8
        self.assertEqual(latent.shape, (1, 8, 8, 16))
        # Ensure values are finite and calculated
        self.assertFalse(mx.any(mx.isnan(latent)).item())

    def test_pipeline_custom_forward_fn(self):
        pipeline = SpatialPipeline()

        call_counts = {"Warrior": 0, "King": 0, "Base": 0}

        def mock_forward(latent, prompt, t):
            if "Warrior" in prompt:
                call_counts["Warrior"] += 1
                return mx.ones_like(latent) * 1.0
            elif "King" in prompt:
                call_counts["King"] += 1
                return mx.ones_like(latent) * (-1.0)
            else:
                call_counts["Base"] += 1
                return mx.zeros_like(latent)

        regions = [
            Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Warrior"),
            Region(box=[0.0, 0.5, 1.0, 1.0], prompt="King"),
        ]

        latent = pipeline.generate(
            regions=regions,
            base_prompt="Base",
            height=32,
            width=32,
            steps=3,
            model_forward_fn=mock_forward,
        )

        # With steps=3, each regional forward function should be called exactly 3 times
        self.assertEqual(call_counts["Warrior"], 3)
        self.assertEqual(call_counts["King"], 3)
        self.assertEqual(call_counts["Base"], 3)
        self.assertEqual(latent.shape, (1, 4, 4, 16))


if __name__ == "__main__":
    unittest.main()
