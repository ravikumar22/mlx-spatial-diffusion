"""Unit tests for spatial flow match Euler scheduler."""

import unittest
import mlx.core as mx
from mlx_spatial_diffusion.core.mask import Region, RegionMaskGenerator
from mlx_spatial_diffusion.core.scheduler import SpatialFlowMatchScheduler


class TestSpatialScheduler(unittest.TestCase):
    def test_timesteps_monotonic(self):
        scheduler = SpatialFlowMatchScheduler()
        timesteps = scheduler.get_timesteps(num_steps=4)
        self.assertEqual(len(timesteps), 5)
        self.assertAlmostEqual(timesteps[0], 1.0)
        self.assertAlmostEqual(timesteps[-1], 0.0)
        # Check strictly decreasing
        for i in range(len(timesteps) - 1):
            self.assertTrue(timesteps[i] > timesteps[i + 1])

    def test_scheduler_step(self):
        scheduler = SpatialFlowMatchScheduler()
        gen = RegionMaskGenerator(latent_height=8, latent_width=8)

        r1 = Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Left", feather_radius=1)
        m1 = gen.generate_feathered_mask(r1)
        v1 = mx.ones((1, 8, 8, 4)) * 2.0

        r2 = Region(box=[0.0, 0.5, 1.0, 1.0], prompt="Right", feather_radius=1)
        m2 = gen.generate_feathered_mask(r2)
        v2 = mx.ones((1, 8, 8, 4)) * (-2.0)

        latent = mx.zeros((1, 8, 8, 4))
        t_curr = 1.0
        t_prev = 0.75  # dt = -0.25

        next_latent, blended_v = scheduler.step(
            latent=latent,
            regional_velocities=[v1, v2],
            regional_masks=[m1, m2],
            t_curr=t_curr,
            t_prev=t_prev,
        )

        self.assertEqual(next_latent.shape, (1, 8, 8, 4))
        # Left side: dt * v = -0.25 * 2.0 = -0.5
        left_val = next_latent[0, 4, 1, 0].item()
        self.assertAlmostEqual(left_val, -0.5, places=2)

        # Right side: dt * v = -0.25 * (-2.0) = +0.5
        right_val = next_latent[0, 4, 7, 0].item()
        self.assertAlmostEqual(right_val, 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
