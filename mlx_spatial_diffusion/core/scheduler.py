"""Spatial Flow Matching Euler Scheduler in MLX."""

from typing import List, Tuple, Optional, Callable
import mlx.core as mx

from mlx_spatial_diffusion.core.blending import FeatheredBlender


class SpatialFlowMatchScheduler:
    """Euler flow matching scheduler supporting regional MultiDiffusion in MLX.

    Formulation:
        dx/dt = v(x, t)
        x_{t - dt} = x_t + (t_prev - t_curr) * v_blended
    """

    def __init__(self, shift: float = 1.0, eps: float = 1e-6):
        self.shift = shift
        self.blender = FeatheredBlender(eps=eps)

    def get_timesteps(self, num_steps: int) -> List[float]:
        """Calculates linearly spaced or shifted flow match timesteps from 1.0 to 0.0."""
        # Standard flow match schedule from t=1.0 (pure noise) down to t=0.0 (clean latent)
        timesteps = [1.0 - (i / num_steps) for i in range(num_steps + 1)]
        if self.shift != 1.0:
            # Time-shift function: t_shifted = (shift * t) / (1 + (shift - 1) * t)
            timesteps = [
                (self.shift * t) / (1.0 + (self.shift - 1.0) * t) if t > 0.0 else 0.0
                for t in timesteps
            ]
        return timesteps

    def step(
        self,
        latent: mx.array,
        regional_velocities: List[mx.array],
        regional_masks: List[mx.array],
        t_curr: float,
        t_prev: float,
        base_velocity: Optional[mx.array] = None,
        base_weight: float = 1.0,
    ) -> Tuple[mx.array, mx.array]:
        """Executes a single Euler ODE integration step with blended regional velocities.

        Args:
            latent: Current latent state x_t
            regional_velocities: Predicted velocity field per region
            regional_masks: Feathered mask per region
            t_curr: Current timestep
            t_prev: Next/target timestep (t_prev < t_curr)
            base_velocity: Optional background/base prompt velocity
            base_weight: Influence of base velocity

        Returns:
            (next_latent, blended_velocity)
        """
        # Blend velocities across regions
        blended_velocity = self.blender.blend_spatial(
            regional_predictions=regional_velocities,
            regional_masks=regional_masks,
            base_prediction=base_velocity,
            base_weight=base_weight,
        )

        # Euler step: x_{prev} = x_{curr} + dt * v
        dt = t_prev - t_curr
        next_latent = latent + (dt * blended_velocity)

        return next_latent, blended_velocity
