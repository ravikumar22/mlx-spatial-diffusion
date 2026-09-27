"""High-level spatial diffusion generation pipeline for MLX."""

from typing import List, Optional, Union
import mlx.core as mx

from mlx_spatial_diffusion.core.mask import Region, RegionMaskGenerator
from mlx_spatial_diffusion.core.blending import FeatheredBlender
from mlx_spatial_diffusion.core.scheduler import SpatialFlowMatchScheduler


class SpatialPipeline:
    """High-level API for running MultiDiffusion with regional prompts on Apple Silicon."""

    def __init__(
        self,
        model_name: str = "flux-schnell",
        quantize: int = 4,
        latent_channels: int = 16,
        downsample_factor: int = 8,
    ):
        self.model_name = model_name
        self.quantize = quantize
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.scheduler = SpatialFlowMatchScheduler()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "flux-schnell",
        quantize: int = 4,
    ) -> "SpatialPipeline":
        """Instantiates the pipeline configured for the given model and quantization."""
        return cls(model_name=model_name, quantize=quantize)

    def generate(
        self,
        regions: List[Region],
        base_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        steps: int = 4,
        seed: Optional[int] = None,
        model_forward_fn: Optional[callable] = None,
    ) -> mx.array:
        """Generates an image latent tensor using multi-pass spatial denoising.

        Args:
            regions: List of Region instances with bounding boxes and prompts
            base_prompt: Optional fallback prompt for background/uncovered canvas
            height: Image pixel height (multiple of downsample_factor)
            width: Image pixel width (multiple of downsample_factor)
            steps: Number of Euler integration steps (default: 4 for schnell)
            seed: Optional random seed for reproducible initial noise
            model_forward_fn: Custom model evaluation callable (latent, prompt, t) -> velocity

        Returns:
            Final denoised latent tensor of shape (1, latent_h, latent_w, latent_c)
        """
        latent_h = height // self.downsample_factor
        latent_w = width // self.downsample_factor

        # Set random seed if provided
        if seed is not None:
            mx.random.seed(seed)

        # 1. Initialize random Gaussian latent at t=1.0
        latent = mx.random.normal((1, latent_h, latent_w, self.latent_channels))

        # 2. Pre-generate feathered spatial masks for all regions
        mask_gen = RegionMaskGenerator(latent_height=latent_h, latent_width=latent_w)
        masks = [mask_gen.generate_feathered_mask(r) for r in regions]

        # 3. Retrieve scheduler timesteps (e.g. 1.0 -> 0.75 -> 0.5 -> 0.25 -> 0.0)
        timesteps = self.scheduler.get_timesteps(num_steps=steps)

        # 4. MultiDiffusion Denoising Loop
        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_prev = timesteps[i + 1]

            regional_velocities = []

            # Evaluate each region's prompt
            for region in regions:
                if model_forward_fn is not None:
                    v_reg = model_forward_fn(latent, region.prompt, t_curr)
                else:
                    # Default mock velocity generator for testing pipeline orchestration
                    v_reg = mx.sin(latent) * (1.0 + hash(region.prompt) % 10 * 0.05)
                regional_velocities.append(v_reg)

            # Evaluate base/background prompt if provided
            base_velocity = None
            if base_prompt is not None:
                if model_forward_fn is not None:
                    base_velocity = model_forward_fn(latent, base_prompt, t_curr)
                else:
                    base_velocity = mx.cos(latent) * 0.5

            # Step forward in time with blended regional predictions
            latent, _ = self.scheduler.step(
                latent=latent,
                regional_velocities=regional_velocities,
                regional_masks=masks,
                t_curr=t_curr,
                t_prev=t_prev,
                base_velocity=base_velocity,
            )

        return latent
