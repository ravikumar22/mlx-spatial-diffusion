"""MLX Spatial Diffusion - Native Regional Prompting for Apple Silicon."""

__version__ = "0.1.0"

from mlx_spatial_diffusion.core.mask import Region, RegionMaskGenerator, build_gaussian_kernel_2d
from mlx_spatial_diffusion.core.blending import FeatheredBlender
from mlx_spatial_diffusion.core.scheduler import SpatialFlowMatchScheduler
from mlx_spatial_diffusion.core.spatial_attention import build_spatial_cross_attention_mask
from mlx_spatial_diffusion.pipeline import SpatialPipeline
from mlx_spatial_diffusion.models.z_image_spatial import SpatialZImageTurbo
from mlx_spatial_diffusion.layout import LayoutResolver, CINEMATIC_PRESETS

__all__ = [
    "Region",
    "RegionMaskGenerator",
    "build_gaussian_kernel_2d",
    "FeatheredBlender",
    "SpatialFlowMatchScheduler",
    "build_spatial_cross_attention_mask",
    "SpatialPipeline",
    "SpatialZImageTurbo",
    "LayoutResolver",
    "CINEMATIC_PRESETS",
    "__version__",
]
