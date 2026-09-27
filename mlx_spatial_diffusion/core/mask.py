"""Spatial region masking and coordinate translation for MLX."""

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import mlx.core as mx


@dataclass
class Region:
    """Defines a spatial region within an image canvas.

    Coordinates are normalized in [0.0, 1.0]:
        box = [ymin, xmin, ymax, xmax]
    """
    box: List[float]
    prompt: str
    weight: float = 1.0
    feather_radius: int = 4
    negative_prompt: Optional[str] = None

    def __post_init__(self):
        if len(self.box) != 4:
            raise ValueError(f"Region box must have 4 elements [ymin, xmin, ymax, xmax], got {len(self.box)}")
        ymin, xmin, ymax, xmax = self.box
        if not (0.0 <= ymin <= ymax <= 1.0) or not (0.0 <= xmin <= xmax <= 1.0):
            raise ValueError(
                f"Coordinates must satisfy 0.0 <= ymin <= ymax <= 1.0 and 0.0 <= xmin <= xmax <= 1.0. "
                f"Got y: [{ymin}, {ymax}], x: [{xmin}, {xmax}]"
            )
        if self.feather_radius < 0:
            raise ValueError(f"feather_radius must be non-negative, got {self.feather_radius}")


def build_gaussian_kernel_2d(radius: int, sigma: Optional[float] = None) -> mx.array:
    """Builds a normalized 2D Gaussian convolution kernel in MLX.

    Kernel shape: (1, 2*radius+1, 2*radius+1, 1) suitable for mx.conv2d on NHWC.
    """
    if radius <= 0:
        return mx.ones((1, 1, 1, 1), dtype=mx.float32)

    if sigma is None:
        sigma = max(radius / 2.0, 0.5)

    x = mx.arange(-radius, radius + 1, dtype=mx.float32)
    k1d = mx.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / mx.sum(k1d)
    k2d = k1d[:, None] * k1d[None, :]
    # Return shape (C_out, KH, KW, C_in) = (1, 2*radius+1, 2*radius+1, 1)
    return k2d[None, :, :, None]


class RegionMaskGenerator:
    """Generates binary and feathered spatial masks on MLX arrays."""

    def __init__(self, latent_height: int, latent_width: int):
        self.height = latent_height
        self.width = latent_width

    def to_latent_indices(self, box: List[float]) -> Tuple[int, int, int, int]:
        """Converts normalized [ymin, xmin, ymax, xmax] into discrete latent grid indices."""
        ymin, xmin, ymax, xmax = box
        y_start = int(round(ymin * self.height))
        y_end = int(round(ymax * self.height))
        x_start = int(round(xmin * self.width))
        x_end = int(round(xmax * self.width))

        # Clamp to bounds
        y_start = max(0, min(self.height, y_start))
        y_end = max(y_start, min(self.height, y_end))
        x_start = max(0, min(self.width, x_start))
        x_end = max(x_start, min(self.width, x_end))

        return y_start, x_start, y_end, x_end

    def generate_binary_mask(self, region: Region) -> mx.array:
        """Generates a binary mask of shape (1, height, width, 1) with values in {0.0, 1.0}."""
        y_start, x_start, y_end, x_end = self.to_latent_indices(region.box)

        # Build coordinate meshgrid
        grid_y, grid_x = mx.meshgrid(
            mx.arange(self.height), mx.arange(self.width), indexing="ij"
        )
        is_inside = (
            (grid_y >= y_start)
            & (grid_y < y_end)
            & (grid_x >= x_start)
            & (grid_x < x_end)
        )
        mask = mx.where(is_inside, 1.0, 0.0)
        return mask[None, :, :, None]

    def generate_feathered_mask(self, region: Region) -> mx.array:
        """Generates a smoothed Gaussian-feathered mask of shape (1, height, width, 1)."""
        binary_mask = self.generate_binary_mask(region)
        if region.feather_radius <= 0:
            return binary_mask * region.weight

        kernel = build_gaussian_kernel_2d(region.feather_radius)
        feathered = mx.conv2d(binary_mask, kernel, padding=region.feather_radius)

        # Scale by region weight
        return feathered * region.weight
