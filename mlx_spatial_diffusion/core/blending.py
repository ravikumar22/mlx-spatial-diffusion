"""Feathered latent blending for MultiDiffusion in MLX."""

from typing import List, Tuple, Optional
import mlx.core as mx


class FeatheredBlender:
    """Blends multiple regional latent noise or velocity predictions into a coherent global tensor."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def blend_spatial(
        self,
        regional_predictions: List[mx.array],
        regional_masks: List[mx.array],
        base_prediction: Optional[mx.array] = None,
        base_weight: float = 1.0,
    ) -> mx.array:
        """Blends 4D spatial tensors (e.g. NHWC or NCHW).

        Args:
            regional_predictions: List of tensors of shape (N, H, W, C)
            regional_masks: List of mask tensors of shape (1, H, W, 1)
            base_prediction: Optional background prediction tensor of shape (N, H, W, C)
            base_weight: Strength factor for the base/background prediction

        Returns:
            Blended tensor of shape (N, H, W, C)
        """
        if len(regional_predictions) != len(regional_masks):
            raise ValueError(
                f"Number of predictions ({len(regional_predictions)}) does not match "
                f"number of masks ({len(regional_masks)})"
            )

        if len(regional_predictions) == 0:
            if base_prediction is not None:
                return base_prediction
            raise ValueError("No regional predictions or base prediction provided to blender.")

        target_shape = regional_predictions[0].shape
        accumulated_pred = mx.zeros(target_shape, dtype=regional_predictions[0].dtype)
        accumulated_weight = mx.zeros((1, target_shape[1], target_shape[2], 1), dtype=mx.float32)

        # Accumulate regional predictions weighted by their feathered masks
        for pred, mask in zip(regional_predictions, regional_masks):
            accumulated_pred = accumulated_pred + (pred * mask)
            accumulated_weight = accumulated_weight + mask

        # If a base/global prediction is provided, calculate the uncovered canvas residual
        if base_prediction is not None:
            # Residual weight fills any areas not fully covered by regions
            uncovered = mx.maximum(0.0, 1.0 - accumulated_weight) * base_weight
            accumulated_pred = accumulated_pred + (base_prediction * uncovered)
            accumulated_weight = accumulated_weight + uncovered

        # Normalize by accumulated mask weights
        norm_factor = mx.maximum(accumulated_weight, self.eps)
        blended = accumulated_pred / norm_factor

        return blended

    def blend_tokens(
        self,
        regional_predictions: List[mx.array],
        regional_masks_2d: List[mx.array],
        patch_size: int = 2,
        base_prediction: Optional[mx.array] = None,
        base_weight: float = 1.0,
    ) -> mx.array:
        """Blends flattened DiT token sequences (e.g. B, Sequence_Length, Dim)

        by pooling 2D masks into patch tokens.
        """
        # Downsample masks to patch resolution
        # If mask is (1, H, W, 1), patch resolution is (1, H // patch_size, W // patch_size, 1)
        pooled_masks = []
        for m in regional_masks_2d:
            b, h, w, c = m.shape
            ph, pw = h // patch_size, w // patch_size
            reshaped = mx.reshape(m[:, :ph * patch_size, :pw * patch_size, :], (b, ph, patch_size, pw, patch_size, c))
            pooled = mx.mean(reshaped, axis=(2, 4))  # (b, ph, pw, c)
            # Flatten to token sequence (b, ph * pw, 1)
            token_mask = mx.reshape(pooled, (b, ph * pw, 1))
            pooled_masks.append(token_mask)

        # Accumulate in token space
        accumulated_pred = mx.zeros_like(regional_predictions[0])
        accumulated_weight = mx.zeros((1, regional_predictions[0].shape[1], 1), dtype=mx.float32)

        for pred, t_mask in zip(regional_predictions, pooled_masks):
            accumulated_pred = accumulated_pred + (pred * t_mask)
            accumulated_weight = accumulated_weight + t_mask

        if base_prediction is not None:
            uncovered = mx.maximum(0.0, 1.0 - accumulated_weight) * base_weight
            accumulated_pred = accumulated_pred + (base_prediction * uncovered)
            accumulated_weight = accumulated_weight + uncovered

        norm_factor = mx.maximum(accumulated_weight, self.eps)
        return accumulated_pred / norm_factor
