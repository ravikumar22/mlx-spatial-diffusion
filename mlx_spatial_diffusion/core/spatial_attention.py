"""Single-Pass Spatial Attention Masking for Diffusion Transformers (DiT).

Constructs unified spatial attention masks that constrain image patch tokens
to attend only to their designated regional text prompt tokens within the GPU attention kernel.
"""

from typing import List, Tuple, Optional
import mlx.core as mx

from mlx_spatial_diffusion.core.mask import Region


def build_spatial_cross_attention_mask(
    regions: List[Region],
    h_tokens: int,
    w_tokens: int,
    caption_token_lengths: List[int],
    total_caption_len: Optional[int] = None,
    feather_radius: int = 1,
) -> mx.array:
    """Builds a 4D additive attention mask of shape (1, 1, L_unified, L_unified)

    where L_unified = L_image + L_caption.

    Block structure of the attention matrix:
    [ Image Tokens (L_img) x Image Tokens (L_img) ]  -> 0.0 (Full Self-Attention)
    [ Image Tokens (L_img) x Caption Tokens (L_cap) ] -> Spatial Mask (0.0 for matching region, -inf otherwise)
    [ Caption Tokens (L_cap) x Unified Tokens (L_uni) ] -> 0.0 (Unrestricted Caption Attention)
    """
    l_img = h_tokens * w_tokens
    l_cap_actual = sum(caption_token_lengths)
    l_cap = total_caption_len if total_caption_len is not None else l_cap_actual
    l_total = l_img + l_cap

    # 1. Initialize full mask with 0.0 (allow all attention by default)
    mask = mx.zeros((l_total, l_total), dtype=mx.float32)

    # 2. Build coordinate grids for image tokens: (h_tokens, w_tokens)
    grid_y, grid_x = mx.meshgrid(
        mx.arange(h_tokens, dtype=mx.float32),
        mx.arange(w_tokens, dtype=mx.float32),
        indexing="ij",
    )
    # Normalized coordinates in [0.0, 1.0]
    norm_y = (grid_y + 0.5) / float(h_tokens)
    norm_x = (grid_x + 0.5) / float(w_tokens)
    flat_y = norm_y.reshape(-1)
    flat_x = norm_x.reshape(-1)

    # 3. Create the Image -> Caption cross-attention slice: shape (L_img, L_cap)
    # Default: mask out all caption tokens with -inf
    img_to_cap = mx.full((l_img, l_cap), float("-inf"), dtype=mx.float32)

    # 4. For each region, enable attention to its corresponding text tokens
    cap_start = 0
    neg_inf = float("-inf")

    for region, cap_len in zip(regions, caption_token_lengths):
        ymin, xmin, ymax, xmax = region.box
        cap_end = cap_start + cap_len

        # Check which image tokens are inside this region's bounding box
        is_inside = (
            (flat_y >= ymin)
            & (flat_y <= ymax)
            & (flat_x >= xmin)
            & (flat_x <= xmax)
        )

        # Region column indices mask
        col_indices = mx.arange(l_cap)
        is_region_col = (col_indices >= cap_start) & (col_indices < cap_end)

        # Enable attention (0.0) where token is inside and column belongs to this prompt
        # We use outer boolean broadcasting: (l_img, 1) & (1, l_cap)
        allow_attention = is_inside[:, None] & is_region_col[None, :]
        img_to_cap = mx.where(allow_attention, 0.0, img_to_cap)

        cap_start = cap_end

    # 5. Place img_to_cap into the unified attention matrix: rows [0:l_img], cols [l_img:l_total]
    # In MLX, we construct the combined matrix using concatenation
    top_left = mx.zeros((l_img, l_img), dtype=mx.float32)  # Image <-> Image is full attention (0.0)
    top_block = mx.concatenate([top_left, img_to_cap], axis=1)

    bottom_block = mx.zeros((l_cap, l_total), dtype=mx.float32)  # Caption tokens attend freely (0.0)

    unified_mask = mx.concatenate([top_block, bottom_block], axis=0)

    # Return shape (1, 1, L_total, L_total) suitable for SDPA
    return unified_mask[None, None, :, :]
