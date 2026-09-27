"""Unit tests for single-pass spatial attention masking."""

import unittest
import mlx.core as mx
from mlx_spatial_diffusion.core.mask import Region
from mlx_spatial_diffusion.core.spatial_attention import build_spatial_cross_attention_mask


class TestSpatialAttentionMask(unittest.TestCase):
    def test_attention_mask_blocks(self):
        # 4x4 image grid = 16 tokens
        h_tokens, w_tokens = 4, 4
        l_img = 16

        # Two regions: Left half vs Right half
        r1 = Region(box=[0.0, 0.0, 1.0, 0.5], prompt="Left prompt")
        r2 = Region(box=[0.0, 0.5, 1.0, 1.0], prompt="Right prompt")

        # Each prompt has 10 caption tokens
        cap_lens = [10, 10]
        l_cap = 20
        l_total = l_img + l_cap

        mask = build_spatial_cross_attention_mask(
            regions=[r1, r2],
            h_tokens=h_tokens,
            w_tokens=w_tokens,
            caption_token_lengths=cap_lens,
        )

        self.assertEqual(mask.shape, (1, 1, l_total, l_total))

        # Check Top-Left block (Image -> Image): must be unrestricted (0.0)
        img_self_attn = mask[0, 0, :l_img, :l_img]
        self.assertTrue(mx.all(img_self_attn == 0.0).item())

        # Check Bottom block (Caption -> All): must be unrestricted (0.0)
        cap_attn = mask[0, 0, l_img:, :]
        self.assertTrue(mx.all(cap_attn == 0.0).item())

        # Check Token 0: (y=0, x=0) -> Top-left corner -> In Region 1 (Left)
        # Should attend to Prompt 1 (cols l_img to l_img + 10) with 0.0
        # Should be blocked from Prompt 2 (cols l_img + 10 to l_total) with -inf
        tok0_to_p1 = mask[0, 0, 0, l_img : l_img + 10]
        tok0_to_p2 = mask[0, 0, 0, l_img + 10 : l_total]
        self.assertTrue(mx.all(tok0_to_p1 == 0.0).item())
        self.assertTrue(mx.all(tok0_to_p2 == float("-inf")).item())

        # Check Token 3: (y=0, x=3) -> Top-right corner -> In Region 2 (Right)
        # Should be blocked from Prompt 1 with -inf
        # Should attend to Prompt 2 with 0.0
        tok3_to_p1 = mask[0, 0, 3, l_img : l_img + 10]
        tok3_to_p2 = mask[0, 0, 3, l_img + 10 : l_total]
        self.assertTrue(mx.all(tok3_to_p1 == float("-inf")).item())
        self.assertTrue(mx.all(tok3_to_p2 == 0.0).item())


if __name__ == "__main__":
    unittest.main()
