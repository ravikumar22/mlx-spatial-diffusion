"""Anime Story Test: The Celestial Alchemist & The Abyssal Knight

Runs MLX Spatial Diffusion with 8-bit quantization (quantize=8) to eliminate VRAM swap thrashing.
"""

import os
os.environ["MFLUX_MLX_CACHE_LIMIT_GB"] = "-1"

import time
from mlx_spatial_diffusion import SpatialZImageTurbo, Region

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("🚀 Loading Z-Image-Turbo with 8-bit Quantization (quantize=8)...")
print("   -> Shrinking memory footprint from 31 GB down to ~15 GB")
print("   -> Prevents SSD swap thrashing on Apple Silicon")
t0 = time.time()
model = SpatialZImageTurbo(quantize=8)
t_load = time.time() - t0
print(f"✅ Model loaded and quantized to 8-bit in {t_load:.2f}s!")

# Creative Anime Story: "The Celestial Alchemist & The Abyssal Knight"
regions = [
    Region(
        box=[0.0, 0.0, 1.0, 0.5],
        prompt=(
            "Masterpiece anime illustration, a radiant celestial alchemist girl with "
            "long flowing pastel pink hair, golden eyes, wearing an ornate white and gold "
            "Academy coat, holding a glowing sphere of starlight, floating cherry blossom petals, "
            "vibrant daylight, Makoto Shinkai aesthetic, cinematic lighting, high quality anime art"
        ),
        feather_radius=3,
    ),
    Region(
        box=[0.0, 0.5, 1.0, 1.0],
        prompt=(
            "Masterpiece anime illustration, a brooding shadow knight with spiky raven-black hair, "
            "crimson glowing eyes, wearing dark obsidian armor and a tattered midnight cape, "
            "wielding a dark blade crackling with violet lightning and nebula energy, dark night sky, "
            "Ufotable aesthetic, dramatic lighting, high quality anime art"
        ),
        feather_radius=3,
    ),
]

print("=" * 65)
print("🎨 Generating Anime Story with MLX Spatial MultiDiffusion...")
print("   - Left (50%):  Celestial Alchemist (Pink hair, Starlight, Cherry blossoms)")
print("   - Right (50%): Abyssal Knight (Black hair, Violet lightning, Obsidian armor)")
print("   - Precision:   8-bit Quantized")
print("   - Steps:       4 steps (Turbo Flow-Match)")

t_gen0 = time.time()
image = model.generate_spatial(
    regions=regions,
    seed=108,
    height=512,
    width=512,
    num_inference_steps=4,
)
t_gen = time.time() - t_gen0

out_path = "outputs/anime_story_spatial_8bit.png"
image.save(out_path)
print("=" * 65)
print(f"🎉 Generation complete in {t_gen:.2f}s!")
print(f"💾 Saved image to: {out_path}")
print("=" * 65)
