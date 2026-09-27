"""Quick test script for MLX Spatial Diffusion with Z-Image-Turbo."""

import os
os.environ["MFLUX_MLX_CACHE_LIMIT_GB"] = "-1"

import time
from mlx_spatial_diffusion import SpatialZImageTurbo, Region

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("🚀 Loading Z-Image-Turbo weights into MLX...")
t0 = time.time()
model = SpatialZImageTurbo()
t_load = time.time() - t0
print(f"✅ Model loaded successfully in {t_load:.2f}s!")

# Define regions: Left half vs Right half (covers 100% of canvas)
regions = [
    Region(
        box=[0.0, 0.0, 1.0, 0.5],
        prompt="A warrior in blue glowing armor holding a sword, fantasy art, highly detailed",
        feather_radius=3,
    ),
    Region(
        box=[0.0, 0.5, 1.0, 1.0],
        prompt="A king in regal red velvet robes wearing a golden crown, fantasy art, highly detailed",
        feather_radius=3,
    ),
]

print("=" * 60)
print("🎨 Running MLX Spatial MultiDiffusion (4 steps, 512x512)...")
print("   - Region 1 (Left 50%):  'Warrior in blue armor'")
print("   - Region 2 (Right 50%): 'King in red robes'")
print("   - Note: Step 1 includes Metal GPU kernel JIT compilation")

t_gen0 = time.time()
image = model.generate_spatial(
    regions=regions,
    seed=42,
    height=512,
    width=512,
    num_inference_steps=4,
)
t_gen = time.time() - t_gen0

out_path = "outputs/test_spatial.png"
image.save(out_path)
print("=" * 60)
print(f"🎉 Generation complete in {t_gen:.2f}s!")
print(f"💾 Saved image to: {out_path}")
print("=" * 60)
