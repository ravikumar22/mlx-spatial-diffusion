"""Cinematic Knighting Scene: Unified Dynamic Background + 2-Character Focus

Features:
1. 8-bit Quantized Z-Image-Turbo (~15 GB VRAM, 0 swap)
2. Global Base Prompt for unbroken cathedral vaulted arches and marble floor reflections
3. Realistically bounded character boxes with ceiling headroom
4. Proven MultiDiffusion Latent Blending for 100% spatial isolation
"""

import os
os.environ["MFLUX_MLX_CACHE_LIMIT_GB"] = "-1"

import time
from mlx_spatial_diffusion import SpatialZImageTurbo, Region

os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("🎬 Cinematic Knighting Ceremony (Unified Background + 2-Character Focus)")
print("   -> Resolution: 512x512 (Standard Video Frame)")
print("   -> Precision:  8-bit Quantized (~15 GB, Zero Swap)")
print("   -> Engine:     Layered MultiDiffusion with Global Base Environment")
print("=" * 70)

t0 = time.time()
print("🚀 Loading Z-Image-Turbo with 8-bit Quantization...")
model = SpatialZImageTurbo(quantize=8)
print(f"✅ Model ready in {time.time() - t0:.2f}s!")

# 1. Global Environment (Base Prompt): fills the upper cathedral ceiling,
# arched stained-glass windows, and continuous floor reflections under both characters.
base_prompt = (
    "Masterpiece anime illustration, grand imperial gothic cathedral throne room, "
    "towering stone pillars and soaring vaulted ceilings, radiant stained glass windows "
    "with majestic golden god-rays streaming through, dark polished marble floor with "
    "ambient reflections, floating golden dust motes, cinematic lighting, Ufotable aesthetic"
)

# 2. Layered Character Regions with Ceiling Headroom:
regions = [
    # Left: Kneeling Soldier (y in [0.28, 0.98], x in [0.05, 0.48])
    # The top 28% of the left side is open for the cathedral arches & stained glass!
    Region(
        box=[0.28, 0.05, 0.98, 0.48],
        prompt=(
            "Masterpiece anime illustration, a valiant young soldier kneeling reverently "
            "on one bent knee, facing right toward the king, head bowed with solemn devotion, "
            "wearing polished silver knight plate armor with sapphire-blue mantle, "
            "receiving the sacred knighthood, cinematic rim lighting on steel armor, "
            "in a grand cathedral with polished marble reflections"
        ),
        feather_radius=3,
    ),
    # Right: King on Golden Throne (y in [0.12, 0.92], x in [0.48, 0.95])
    # Elevated on throne, extending ceremonial sword to the left toward the soldier.
    Region(
        box=[0.12, 0.48, 0.92, 0.95],
        prompt=(
            "Masterpiece anime illustration, a majestic noble king with a golden crown "
            "and regal beard, seated on an ornate golden throne, facing left toward the soldier, "
            "wearing crimson velvet royal robes with ermine trim, extending a glowing ceremonial "
            "silver sword to the left toward the soldier's shoulder, commanding regal expression, "
            "in a grand cathedral with warm golden illumination"
        ),
        feather_radius=3,
    ),
]

print("\n🎨 Generating Scene with Layered MultiDiffusion...")
print("   - Canvas Top (y: 0.0 - 0.28): Global Cathedral Arches & Stained Glass")
print("   - Left Character:             Kneeling Soldier (Facing Right, Silver Armor)")
print("   - Right Character:            King on Throne (Facing Left, Sword to Soldier)")
print("   - Canvas Bottom:              Continuous Reflective Marble Floor")
print("   - Steps: 4 steps | Seed: 314")
print("=" * 70)

t_gen0 = time.time()
image = model.generate_spatial(
    regions=regions,
    base_prompt=base_prompt,
    seed=314,
    height=512,
    width=512,
    num_inference_steps=4,
)
t_gen = time.time() - t_gen0

out_path = "outputs/cinematic_knighting_unified_8bit.png"
image.save(out_path)

print("=" * 70)
print(f"🎉 Cinematic Scene complete in {t_gen:.2f}s!")
print(f"💾 Saved image to: {out_path}")
print("=" * 70)
