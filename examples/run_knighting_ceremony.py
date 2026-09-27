"""Complex Multi-Character Anime Scene: The Royal Knighting Ceremony

Demonstrates Option B (Single-Pass Spatial Attention Masking).
Characters:
1. The King (Grand throne, golden crown, crimson robes, holding ceremonial sword)
2. The Queen (Beside throne, sapphire silk gown, silver tiara, watching with pride)
3. The Kneeling Soldier (On bent knee, bowed head, polished knight plate armor)
4. The High Priest (Sacred white & gold vestments, holding golden blessing staff)
"""

import os
os.environ["MFLUX_MLX_CACHE_LIMIT_GB"] = "-1"

import time
from mlx_spatial_diffusion import SpatialZImageTurbo, Region

os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("👑 The Royal Knighting Ceremony (Single-Pass Spatial Attention)")
print("   -> 4 Distinct Characters across the Canvas Grid")
print("   -> Option B: Single-Pass SDPA Attention Masking")
print("=" * 70)

print("\n🚀 Loading Z-Image-Turbo with 8-bit Quantization...")
t0 = time.time()
model = SpatialZImageTurbo(quantize=8)
print(f"✅ Model ready in {time.time() - t0:.2f}s!")

# Define spatial coordinates for 4 characters in the cathedral throne room:
regions = [
    # 1. The High Priest (Far Left: x in [0.0, 0.25])
    Region(
        box=[0.0, 0.0, 0.70, 0.25],
        prompt=(
            "Masterpiece anime art, a venerable high priest with long white beard and serene eyes, "
            "wearing white and gold ceremonial sacred vestments, holding an ornate golden sun staff, "
            "divine golden aura, blessing the ceremony, Ufotable anime style"
        ),
        feather_radius=2,
    ),
    # 2. The Kneeling Soldier (Bottom Center-Left: x in [0.15, 0.50], y in [0.35, 1.0])
    Region(
        box=[0.35, 0.15, 1.0, 0.50],
        prompt=(
            "Masterpiece anime art, a valiant young soldier kneeling reverently on one bent knee, "
            "head bowed in solemn devotion, wearing polished silver knight plate armor with blue cape, "
            "accepting the royal oath, dramatic rim lighting, detailed anime illustration"
        ),
        feather_radius=2,
    ),
    # 3. The King (Center-Right Throne: x in [0.45, 0.75], y in [0.0, 0.75])
    Region(
        box=[0.0, 0.45, 0.75, 0.75],
        prompt=(
            "Masterpiece anime art, a majestic noble king with a golden crown and regal beard, "
            "seated on a grand throne, wearing crimson velvet robes with gold trim, "
            "extending a glowing silver ceremonial sword touching the soldier's shoulder, "
            "commanding presence, cinematic lighting"
        ),
        feather_radius=2,
    ),
    # 4. The Queen (Far Right: x in [0.75, 1.0], y in [0.0, 0.70])
    Region(
        box=[0.0, 0.75, 0.70, 1.0],
        prompt=(
            "Masterpiece anime art, a beautiful graceful queen standing beside the throne, "
            "wearing an elegant sapphire-blue royal silk gown and diamond tiara, "
            "watching the knighting ceremony with a warm proud smile, Kyoto Animation aesthetic"
        ),
        feather_radius=2,
    ),
]

print("\n🎨 Generating 4-Character Scene via Single-Pass Spatial Attention...")
print("   [1] High Priest:       Far Left (White & Gold Vestments, Sun Staff)")
print("   [2] Kneeling Soldier:  Bottom Center (Silver Plate Armor, Bent Knee)")
print("   [3] King on Throne:    Center (Crown, Crimson Robes, Ceremonial Sword)")
print("   [4] Queen:             Right (Sapphire Silk Gown, Diamond Tiara)")
print("   - Canvas: 512x512 | Steps: 4")
print("=" * 70)

t_gen0 = time.time()
image = model.generate_single_pass(
    regions=regions,
    seed=777,
    height=512,
    width=512,
    num_inference_steps=4,
)
t_gen = time.time() - t_gen0

out_path = "outputs/knighting_ceremony_single_pass.png"
image.save(out_path)

print("=" * 70)
print(f"🎉 Scene generation complete in {t_gen:.2f}s!")
print(f"💾 Saved image to: {out_path}")
print("=" * 70)
