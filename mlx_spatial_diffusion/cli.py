"""Command Line Interface for MLX Spatial Diffusion."""

import argparse
import os
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-spatial",
        description="MLX Spatial Diffusion: Native Regional Prompting on Apple Silicon 🍏🎨",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: generate
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate an image using cinematic spatial layout presets",
    )
    gen_parser.add_argument(
        "--preset",
        type=str,
        default="two_shot_eye_level",
        help="Cinematic shot preset (e.g. two_shot_eye_level, kneeling_ceremony, duel_confrontation)",
    )
    gen_parser.add_argument(
        "--left",
        type=str,
        required=True,
        help="Prompt for left subject / primary character",
    )
    gen_parser.add_argument(
        "--right",
        type=str,
        required=True,
        help="Prompt for right subject / secondary character",
    )
    gen_parser.add_argument(
        "--env",
        "--background",
        dest="env",
        type=str,
        default="",
        help="Continuous background environment prompt (e.g. 'Gothic cathedral hall with stained glass')",
    )
    gen_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output_spatial.png",
        help="Path to save the generated image (default: output_spatial.png)",
    )
    gen_parser.add_argument(
        "--quantize",
        "-q",
        type=int,
        default=8,
        choices=[4, 8],
        help="Model weight quantization bits (default: 8)",
    )
    gen_parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=4,
        help="Number of denoising inference steps (default: 4)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random generation seed (default: 42)",
    )
    gen_parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (default: 512)",
    )
    gen_parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (default: 512)",
    )

    # Command: presets
    subparsers.add_parser(
        "presets",
        help="List all available cinematic shot archetypes and slot mappings",
    )

    # Command: ui
    subparsers.add_parser(
        "ui",
        help="Launch the interactive Web UI studio (planned for v0.2.0)",
    )

    # Backwards-compatible flag: --ui
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch interactive Web UI studio",
    )

    return parser


def handle_generate(args: argparse.Namespace):
    from mlx_spatial_diffusion.layout.resolver import LayoutResolver
    from mlx_spatial_diffusion.models.z_image_spatial import SpatialZImageTurbo

    print("=" * 65)
    print("🍏 MLX Spatial Diffusion — Regional Prompting on Apple Silicon")
    print("=" * 65)

    # Uncap MLX cache to avoid artificial thrashing on Apple Silicon
    os.environ["MFLUX_MLX_CACHE_LIMIT_GB"] = "-1"

    resolver = LayoutResolver()
    scene = {
        "layout": args.preset,
        "environment": args.env,
        "characters_in_scene": {
            "left": args.left,
            "right": args.right,
        },
    }

    print(f"📐 Resolving spatial layout: '{args.preset}'...")
    regions, base_prompt = resolver.resolve(scene)
    for idx, reg in enumerate(regions):
        print(f"   • Region {idx + 1} ({reg.box}): {reg.prompt[:60]}...")
    if base_prompt:
        print(f"   • Base Canvas: {base_prompt[:60]}...")

    print(f"\n📦 Loading Z-Image-Turbo ({args.quantize}-bit)...")
    model = SpatialZImageTurbo(quantize=args.quantize)

    print(f"\n⚡ Generating image ({args.width}x{args.height}, {args.steps} steps, seed={args.seed})...")
    image = model.generate_spatial(
        regions=regions,
        base_prompt=base_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"\n✨ Success! Image saved to: {out_path.resolve()}")


def handle_presets():
    from mlx_spatial_diffusion.layout.resolver import CINEMATIC_PRESETS

    print("=" * 70)
    print("🎬 MLX Spatial Diffusion — Built-in Cinematic Shot Archetypes")
    print("=" * 70)
    for name, preset in CINEMATIC_PRESETS.items():
        print(f"\n[{name}]")
        print(f"  Description: {preset.description}")
        print("  Slots:")
        for slot_name, slot in preset.slots.items():
            aliases = f" (aliases: {', '.join(slot.aliases)})" if slot.aliases else ""
            print(f"    - {slot_name}: box={slot.box}, feather={slot.feather_radius}{aliases}")
    print("\n" + "=" * 70)


def handle_ui():
    print("=" * 65)
    print("🎨 MLX Spatial Studio Web UI")
    print("=" * 65)
    print("The interactive Web UI drag-and-drop canvas is slated for v0.2.0.")
    print("To track progress or contribute, see docs/ROADMAP.md.")
    print("\nIn the meantime, you can generate bleed-free images using the CLI:")
    print("  mlx-spatial generate --preset two_shot_eye_level \\")
    print("    --left \"Warrior in glowing blue armor\" \\")
    print("    --right \"King in royal red velvet robes\" \\")
    print("    --output output.png")
    print("=" * 65)


def main():
    parser = create_parser()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.ui or args.command == "ui":
        handle_ui()
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "presets":
        handle_presets()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
