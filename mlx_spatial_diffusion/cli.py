"""CLI entry point for MLX Spatial Diffusion."""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLX Spatial Diffusion: Native Regional Prompting on Apple Silicon"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive local web canvas UI",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flux-schnell",
        choices=["flux-schnell", "flux-dev"],
        help="Base diffusion model to use (default: flux-schnell)",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bit width (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ui:
        print("Launching MLX Spatial Studio Web UI on http://localhost:7860 ...")
        # UI launch implementation will be linked here in Phase 2
        sys.exit(0)
    else:
        print("MLX Spatial Diffusion CLI")
        print("Run with --ui to open the visual bounding box interface.")


if __name__ == "__main__":
    main()
