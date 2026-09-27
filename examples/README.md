# Examples & Experiments

This directory contains standalone scripts demonstrating various spatial diffusion capabilities on Apple Silicon:

| Script | Model / Quantization | Description |
| :--- | :--- | :--- |
| `run_anime_story.py` | Z-Image-Turbo (8-bit) | Two-character fantasy anime scene (Celestial Alchemist vs. Abyssal Death Knight) demonstrating 100% split separation and zero color bleed. |
| `run_cinematic_knighting_test.py` | Z-Image-Turbo (8-bit) | The Royal Knighting Ceremony using pre-calibrated `kneeling_ceremony` bounding boxes and vaulted ceiling background. |
| `run_knighting_ceremony.py` | Z-Image-Turbo (8-bit) | Single-pass spatial cross-attention masking test on multi-character scenes. |
| `run_knighting_ceremony_4bit.py` | Z-Image-Turbo (4-bit) | 4-bit quantization comparison test analyzing token density and quantization limits. |
| `test_spatial_run.py` | Z-Image-Turbo (8-bit) | Basic left-vs-right split test (Warrior in blue armor vs. King in red robes). |

## How to Run

```bash
# Ensure dependencies are installed
pip install -e .

# Run any example directly
python examples/run_anime_story.py
python examples/run_cinematic_knighting_test.py
```
