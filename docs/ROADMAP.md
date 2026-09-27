# Development Roadmap

This roadmap defines the implementation trajectory for **MLX Spatial Diffusion**, prioritizing fast developer feedback, rapid community release, and hardware optimization on Apple Silicon.

---

## 🎯 Phase 1: Core Engine & Fast Prototype (Target: Day 1–7)
- [ ] **Repository Setup**: Initial Python package configuration (`pyproject.toml`, dependencies, CI tests).
- [ ] **Latent Coordinate Mapping (`RegionMask`)**:
  - Translate normalized bounding boxes `[ymin, xmin, ymax, xmax]` into latent tensor slice indices.
  - Implement Gaussian feathering on MLX arrays (`mx.array`) to produce soft blending weights.
- [ ] **Multi-Pass Denoising Loop**:
  - Hook into `mflux`'s generation pipeline or custom Euler Flow Match loop.
  - Evaluate regional prompt predictions and blend back into global latent state per timestep.
- [ ] **FLUX.1-schnell 4-bit Baseline**:
  - Achieve functional end-to-end generation in 4 steps on standard 16GB/18GB M-series Macs.
  - Verify zero concept bleed on the reference test: *"Warrior in blue armor on left, King in red robes on right"*.

---

## 🖥️ Phase 2: Visual Bounding-Box Web UI (Target: Week 2)
- [ ] **Interactive Canvas Interface**:
  - Lightweight local web UI (via Gradio or embedded HTML5 canvas).
  - Users can click and drag rectangular bounding boxes directly over a canvas.
  - Assign individual prompt inputs and strength sliders to each drawn box.
- [ ] **One-Command CLI Launch**:
  - Run `mlx-spatial --ui` to boot the interface locally on `http://localhost:7860`.
  - Export generation settings to reproducible JSON or Python code snippets.
- [ ] **Side-by-Side Generation Mode**:
  - Single-button "Compare vs Vanilla" to run standard generation alongside spatial generation for instant visual proof.

---

## ⚡ Phase 3: DiT Attention Optimization (Target: Week 3–4)
- [ ] **Batched Regional Text Ingestion**:
  - Pre-encode all sub-prompts in a single forward pass through T5 and CLIP text encoders to save memory and inference time.
- [ ] **Attention Masking / Spatial Bias**:
  - Transition from multi-pass model evaluations to single-pass masked self-attention within the DiT blocks where possible.
  - Cut generation latency down to near-baseline generation times.
- [ ] **Arbitrary Mask Painting**:
  - Support freehand brush masks in addition to rectangular bounding boxes.

---

## 🌐 Phase 4: Ecosystem Integration & Upstreaming (Target: Month 2)
- [ ] **ComfyUI-MLX Custom Node**:
  - Package the MLX spatial sampler as a custom node for Mac users using ComfyUI.
- [ ] **Upstream Collaboration**:
  - Submit modular PRs or integrate with `mflux-community` projects.
  - Provide a clean API for third-party macOS app developers (SwiftUI / Python bridges).
