# Open Source Launch Strategy & Pre-Release Checklist

This document details **Strategy 3: The Native MLX Playbook**, modeled after successful Apple Silicon projects (`mflux`, `mlx-vlm`, `mlx-audio`). Follow this checklist strictly **before** announcing the project publicly.

---

## 🚀 The Core Philosophy: Strategy 3

Open-source users on Apple Silicon do not want cloud GPU demos; they want **local empowerment**. They chose Mac hardware for privacy, portability, and unified memory. 

To maximize viral adoption, our launch relies on two pillars:
1. **Instant Visual Proof (15-Second Hook)**: Seeing is believing. A side-by-side comparison of "Concept Bleed" failure vs. MLX Spatial success.
2. **Zero-Friction Local Execution**: Installing and generating an image within 60 seconds with a single command:
   ```bash
   pip install mlx-spatial-diffusion
   mlx-spatial --ui
   ```

---

## 📋 Mandatory Pre-Release Checklist

Do **NOT** post to public forums until every item in this checklist is verified.

### 1. Visual Proof Assets
- [ ] **The Hero Side-by-Side Image**:
  - Left: Standard FLUX output (Concept bleed: colors and subjects mixed).
  - Right: MLX Spatial output (Clean separation: warrior in blue on left, king in red on right).
  - Both clearly annotated with generation time, RAM consumption, and Mac model used (e.g., `M3 Pro 18GB - 11.2s`).
- [ ] **15-Second Screen Recording / GIF**:
  - Show the user dragging two bounding boxes in the web UI.
  - Type prompt for Box 1: *"A cyberpunk neon motorcycle"*.
  - Type prompt for Box 2: *"An ancient stone castle in fog"*.
  - Hit **Generate** $\rightarrow$ see the image render in seconds on Apple Silicon.
- [ ] Place the GIF at the very top of `README.md` (above the fold).

### 2. Hardware & Installation Sanity Check
- [ ] Verify `pip install -e .` on a clean virtual environment without manual Metal compilation steps.
- [ ] Test on base 16GB/18GB Apple Silicon models to ensure no out-of-memory (OOM) or memory pressure panics.
- [ ] Ensure FLUX.1-schnell (4-bit) downloads seamlessly from Hugging Face on the first run with automatic progress bars.

### 3. Documentation & Developer Ergonomics
- [ ] Root `README.md` contains clear copy-paste commands and architecture highlights.
- [ ] Quickstart code snippet works out of the box in under 10 lines of Python.
- [ ] License file (MIT / Apache 2.0) present in repository.

---

## 📢 Public Launch Channels & Copy Templates

### Channel 1: Hacker News (Show HN)
* **Title**: `Show HN: MLX Spatial Diffusion – Regional Prompting on Apple Silicon`
* **Body Outline**:
  - The problem: Text-to-image models suffer from concept bleed because cross-attention is global.
  - The Mac penalty: Running ComfyUI with PyTorch MPS is slow, memory-inefficient, and prone to crashes.
  - The solution: Built a native MLX implementation that performs multi-diffusion and latent blending on unified memory in seconds.
  - Link directly to GitHub repo with the visual side-by-side demo.

### Channel 2: Reddit (`r/StableDiffusion`, `r/LocalLLaMA`, `r/AppleSilicon`)
* **Format**: Image/Video post with the 15-second comparison clip.
* **Title**: `[P] We built native Regional Prompting for Apple Silicon with MLX (No PyTorch/MPS bottlenecks, runs on 16GB Macs)`
* **Top Comment**: Deep-dive explanation of the latent slicing mechanism and how community members can try `mlx-spatial --ui`.

### Channel 3: X / Twitter
* **Hook**: Tag Apple MLX leadership (`@awnihannun`) and key open-source creators (`@mflux_ai` / Filip Strand).
* **Format**: 15s video clip with direct GitHub link.
* **Sample Tweet**:
  > *"Tired of concept bleed when generating multi-character scenes on Mac? We just open-sourced MLX Spatial Diffusion: native regional prompting on Apple Silicon. 4-bit FLUX.1-schnell in seconds on 18GB MacBooks with zero PyTorch overhead. 🍏✨ Repo below 👇"*

### Channel 4: Upstream Ecosystem Collaboration
* Open an issue or discussion on `mflux-community/mflux` sharing the spatial implementation and exploring integration into their "Related Projects" directory.
