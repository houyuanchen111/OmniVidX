# OmniVidX: Omni-directional Video Generation in One Diffusion Model

<div align="center">
  <img src="assets/teaser_github.gif" width="800px" alt="OmniVidX Teaser">
  <br>
  <strong>SIGGRAPH 2026 (Anonymous Submission)</strong>
</div>

---

## 📖 Introduction

**OmniVidX** is a unified framework enabling **cross-modal any-to-any video generation** within a **single diffusion model**.  
Built upon the **Wan2.1-T2V-14B** backbone, it unifies diverse video generation and editing tasks into a **shared multimodal latent space**, including:

- Text-to-video  
- Inverse rendering  
- Forward rendering  
- Video matting  
- Relighting  

### Core Contributions

- **Stochastic Condition Masking (SCM)**  
  Enables dynamic switching between condition and target modalities during inference.

- **Decoupled Gated LoRA (DGL)**  
  Preserves native VDM priors while adapting efficiently to new modalities.

- **Cross-Modal Self-Attention (CMSA)**  
  Enforces pixel-aligned consistency across generated modality streams.

This repository contains the **official PyTorch implementation**, **pretrained checkpoints**, and **inference scripts** for:

- **OmniVid-Intrinsic**: RGB ↔ Albedo, Irradiance, Normal  
- **OmniVid-Alpha**: RGB ↔ Foreground, Background, Alpha  

---

## 🚀 News

- **[2026/05/20]** Release of the **OmniVid-Alpha** checkpoint (Video Matting focus)  
- **[2026/05/15]** Initial release of **OmniVid-Intrinsic** and inference code  
- **[2026/05/01]** Paper submitted to **SIGGRAPH 2026**

---

## 🛠️ Installation

We recommend using **Anaconda** to manage the environment.  
The codebase is tested on **Python 3.10+** and **PyTorch 2.4+**.

```bash
# Clone the repository
git clone https://github.com/your-username/OmniVidX.git
cd OmniVidX

# Create environment
conda create -n omnividx python=3.10
conda activate omnividx

# Install dependencies (requires CUDA 12.1+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate imageio[ffmpeg] opencv-python xformers
pip install -r requirements.txt
