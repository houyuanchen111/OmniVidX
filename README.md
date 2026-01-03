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



```bash
# Clone the repository
git clone https://github.com/your-username/OmniVidX.git
cd OmniVidX

# Create environment
conda create -n omnividx python=3.10
conda activate omnividx

# Install dependencies (requires CUDA 12.1+)
pip install -r requirements.txt
```

## 🤖 Model Zoo

Download checkpoints manually from Hugging Face or let the scripts auto-download them.

| Model Name        | Modalities     | Task Focus                      | Backbone     | Size | Link |
|------------------|----------------|----------------------------------|--------------|------|------|
| OmniVid-Intrinsic | R, A, I, N     | Inverse Rendering, Relighting    | Wan2.1-14B   | 28GB | 🤗 Download |
| OmniVid-Alpha     | R, P, F, B     | Matting, Composition             | Wan2.1-14B   | 28GB | 🤗 Download |

**Note:**

- R = RGB  
- A = Albedo  
- I = Irradiance  
- N = Normal  
- P = Alpha  
- F = Foreground  
- B = Background  

---

## 💻 Inference

We provide a unified inference script `inference.py` supporting **30 sub-tasks** via the `--task` argument.

### 1. Inverse Rendering  
**RGB → Albedo + Irradiance + Normal**

```bash
python inference.py \
  --model_path checkpoints/OmniVid-Intrinsic \
  --task R2AIN \
  --input_video assets/input_rgb.mp4 \
  --output_dir outputs/inverse_rendering \
  --resolution 480 640 \
  --frames 21
```
💻 Inference

We provide a unified inference script inference.py supporting 30 sub-tasks via the --task argument.

1. Inverse Rendering

RGB → Albedo + Irradiance + Normal

python inference.py \
  --model_path checkpoints/OmniVid-Intrinsic \
  --task R2AIN \
  --input_video assets/input_rgb.mp4 \
  --output_dir outputs/inverse_rendering \
  --resolution 480 640 \
  --frames 21

2. Video Matting

RGB → Alpha + Foreground + Background

python inference.py \
  --model_path checkpoints/OmniVid-Alpha \
  --task R2PFB \
  --input_video assets/human_dance.mp4 \
  --prompt "a dancer in a studio" \
  --output_dir outputs/matting

3. Text-to-All Generation

Generate all modalities from scratch using a text prompt.

python inference.py \
  --model_path checkpoints/OmniVid-Intrinsic \
  --task t2RAIN \
  --prompt "A golden statue in a dark cave, cinematic lighting" \
  --seed 42

4. Custom Combination (Advanced)

Manually specify input and output modalities for flexible editing.

# Example: Retexturing
# Keep Normal + Irradiance, generate RGB + Albedo
python inference.py \
  --model_path checkpoints/OmniVid-Intrinsic \
  --condition_modalities normal irradiance \
  --target_modalities rgb albedo \
  --input_video assets/source.mp4

📦 Gradio Demo

Launch an interactive web demo for side-by-side visualization:

python app.py --server_port 7860

🏋️ Training

To train OmniVidX on your own dataset, format the data as follows:

dataset/
├── videos/
│   ├── 00001_rgb.mp4
│   ├── 00001_albedo.mp4
│   ├── 00001_normal.mp4
│   └── ...
└── meta.json   # prompts and file paths

Running Training

Training uses Accelerate and enables Stochastic Condition Masking (SCM) by default.

accelerate launch train.py \
  --config configs/train_intrinsic.yaml \
  --pretrained_model_path Wan2.1-T2V-14B \
  --data_path ./dataset \
  --output_dir ./checkpoints/my_experiment \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --use_sc_masking True \
  --lora_rank 32

📊 Citation

If you find this work useful, please cite:

@inproceedings{omnividx2026,
  title     = {OmniVidX: Omni-directional Video Generation in One Diffusion Model},
  author    = {Anonymous Author(s)},
  booktitle = {SIGGRAPH},
  year      = {2026}
}

📝 Acknowledgements

This codebase is built upon Wan2.1 and Diffusers.
We thank the authors for their open-source contributions.