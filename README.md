# OmniVidX: Omni-directional Video Generation in One Diffusion Model

<div align="center">
  <img src="assets/teaser.png" width="800px" alt="OmniVidX Teaser">
  <br>
</div>


## 📖 Overview

TODO

---

## 🚀 News
- **[2026/05/15]** Initial release of **OmniVidX**.
---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/OmniVidX.git
cd OmniVidX

# Create environment
conda create -n omnividx python=3.10
conda activate omnividx

# Install dependencies
pip install -r requirements.txt
```

## 🤖 Model Zoo

You can download the weights of backbone Wan2.1-T2V-14B from either **ModelScope** or **Hugging Face**.

**Option 1: ModelScope**
```bash
pip install modelscope
mkdir -p ./checkpoints/Wan-AI
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./checkpoints/Wan-AI/Wan2.1-T2V-14B
```

**Option 2: Hugging Face**
```bash
pip install "huggingface_hub[cli]"
mkdir -p ./models/Wan-AI
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./models/Wan-AI/Wan2.1-T2V-14B
```

Then, download checkpoints of **OmniVid-Intrinsic** and **OmniVid-Alpha** manually from Hugging Face or let the scripts auto-download them.

| Model Name | Link |
| :--- | :--- |
| OmniVid-Intrinsic | 🤗 Download |
| OmniVid-Alpha | 🤗 Download |

```bash
TODO
```
---

## 💻 Inference
我们使用yaml文件来配置推理参数，

```yaml
# OmniVid-Intrinsic

experiment_name: "omnivid_intrinsic_inference"   # Name for the output folder
mode: "R2AIN" # OmniVid-Intrinsic 支持的十五个任务之一

# Conditional Inputs (Optional)，根据不同任务来配置
inference_rgb_path: "./assets/R2AIN/rgb.mp4"
inference_albedo_path: null
inference_irradiance_path: null
inference_normal_path: null

# prompt 
prompt: ""

# Model Configuration
model:
  name: 'OmniVidIntrinsic' 
  params:
    model_paths: '["models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth","models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"]'
    resume_from_checkpoint: "checkpoints/omnivid_intrinsic.safetensors"
    lora_base_model: "dit"
    lora_target_modules: "self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2"
    lora_rank: 32
    lora_modalities: ["rgb","albedo","irradiance","normal"] # decoupled LoRA的名字
```

```yaml
# OmniVid-Alpha

experiment_name: "omnivid_alpha_inference"   

mode: "t2RPFB"

inference_rgb_path: null

inference_pha_path: null

inference_fgr_path: null

inference_bgr_path: null

prompt: "一只大熊猫直立坐着，双手捧着一根竹子，满足地咀嚼着。背景为：四川山区茂密、多雾的竹林，天空飘着蒙蒙细雨。"

model:
  name: 'OmniVidAlpha' 
  params:
    model_paths: '["models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth","models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"]'
    lora_base_model: "dit"
    lora_target_modules: "self_attn.q,self_attn.k,self_attn.v,self_attn.o,ffn.0,ffn.2"
    lora_rank: 32
    lora_modalities: ["com","pha","fgr","bgr"] 
    resume_from_checkpoint: "checkpoints/omnivid_alpha.safetensors"
```

然后，你可以使用配置好的yaml文件进行inference
```bash
# omnivid_alpha_inference
python scripts/inference_omnivid_alpha.py --config configs/omnivid_alpha_inference.yaml

# omnivid_intrinsic_inference
python scripts/inference_omnivid_intrinsic.py --config configs/omnivid_intrinsic_inference.yaml

```
## 🏋️ Training

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