import argparse
import glob
import os
import sys
from datetime import datetime
import torch
import torch.nn.functional as F
import torchvision
from torchvision.io import write_video
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.registry import (
    MODEL_REGISTRY,
)

now = datetime.now().strftime("%Y%m%d_%H%M%S")


def _tensor2video(tensor: torch.Tensor, file_path: str, fps: int = 15):
    """将张量保存为视频文件。期望张量为 (C,T,H,W) 且范围 [0,1]。"""
    assert tensor.ndim == 4, "tensor must be [c,t,h,w]"
    tensor = tensor.detach().cpu()
    video_tensor = tensor.permute(1, 2, 3, 0)  # c t h w -> t h w c
    video_tensor = (video_tensor.clamp(0, 1) * 255).to(torch.uint8)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        from torchvision.io import write_video
        write_video(
            filename=file_path,
            video_array=video_tensor,
            fps=fps,
            video_codec='h264',
        )
    except Exception as e:
        print(f"视频保存失败: {file_path}, error: {e}")

def _load_mp4_as_video_tensor(
    mp4_path: str,
    *,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    video, _, _ = torchvision.io.read_video(mp4_path, pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Empty video: {mp4_path}")
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Unexpected video shape {tuple(video.shape)} for {mp4_path}; expected [T,H,W,3]")

    video = video.to(torch.float32) / 255.0
    video = video.permute(0, 3, 1, 2)  # T,H,W,3 -> T,3,H,W

    if (video.shape[-2] != height) or (video.shape[-1] != width):
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)

    T = video.shape[0]
    if T >= num_frames:
        video = video[:num_frames]
    else:
        pad = video[-1:].repeat(num_frames - T, 1, 1, 1)
        video = torch.cat([video, pad], dim=0)

    video = video.permute(1, 0, 2, 3).contiguous()  # T,3,H,W -> 3,T,H,W
    video = video * 2.0 - 1.0
    video = video.unsqueeze(0).to(device=device, dtype=dtype)  # 1,3,T,H,W
    return video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/omnivid_alpha_inference.yaml", help='YAML 配置文件路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # 设置环境变量
    if 'environment_variables' in config and config['environment_variables']:
        print("--- 正在根据配置文件设置环境变量 ---")
        for key, value in config['environment_variables'].items():
            str_value = str(value)
            os.environ[key] = str_value
            print(f"✅已设置: {key} = {str_value}")
        print("------------------------------------")

    # 构建模型
    mode = config["mode"]
    inference_rgb_path = config.get("inference_rgb_path", None)
    inference_pha_path = config.get("inference_pha_path", None)
    inference_fgr_path = config.get("inference_fgr_path", None)
    inference_bgr_path = config.get("inference_bgr_path", None)
    prompt = config.get("prompt", "")
    model_class = MODEL_REGISTRY[config['model']['name']]
    model = model_class(**config['model']['params'])
    print(f"✅ 模型 '{config['model']['name']}' 已创建")
    # 设置模型为评估模式
    model.eval()
    print("✅ 模型已设置为评估模式")

    # 设置推理参数
    inference_params = {
        "negative_prompt":"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "seed": 1,
        "num_inference_steps": 50,
        "cfg_scale": 1.0,  # Matte 任务通常不需要 CFG
        "cfg_merge": False,
        "height": 432,
        "width": 768,
        "num_frames": 21,  # 改为21帧
        "denoising_strength": 1.0,
        "tiled": True,
        "tile_size": [30, 52],
        "tile_stride": [15, 26],
        "is_inference": True,
        "training_mode": mode,
        "inference_rgb": None,
        "inference_pha": None,
        "inference_fgr": None,
        "inference_bgr": None,
    }    

    if inference_rgb_path:
        rgb_video = _load_mp4_as_video_tensor(
            inference_rgb_path,
            num_frames=inference_params["num_frames"],
            height=inference_params["height"],
            width=inference_params["width"],
            device=torch.device(model.pipe.device),
            dtype=model.pipe.torch_dtype,
        )
        inference_params["inference_rgb"] = rgb_video
    if inference_pha_path:
        pha_video = _load_mp4_as_video_tensor(
            inference_pha_path,
            num_frames=inference_params["num_frames"],
            height=inference_params["height"],
            width=inference_params["width"],
            device=torch.device(model.pipe.device),
            dtype=model.pipe.torch_dtype,
        )
        inference_params["inference_pha"] = pha_video
    if inference_fgr_path:
        fgr_video = _load_mp4_as_video_tensor(
            inference_fgr_path,
            num_frames=inference_params["num_frames"],
            height=inference_params["height"],
            width=inference_params["width"],
            device=torch.device(model.pipe.device),
            dtype=model.pipe.torch_dtype,
        )
        inference_params["inference_fgr"] = fgr_video
    if inference_bgr_path:
        bgr_video = _load_mp4_as_video_tensor(
            inference_bgr_path,
            num_frames=inference_params["num_frames"],
            height=inference_params["height"],
            width=inference_params["width"],
            device=torch.device(model.pipe.device),
            dtype=model.pipe.torch_dtype,
        )
        inference_params["inference_bgr"] = bgr_video

    output_path = f"results/{config['experiment_name']}/inference_results_{now}"
    os.makedirs(output_path, exist_ok=True)
    print(f"✅ 输出目录: {output_path}")

    # 开始推理
    print("\n--- 开始推理！ ---")
    inference_params["prompt"] = [
        prompt,
        prompt,
        prompt,
        prompt
    ]
    video_dict = model.pipe(**inference_params)
    if not inference_rgb_path:
        rgb_gen = video_dict["rgb"]  # [C, T, H, W]
        _tensor2video(rgb_gen * 0.5 + 0.5, f"{output_path}/inference/rgb_gen.mp4")
    if not inference_pha_path:
        pha_gen = video_dict["pha"]  # [C, T, H, W]
        _tensor2video(pha_gen * 0.5 + 0.5, f"{output_path}/inference/pha_gen.mp4")
    if not inference_fgr_path:
        fgr_gen = video_dict["fgr"]  # [C, T, H, W]
        _tensor2video(fgr_gen * 0.5 + 0.5, f"{output_path}/inference/fgr_gen.mp4")
    if not inference_bgr_path:
        bgr_gen = video_dict["bgr"]  # [C, T, H, W]
        _tensor2video(bgr_gen * 0.5 + 0.5, f"{output_path}/inference/bgr_gen.mp4")
                
    print(f"推理&保存全部完成！ 路径: {output_path}")
    
    

if __name__ == "__main__":
    main()