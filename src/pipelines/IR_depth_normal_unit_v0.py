"""
unit才是核心的，每个训练配置不同的代码
定义自己的Unit, 不用继承原来的unit，继承PipelineUnit 即可
"""
import torch

from .util import WanVideoPipeline
from .util import PipelineUnit

class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        """
        是否要引逐帧处理？
        """
        if isinstance(height, torch.Tensor):
            height = height[0].item()
        if isinstance(width, torch.Tensor):
            width = width[0].item()
        if isinstance(num_frames, torch.Tensor):
            num_frames = num_frames[0].item()
        result = pipe.check_resize_height_width(height, width, num_frames)
        if len(result) == 2:
            height, width = result
        else:
            height, width, num_frames = result
        return {"height": height, "width": width, "num_frames": num_frames}

class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "num_input_videos"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, num_input_videos):
        length = (num_frames - 1) // 4 + 1 
        noise = pipe.generate_noise((1, 16 * num_input_videos, length, height//8, width//8), seed=seed, rand_device=rand_device)
        return {"noise": noise}

class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    # 用vae将video(& ref img) encode 到 latent space
    def __init__(self):
        super().__init__(
            input_params=("input_videos", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_videos: list, noise, tiled, tile_size, tile_stride):
        if input_videos is None:
            return {"latents": noise} # 如果没有input_video，则直接从noise出发; noise和latent的shape相同
        pipe.load_models_to_device(["vae"])
        input_latents_list = []
        for input_video in input_videos:
            #input_video = pipe.preprocess_video(input_video) # list ot Image to tensor
            input_video = pipe.preprocess_video(input_video) # 确保tensor正确的dtype以及正确的device
            input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents_list.append(input_latents)
        input_latents = torch.cat(input_latents_list, dim=1) # cat 在C
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}

class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True, # 
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=str(pipe.device)) # 如果device本身就已经是str了，str()会返回原值
        return {"context": prompt_emb} # 我觉得这个可以预处理保存成pth
        
class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        # control_video = pipe.preprocess_video(control_video) # Image 2 tensor, 现在这部分功能转移至dataloader
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device) # tiled: Bool 是否分块处理 tile_size: patch size tile_stride 步长
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if clip_feature is None or y is None:
            B = control_latents.shape[0] # 支持batch size
            clip_feature = torch.zeros((B, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((B, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -16:]
        y = torch.concat([control_latents, y], dim=1) # Tensor(control video, y:0)
        return {"clip_feature": clip_feature, "y": y}

