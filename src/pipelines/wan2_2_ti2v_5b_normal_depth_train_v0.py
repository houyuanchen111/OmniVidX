"""
v3: cat on the frame dim depth and normal, different timestep
""" 

import torch, types, copy,json,random
from typing import Optional, Union
from einops import reduce
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from ..trainers.util import DiffusionTrainingModule
from .util import BasePipeline, PipelineUnit, PipelineUnitRunner, ModelConfig, TeaCache, TemporalTiler_BCTHW, vae_normal_output_to_video, vae_disparity_and_depth_output_to_video
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample 
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader



class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_ControlVideoEmbedder(),
            WanVideoUnit_ClipEncoder(),
        ]
        
        self.model_fn = model_fn_wan_video
    

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        B, C, T, H, W = inputs["input_latents"].shape
        inputs["input_latents"] = torch.cat([inputs["input_latents"], inputs["control_latents"]], dim=2) # 前半是tgt，后半是cond（src）
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        inputs["latents"][:,:,T:] = inputs["control_latents"] # cond 不加噪声
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        loss = torch.nn.functional.mse_loss(noise_pred[:,:,:T].float(), training_target[:,:,:T].float()) # 有用的只有前一半
        loss = loss * self.scheduler.training_weight(timestep)
        return loss


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        **kwargs 
    ):
        
        input_data = kwargs.copy()
        inputs_posi = {"prompt": input_data["prompt"]}
        inputs_nega = {}
        inference_params = {
            "prompt": "Output a video that assigns each 3D location in the world a consistent color.",
            "negative_prompt": "",
            "seed": 42,
            "num_inference_steps": 50,
            "cfg_scale": 5.0,
            "cfg_merge": False,
            "denoising_strength": 1.0,
            "tiled": True,
            "tile_size": [30, 52],
            "tile_stride": [15, 26],
        } 
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "height": input_data["height"],
            "width": input_data["width"],
            "num_frames": input_data["num_frames"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.device,
            "cfg_merge": False,
            "vace_scale": 1,
            "clip_feature": None,
            "y": None,
            "is_inference": True,
        }
        inputs_shared.update(inference_params)
        if random.random() < 0.5:
            inputs_shared["input_videos"] = input_data.get("normals", None)
            inputs_shared["normals"] = input_data.get("normals", None)
            inputs_shared["input_image"] = input_data.get("ref_normal", None)
            inputs_shared["context_index"] = torch.tensor([[0]], device=self.device) 
        else:
            inputs_shared["input_videos"] = input_data.get("depths", None)
            inputs_shared["depths"] = input_data.get("depths", None)
            inputs_shared["input_image"] = input_data.get("ref_depth", None)
            inputs_shared["context_index"] = torch.tensor([[1]], device=self.device)
        inputs_shared["batch_size"] = inputs_shared["input_videos"].shape[0]
        inputs_shared["control_video"] = input_data["rgbs"]


        num_inference_steps = inputs_shared.pop("num_inference_steps", 50)
        denoising_strength = inputs_shared.get("denoising_strength", 1.0)
        sigma_shift = inputs_shared.get("sigma_shift", 5.0)
        prompt = inputs_shared.pop("prompt", "") # 这个需要pop，避免重复输入
        tea_cache_l1_thresh = inputs_shared.pop("tea_cache_l1_thresh", None)
        tea_cache_model_id = inputs_shared.pop("tea_cache_model_id", "")
        negative_prompt = inputs_shared.pop("negative_prompt", "")
        vace_reference_image = inputs_shared.get("vace_reference_image", None)
        cfg_scale = inputs_shared.get("cfg_scale", 5.0)
        cfg_merge = inputs_shared.get("cfg_merge", False)
        tiled = inputs_shared.get("tiled", True)
        tile_size = inputs_shared.get("tile_size", (30,52))
        tile_stride = inputs_shared.get("tile_stride", (15,26))
        # Scheduler, here is the flow matching scheduler
        scheduler_copy = copy.deepcopy(self.scheduler)
        scheduler_copy.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        

        for unit in self.units: # type: ignore
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega) # type: ignore

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        B, C, T, H, W = inputs_shared["control_latents"].shape
        for progress_id, timestep in enumerate(tqdm(scheduler_copy.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            inputs_shared["latents"][:,:,T:] = inputs_shared["control_latents"] # 每个循环cond都是clean的
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep) # nega prompt的噪声
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega) # cfg生效的核心代码
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = scheduler_copy.step(noise_pred, scheduler_copy.timesteps[progress_id], inputs_shared["latents"]) # 注意！ 推理的时候cat cond一定要写在循环的里面，否则会破坏cond！

       
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        video_dict = {}
        video_dict["context_index"] = inputs_shared["context_index"]
        if video_dict["context_index"].item()== 0:
            video_normal = self.vae.decode(inputs_shared["latents"][:,:,:T], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_normal_unit= vae_normal_output_to_video(video_normal)
            video_dict[f"normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]
        if video_dict["context_index"].item() == 1:
            video_depth = self.vae.decode(inputs_shared["latents"][:,:,:T], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video_depth_loss = vae_disparity_and_depth_output_to_video(video_depth)
            video_dict[f"depth_loss"] = video_depth_loss # torch.tensor[c,t,h,w]
    
        return video_dict


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        """
        本质上是支持了batch输入
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
    """
    correct one bug: noises shoule be different with in a batch
    hard code the length of noise: length * 2
    """

    def __init__(self):
        super().__init__(input_params=("batch_size", "height", "width", "num_frames", "seed", "rand_device"))

    def process(self, pipe: WanVideoPipeline, batch_size, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (batch_size, pipe.vae.model.z_dim, length * 2, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise} 

class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image)[:,:,None,:,:]
        z = pipe.vae.encode(image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z # 替换latents的首帧
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}

class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    # 将normal和depth 分开encode到latent space
    def __init__(self):
        super().__init__(
            input_params=("input_videos", "noise", "tiled", "tile_size", "tile_stride", "is_inference"),
            onload_model_names=("vae",)
        )
    def process(self, pipe: WanVideoPipeline, input_videos, noise, tiled, tile_size, tile_stride, is_inference,):
        if is_inference:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        video = pipe.preprocess_video(input_videos)
        input_latents = pipe.vae.encode(video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        assert input_latents.shape[1] == noise.shape[1], "input_latents.shape[1] != noise.shape[1]"
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}

class WanVideoUnit_ClipEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "height", "width", "context_index"),
            onload_model_names=("vae")
        )
    def preprocess_for_clip(self, img):
        # img: B, 3, 1, H, W, value in [-1, 1]
        # 1. squeeze掉第3维
        assert img.ndim == 5, "img.ndim != 5"
        if not isinstance(img, torch.Tensor):
            raise ValueError("img is not a torch.Tensor")
        img = img.squeeze(2)  # 变成 B, 3, H, W
        return img  # B, 3, H, W, 已经可以输入clip

    def process(self, pipe: WanVideoPipeline, control_video, height, width, context_index):
        if control_video.ndim == 4:
            # 如果是B, 3, H, W，增加一帧维度
            reference_image = control_video.unsqueeze(2)  # B, 3, 1, H, W
        elif control_video.ndim == 5:
            reference_image = control_video[:,:,0:1,:,:] # extract the first frame (keep dim)
        else:
            raise ValueError("control_video must be 4 or 5 dims, got {}".format(control_video.ndim))
        assert reference_image.ndim == 5, "reference_image.ndim != 5"
        assert reference_image.min() >= -1.01 and reference_image.max() <= 1.01, "reference_image.min() < -1 or reference_image.max() > 1, not suit for vae" # float16 的精度问题
        pipe.load_models_to_device(["vae"])
        reference_image = pipe.preprocess_image(reference_image)
        reference_image = self.preprocess_for_clip(reference_image)
        clip_feature = pipe.image_encoder.encode_image([reference_image])
        return {"clip_feature": clip_feature} # clip 首帧

class WanVideoUnit_ControlVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride",),
            onload_model_names=("vae")
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_latents": control_latents} 

# class WanVideoUnit_FunControl(PipelineUnit):
#     def __init__(self):
#         super().__init__(
#             input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride",),
#             onload_model_names=("vae")
#         )

#     def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride):
#         if control_video is None:
#             return {}
#         pipe.load_models_to_device(self.onload_model_names)
#         control_video = pipe.preprocess_video(control_video)
#         control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
#         control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
#         # y 初始化为0
#         y = torch.zeros_like(control_latents,device=pipe.device,dtype=pipe.torch_dtype)
#         y = torch.cat([control_latents, y], dim=1) # cat on channel dim
#         assert y.shape[1] == 32, "y.shape[1] != 32"
#         return {"y": y} # 暂时不用clip

class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("normals", "disparities", "height", "width", "context_index"),
            onload_model_names=("vae")
        )
    def preprocess_for_clip(self, img):
        # img: B, 3, 1, H, W, value in [-1, 1]
        # 1. squeeze掉第3维
        assert img.ndim == 5, "img.ndim != 5"
        if not isinstance(img, torch.Tensor):
            raise ValueError("img is not a torch.Tensor")
        img = img.squeeze(2)  # 变成 B, 3, H, W
       

        return img  # B, 3, H, W, 已经可以输入clip

    def process(self, pipe: WanVideoPipeline, normals, disparities, height, width, context_index):
        # extract the first frame
        if context_index.item() == 0:
            reference_image = normals[:,:,0][:,:,None,:,:]
        elif context_index.item() == 1:
            reference_image = disparities[:,:,0][:,:,None,:,:]
        else:
            return {} # for inference
        assert reference_image.ndim == 5, "reference_image.ndim != 5"
        assert reference_image.min() >= -1.01 and reference_image.max() <= 1.01, "reference_image.min() < -1 or reference_image.max() > 1, not suit for vae" # float16 的精度问题
        pipe.load_models_to_device(["vae"])
        reference_latents = pipe.preprocess_video(reference_image)
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)

        reference_image = pipe.preprocess_image(reference_image)
        reference_image = self.preprocess_for_clip(reference_image)
        clip_feature = pipe.image_encoder.encode_image([reference_image])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}




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
        


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None, # type: ignore
    vace: VaceWanModel = None, # type: ignore
    latents: torch.Tensor = None, # type: ignore
    timestep: torch.Tensor = None, # type: ignore
    context: torch.Tensor = None, # type: ignore
    clip_feature: Optional[torch.Tensor] = None, # type: ignore
    y: Optional[torch.Tensor] = None, # type: ignore
    reference_latents = None, # type: ignore
    vace_context = None, # type: ignore
    vace_scale = 1.0, # type: ignore
    tea_cache: TeaCache = None, # type: ignore
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None, # type: ignore
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None, # type: ignore
    context_index = None, # type: ignore 用于区分不同模态
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep)) # dit.time_embedding: mlp -> silu -> mlp, where mlp is AutoWrappedLinear, which is designed for save varm
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context) # mlp -> silu -> mlp # b l c； c 的维度要和dit的channel对上，L没关系

    x = latents # noise
    B = x.shape[0]
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    if dit.has_image_input: # y 是 ctrl video latent 
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w) # cat在channel维度 # type: ignore
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    if context_index is not None:
        input_context_emb = dit.context_embedding(context_index.long()).repeat(B, 1, 1)
        context = torch.cat([input_context_emb, context], dim=1) # b l + 1 d
    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input) # type: ignore

    
    # Reference image 
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1) # cat 在L维度
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device) # 3d rope, 记录帧间时序
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint( # type: ignore
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint( # type: ignore
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:] # type: ignore
        f -= 1
    x = dit.unpatchify(x, (f, h, w)) # type: ignore just a rearrangement
    return x

class WanTrainingModule_wan2_2_ti2v_5b_normal_depth_train_v0(DiffusionTrainingModule): 
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True, # 默认false好，xiuyu的经验 # 还是改成true吧，不然训不下
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        tokenizer_config = ModelConfig(path="checkpoints/Wan2.1-T2V-1.3B/google/umt5-xxl")
        # 检测可用的设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, tokenizer_config=tokenizer_config, model_configs=model_configs,redirect_common_files=False) # 加载预训练权重
      
        if resume_from_checkpoint is not None:
            state_dict = load_file(resume_from_checkpoint) # safetensors 不需要map_location
            prefix = "pipe." # 去掉不想要的前缀
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            try:
                missing_keys, unexpected_keys = self.pipe.load_state_dict(state_dict, strict=False)
                loaded_keys = [k for k in state_dict.keys() if k not in unexpected_keys]
                assert unexpected_keys == [], "unexpected_keys not empty"
                if loaded_keys: # 成功加载的
                    print("已加载的模块/参数键名:")
                    for key in loaded_keys:
                        print(f"-{key}")

                if missing_keys: # 存在于模型中，但不在state_dict中，大概率是冻住的层
                    print(f"\n⚠️ 模型中存在，但检查点中缺失的键 (共 {len(missing_keys)} 个):")
                    print("   (这些参数将保留其原始预训练权重，请检查是否需要冻结)")
                    for key in missing_keys:
                        print(f"-{key}")
                
                if unexpected_keys: # 存在于state_dict,但不在模型中
                    print(f"\n❌ 警告：检查点中存在，但模型中没有对应位置的键 (共 {len(unexpected_keys)} 个):")
                    print("   (这些参数在加载时被忽略了，请检查模型架构是否匹配)")
                    for key in unexpected_keys:
                        print(f"-{key}")

                print(f"\n从checkpoint {resume_from_checkpoint} 恢复训练") # 您原来的打印语句
            except Exception as e:
                print(f"从checkpoint {resume_from_checkpoint} 恢复训练失败: {e}")
            

        #-------start------#
        self.pipe.units = [ # 一定要加()，这样才实例化
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_ControlVideoEmbedder(),
            # WanVideoUnit_ClipEncoder(), 没有预训练权重，暂时不强求
        ]
  
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze all models first
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        self.pipe.dit.context_embedding = torch.nn.Embedding(16, 3072) # follow diffusion renderer
        self.pipe.dit.context_embedding.requires_grad = True
        # 放开 blocks 和 head
        for param in self.pipe.dit.blocks.parameters():
            param.requires_grad = True
       

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.pipe.dit.parameters())
        print(f"Trainable: {trainable_params/1024/1024:.2f} M / {total_params/1024/1024:.2f} M")

        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model), # 取出要加lora的模块，在这里   就等价于self.pipe.dit
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        # 
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "height": data["height"],
            "width": data["width"],
            "num_frames": data["num_frames"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "clip_feature": None,
            "y": None,
            "is_inference": False,
        }
        # extra 更新一个来适应多输入,用于WanVideoUnit_NoiseInitializer
        if random.random() < 0.5:
            inputs_shared["input_videos"] = data.get("normals", None)
            inputs_shared["normals"] = data.get("normals", None)
            inputs_shared["input_image"] = data.get("ref_normal", None)
            inputs_shared["context_index"] = torch.tensor([[0]], device=self.pipe.device) 
        else:
            inputs_shared["input_videos"] = data.get("depths", None)
            inputs_shared["depths"] = data.get("depths", None)
            inputs_shared["input_image"] = data.get("ref_depth", None)
            inputs_shared["context_index"] = torch.tensor([[1]], device=self.pipe.device)
        inputs_shared["batch_size"] = inputs_shared["input_videos"].shape[0]
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "control_video" and "control_video" not in data and "rgbs" in data:
                inputs_shared[extra_input] = data["rgbs"]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        if self.pipe.units is not None: # 防止是None
            for unit in self.pipe.units:
                inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega) # type: ignore
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss