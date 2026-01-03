"""
v1: cat on the frame dim depth and normal
""" 

import torch, types, copy,json
from typing import Optional, Union
from einops import reduce
from tqdm import tqdm
import torch.nn as nn
from ..trainers.util import DiffusionTrainingModule
from .util import BasePipeline, PipelineUnit, PipelineUnitRunner, ModelConfig, TeaCache, TemporalTiler_BCTHW
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

    def __init__(self, 
                 device="cuda", 
                 torch_dtype=torch.bfloat16, 
                 tokenizer_path=None,
                #  unit_names: list[str] = None
                 ):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None # type: ignore
        self.image_encoder: WanImageEncoder = None # type: ignore
        self.dit: WanModel = None # type: ignore
        self.vae: WanVideoVAE = None # type: ignore
        self.motion_controller: WanMotionControllerModel = None # type: ignore
        self.vace: VaceWanModel = None # type: ignore
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        #改回硬编码，发现pipe的代码还是必须要改动的
        self.units = [ # 硬编码
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ControlVideoEmbedder(),
        ]
       
        self.model_fn = model_fn_wan_video
        
    
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device) # type: ignore
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device) # type: ignore
        loader.load(module, lora, alpha=alpha)

        
    def training_loss(self, **inputs):
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        B, C_2, T, H, W = inputs["input_latents"].shape
        # 只对input_latents加噪声
        noised_latents = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        # cat cond部分
        latents = torch.cat([noised_latents, inputs["control_latents"]], dim=1) # b 3c(normal disparity rgb_cond) t h w
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        inputs["dit"].has_image_input = False
        inputs["latents"] = latents
        noise_pred = self.model_fn(**inputs, timestep=timestep) # model_fn are used to predict noise
        loss = torch.nn.functional.mse_loss(noise_pred[:,:C_2].float(), training_target[:,:C_2].float())
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
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn) # type: ignore
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    def vae_normal_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean") # H W C
        normal_unit = vae_output / (torch.norm(vae_output, dim=-1, keepdim=True) + 1e-6) # [-1,1] unit
        return normal_unit.permute(2,0,1) # C H W


    def vae_normal_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video_normal_unit = []
        for image in vae_output:
            normal_unit= self.vae_normal_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value)
            video_normal_unit.append(normal_unit)
        return torch.stack(video_normal_unit, dim=1)

    def robust_min_max_normalize(self,
        tensor: torch.Tensor, 
        quantiles: tuple[float, float] = (0.001, 0.99), 
        per_channel: bool = False, 
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Args:
            tensor: [H, W, 3]
            quantiles: (0.001, 0.99)
            per_channel: True
            eps: 1e-6
        Returns:
            [H, W, 3]
        """
        assert tensor.ndim == 3 and tensor.shape[2] == 3, f"输入张量形状应为 [H, W, 3]，但得到 {tensor.shape}"
        tensor_cache = tensor.clone()
        tensor = tensor.to(torch.float32)
        if per_channel:
            tensor_flat = tensor.permute(2, 0, 1).flatten(start_dim=1)
            min_vals = torch.nanquantile(tensor_flat, quantiles[0], dim=1)
            max_vals = torch.nanquantile(tensor_flat, quantiles[1], dim=1)
            min_vals = min_vals.view(1, 1, 3)
            max_vals = max_vals.view(1, 1, 3)

        else:
            min_vals = torch.nanquantile(tensor, quantiles[0])
            max_vals = torch.nanquantile(tensor, quantiles[1])
        denominator = max_vals - min_vals
        denominator = torch.where(denominator < eps, torch.tensor(eps, device=tensor.device), denominator)
        normalized_tensor = (tensor - min_vals) / denominator
        normalized_tensor = torch.clamp(normalized_tensor, 0, 1)
        normalized_tensor = normalized_tensor.to(tensor_cache.dtype)
        return normalized_tensor 
    def vae_disparity_output_to_video(self, vae_output, pattern="B C T H W"):
        # 这个处理不知道合不合理，mask先不加
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        assert vae_output.min() >= -1 and vae_output.max() <= 1, "disparity is not in [-1,1]"
        for i in range(vae_output.shape[0]):
            image = vae_output[i] # H W C
            image = self.robust_min_max_normalize(image)
            vae_output[i] = image * 2 - 1  # 这个是为了算loss的
        return vae_output.permute(3,0,1,2) # C T H W
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        local_model_path: str = "./models",
        skip_download: bool = False,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path, 调用以实现pipe的实例化
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
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype) # type: ignore
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(local_model_path, skip_download=skip_download, use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder") # type: ignore
        pipe.dit = model_manager.fetch_model("wan_video_dit") # type: ignore
        pipe.vae = model_manager.fetch_model("wan_video_vae") # type: ignore
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder") # type: ignore
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller") # type: ignore
        pipe.vace = model_manager.fetch_model("wan_video_vace") # type: ignore

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(local_model_path, skip_download=skip_download)
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
        # 只有inference的时候才会调用__call__()
        inputs_shared = kwargs.copy()
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
        C_2 = C * 2 
        for progress_id, timestep in enumerate(tqdm(scheduler_copy.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            if inputs_shared["latents"].shape[1] == C_2:
                inputs_shared["latents"] = torch.cat([inputs_shared["latents"], inputs_shared["control_latents"]], dim=1) # cond 不加噪声
            else:
                inputs_shared["latents"][:,C_2:] = inputs_shared["control_latents"] # 之后就是更新而非cat了，否则就是循环cat
            # Inference
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
        video_normal = self.vae.decode(inputs_shared["latents"][:, :C,], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_disparity = self.vae.decode(inputs_shared["latents"][:, C:C_2,], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_normal_unit= self.vae_normal_output_to_video(video_normal)
        video_disparity_loss = self.vae_disparity_output_to_video(video_disparity)
        video_dict[f"normal_unit"] = video_normal_unit # torch.tensor[c,t,h,w]
        video_dict[f"disparity_loss"] = video_disparity_loss # torch.tensor[c,t,h,w]
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
    # 适配cat on channel的逻辑
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "num_input_videos"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, num_input_videos):
        length = (num_frames - 1) // 4 + 1 
        noise = pipe.generate_noise((1, 16 * num_input_videos, length, height//8, width//8), seed=seed, rand_device=rand_device)
        return {"noise": noise}

class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    # 将normal和depth 分开encode到latent space
    def __init__(self):
        super().__init__(
            input_params=("normals", "disparities", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
    def process(self, pipe: WanVideoPipeline, normals, disparities, noise, tiled, tile_size, tile_stride):
        if normals is None and disparities is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        normals_video = pipe.preprocess_video(normals)
        normals_latents = pipe.vae.encode(normals_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        disparities_video = pipe.preprocess_video(disparities)
        disparities_latents = pipe.vae.encode(disparities_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = torch.cat([normals_latents, disparities_latents], dim=1) # cat 在 C 维度， first half is normal, second half is disparity
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
        
class WanVideoUnit_ControlVideoEmbedder(PipelineUnit):
    # 保留自己的逻辑吧
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device) # tiled: Bool 是否分块处理 tile_size: patch size tile_stride 步长
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_latents": control_latents}

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
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    if dit.has_image_input: # y 是 ctrl video latent 
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w) # cat在channel维度 # type: ignore
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
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

class WanTrainingModule_wan2_1_t2v_1_3b_control_normal_depth_train_po_v1(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True, # 默认false好，xiuyu的经验 # 还是改成true吧，不然训不下
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
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
        tokenizer_config = ModelConfig(path="checkpoints/Wan_txt_encoder_vae/google/umt5-xxl")
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", tokenizer_config=tokenizer_config, model_configs=model_configs,redirect_common_files=False)
        self.adapt_dit_patch_embedding(self.pipe.dit, 3)
        assert (self.pipe.dit.patch_embedding.weight.data[:,:16] == self.pipe.dit.patch_embedding.weight.data[:,32:]).to(torch.float32).mean() == 1, "patch_embedding weight not equal"
        self.adapt_dit_head_head(self.pipe.dit, 3)
        assert (self.pipe.dit.head.head.weight.data[:64,] == self.pipe.dit.head.head.weight.data[128:]).to(torch.float32).mean() == 1, "head.head weight not equal"
        
        # 在这里改dit.patch_embedding和dit.head.head


        # 我是真没招了，就在这里手动赋值吗，不追求极致的解耦了

        #-------start------#
        self.pipe.units = [ # 一定要加()，这样才实例化
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ControlVideoEmbedder()
        ]
        # self.pipe.dit.patch_embedding = torch.nn.Conv3d(64,1536,kernel_size=(1,2,2),stride=(1,2,2)) 
        # self.pipe.dit.head.head = torch.nn.Linear(1536, 128,bias=True) 
        # #-------end------#
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze all models first
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
                # Only unfreeze self_attn modules in DIT blocks
        trainable_params = 0
        total_params = 0
        trainable_modules = []
        
        for name, module in self.pipe.dit.named_modules():
            # 精确匹配self_attn模块，只匹配blocks中的self_attn
            if any(keyword in name for keyword in ["self_attn"]): # 
                module_params = 0
                for param in module.parameters():
                    param.requires_grad = True
                    module_params += param.numel()
                    trainable_params += param.numel()
                trainable_modules.append((name, module_params))
                print(f"Trainable: {name} ({module_params /1024/1024:.2f} MB)")
        
        # 统计所有参数
        for param in self.pipe.dit.parameters():
            total_params += param.numel()
        
        # 计算可训练参数比例
        trainable_ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        # 根据实际数据类型计算参数大小
        param_size_bytes = 0
        for param in self.pipe.dit.parameters():
            if param.requires_grad:
                # param_size_bytes += param.numel() * param.element_size() 不要乘字节数
                param_size_bytes += param.numel()
        print(f"\n=== Parameter Statistics ===")
        print(f"Total DIT parameters: {total_params/1024/1024/1024:.2f} B") 
        print(f"Trainable parameters: ({trainable_ratio:.2f}%)")
        print(f"Trainable modules: {len(trainable_modules)}")
        print(f"Trainable parameters size: {param_size_bytes / 1024 / 1024:.2f} MB")
        print(f"===========================\n")

        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model), # 取出要加lora的模块，在这里就等价于self.pipe.dit
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
    def adapt_dit_patch_embedding(self, model: nn.Module, multiplier: int = 3) -> nn.Module:
        """
        adapt patch_embedding，使其输入通道数变为原来的 n 倍。
        通过复制拼接的方式初始化新权重。
        Args:
            model (nn.Module): 原始的 WanModel 模型。
            multiplier (int): 输入通道的倍数，默认为 3。

        Returns:
            nn.Module: 修改后的模型。
        """
        print(f"--- 正在adapt patch_embedding以接受 {multiplier} 倍通道数... ---")
        
        # 定位到要修改的层
        try:
            orig_layer = model.patch_embedding
            layer_name = "patch_embedding"
            if not isinstance(orig_layer, (nn.Conv2d, nn.Conv3d)):
                raise TypeError(f"{layer_name} 不是一个卷积层。")
        except AttributeError:
            print(f"error: patch_embedding not found")
            return model
        orig_weight = orig_layer.weight.data
        orig_bias = orig_layer.bias.data if orig_layer.bias is not None else None
    
        NewConvLayer = type(orig_layer) # 这行代码要学啊，很智能
        new_layer = NewConvLayer(
            in_channels=orig_layer.in_channels * multiplier,
            out_channels=orig_layer.out_channels,
            kernel_size=orig_layer.kernel_size, # type: ignore
            stride=orig_layer.stride, # type: ignore
            padding=orig_layer.padding, # type: ignore
            dilation=orig_layer.dilation, # type: ignore
            groups=orig_layer.groups,
            bias=(orig_bias is not None)
        )

        # 拼接新权重 (dim=1 是 in_channels 维度)
        new_weight = torch.cat([orig_weight] * multiplier, dim=1)
        new_layer.weight.data.copy_(new_weight)
        if orig_bias is not None:
            new_layer.bias.data.copy_(orig_bias) # type: ignore
        setattr(model, layer_name, new_layer)
        print(f"✅ {layer_name}.in_channels 已从 {orig_layer.in_channels} 变为 {new_layer.in_channels}")
        return model
        
    
    def adapt_dit_head_head(self,model: nn.Module, multiplier: int = 3) -> nn.Module:
        """
        adapt dit.head.head，使其输出特征数变为原来的 n 倍。
        通过复制拼接的方式初始化新权重。

        Args:
            model (nn.Module): 原始的 WanModel 模型。
            multiplier (int): 输出特征的倍数，默认为 3。

        Returns:
            nn.Module: 修改后的模型。
        """
        print(f"--- 正在adapt head.head以生成 {multiplier} 倍通道数的结果... ---")
        try:
            orig_layer = model.head.head # type: ignore
            parent_module = model.head
            layer_name = "head"
            if not isinstance(orig_layer, nn.Linear):
                raise TypeError(f"model.head.{layer_name} 不是一个线性层。")
        except AttributeError:
            print(f"错误：在模型中找不到 'head.head' 属性。")
            return model
        orig_weight = orig_layer.weight.data
        orig_bias = orig_layer.bias.data if orig_layer.bias is not None else None
        new_layer = nn.Linear(
            in_features=orig_layer.in_features,
            out_features=orig_layer.out_features * multiplier,
            bias=(orig_bias is not None)
        )
        new_weight = torch.cat([orig_weight] * multiplier, dim=0)
        new_layer.weight.data.copy_(new_weight)
        if orig_bias is not None:
            new_bias = torch.cat([orig_bias] * multiplier, dim=0)
            new_layer.bias.data.copy_(new_bias)
        setattr(parent_module, layer_name, new_layer)
        print(f"✅ head.head.out_features 已从 {orig_layer.out_features} 变为 {new_layer.out_features}")
        
        return model

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_videos": [
                data["normals"],
                data["disparities"],
            ], 
            "normals": data["normals"],
            "disparities": data["disparities"], # b c t h w，需要仔细确认加载是否正确
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
        }
        # extra 更新一个来适应多输入,用于WanVideoUnit_NoiseInitializer
        inputs_shared["num_input_videos"] = len(inputs_shared["input_videos"])

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