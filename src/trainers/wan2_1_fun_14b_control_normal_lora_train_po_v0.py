import json,torch
from pathlib import Path
import warnings
from tqdm import tqdm
from PIL import Image
from typing import Optional
from torchvision.transforms.functional import to_tensor
from accelerate import Accelerator
import torch.nn.functional as F

from ..trainers.util import Callback, Trainer, DiffusionTrainingModule
from ..pipelines.util import ModelConfig
from ..pipelines.util import WanVideoPipeline
from ..pipelines.wan2_1_fun_14b_control_normal_lora_train_v0 import WanVideoUnit_ShapeChecker   
from ..pipelines.wan2_1_fun_14b_control_normal_lora_train_v0 import WanVideoUnit_NoiseInitializer
from ..pipelines.wan2_1_fun_14b_control_normal_lora_train_v0 import WanVideoUnit_InputVideoEmbedder
from ..pipelines.wan2_1_fun_14b_control_normal_lora_train_v0 import WanVideoUnit_PromptEmbedder
from ..pipelines.wan2_1_fun_14b_control_normal_lora_train_v0 import WanVideoUnit_FunControl # 一共五个模块

# 下面两个callback是需要定制化的，便于控制
class ModelCheckpointCallback_wan2_1_fun_14b_control_normal_lora_train_po_v0(Callback):
    """
    checkpoint callback
    """
    def __init__(self, output_path, remove_prefix_in_ckpt=None):
        self.output_path = Path(output_path)
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt

    def _save_gif(self, images: Optional[list[Image.Image]], output_gif_path: str):
        if images is None:
            raise ValueError("images is None")
        if len(images) == 0:
            raise ValueError("images is empty")
        if not isinstance(images[0], Image.Image):
            raise ValueError("images is not a list of Image.Image")
        print(f"正在将 {len(images)} 帧图像保存为GIF: {output_gif_path}")
        images[0].save(
        output_gif_path,
        save_all=True,          
        append_images=images[1:], #
        duration=67,
        loop=0)            
    def _save_trainable_weights(self, trainer, checkpoint_dir: Path):
        """
        辅助函数：提取、清理并保存可训练的权重。
        """
        print("--- 正在提取并保存仅可训练的权重... ---")
        
        # a. 获取未包装的原始模型
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        
        # b. 筛选出所有可训练的参数
        trainable_state_dict = {
            name: param
            for name, param in unwrapped_model.named_parameters()
            if param.requires_grad
        }

        if not trainable_state_dict:
            warnings.warn("在模型中没有找到可训练的参数，跳过保存 trainable_only.safetensors。")
            return
        if self.remove_prefix_in_ckpt:
            clean_state_dict = {}
            prefix = self.remove_prefix_in_ckpt
            for k, v in trainable_state_dict.items():
                if k.startswith(prefix):
                    # 只保留前缀之后的部分
                    clean_state_dict[k[len(prefix):]] = v
                else:
                    # 如果没有这个前缀，则保留原样
                    clean_state_dict[k] = v
            trainable_state_dict = clean_state_dict

        # d. 定义保存路径并保存
        save_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
        trainer.accelerator.save(trainable_state_dict, str(save_path),safe_serialization=True) # safe_serialization=True 一定要有
        print(f"✅ 仅可训练权重已保存至 -> {save_path}")

    def on_epoch_end(self, trainer):
        trainer.accelerator.wait_for_everyone()
        if trainer.accelerator.is_main_process:
            epoch_checkpoint_dir = self.output_path / f"epoch_{trainer.epoch_id}"
            epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Epoch {trainer.epoch_id}: 正在保存推理用权重...")
            self._save_trainable_weights(trainer, epoch_checkpoint_dir)
            # 保存validation的所有key对应的视频为GIF
            if hasattr(trainer, "video") and isinstance(trainer.video, dict):
                for key, value in trainer.video.items():
                    gif_path = epoch_checkpoint_dir / f"{key}.gif"
                    try:
                        self._save_gif(value, str(gif_path))
                    except Exception as e:
                        print(f"❌ 保存GIF失败: {gif_path}, 错误: {e}")

class WandbLoggingCallback_wan2_1_fun_14b_control_normal_lora_train_po_v0(Callback):
    """wandb callback"""
    def __init__(self, project_name, hps):
        self.project_name = project_name
        self.hps = hps

    def on_train_begin(self, trainer):
        # 在训练开始时初始化 wandb
        trainer.accelerator.init_trackers(project_name=self.project_name, config=self.hps)

    def on_step_end(self, trainer):
        # 在每一步结束时记录loss和学习率
        if trainer.accelerator.is_main_process:
            current_lr = trainer.scheduler.get_last_lr()[0]
            trainer.accelerator.log(
                {"training_loss": trainer.loss.item(), "learning_rate": current_lr},
                step=trainer.global_step
            )

    def on_train_end(self, trainer):
        # 在训练结束时关闭 wandb
        trainer.accelerator.end_training()

class TensorboardLoggingCallback_wan2_1_fun_14b_control_normal_lora_train_po_v0(Callback):
    """tensorboard callback"""
    def __init__(self, logging_dir=None, hps=None):
        self.logging_dir = logging_dir
        self.hps = hps

    def on_train_begin(self, trainer):
        # 在训练开始时初始化 tensorboard
        trainer.accelerator.init_trackers(
            project_name="tensorboard", 
            config=self.hps, 
           # init_kwargs={"tensorboard": {"logging_dir": self.logging_dir}}
            )

    def on_step_end(self, trainer):
        # 在每一步结束时记录loss和学习率
        if trainer.accelerator.is_main_process:
            current_lr = trainer.scheduler.get_last_lr()[0]
            trainer.accelerator.log(
                {"training_loss": trainer.loss.item(), "learning_rate": current_lr},
                step=trainer.global_step
            )
    def on_epoch_end(self, trainer):
        if trainer.accelerator.is_main_process:
            trainer.accelerator.log(
                {"val_normal_loss": trainer.val_normal_loss},
                step=trainer.global_step,
            )
    def on_train_end(self, trainer):
        # tensorboard的SummaryWriter会自动关闭，无需手动处理
        pass

class WanTrainingModule_wan2_1_fun_14b_control_normal_lora_train_po_v0(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
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
        # 检测可用的设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs)
        # 我是真没招了，就在这里手动赋值吗，不追求极致的解耦了

        #-------start------#
        self.pipe.units = [ # 一定要加()，这样才实例化
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_FunControl()
        ]
        # self.pipe.dit.patch_embedding = torch.nn.Conv3d(64,1536,kernel_size=(1,2,2),stride=(1,2,2)) 
        # self.pipe.dit.head.head = torch.nn.Linear(1536, 128,bias=True) 
        # #-------end------#
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
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
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_videos": [
                data["normals"],],
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
                inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

class Trainer_wan2_1_fun_14b_control_normal_lora_train_po_v0:
    def __init__(
            self, 
            model, 
            optimizer, 
            dataloader, 
            scheduler, 
            num_epochs, 
            gradient_accumulation_steps, 
            logger_type=None, 
            callbacks: Optional[list[Callback]] = None,
            val_dataloader=None # 新增验证集dataloader参数
            ):
        # 1. 初始化 Accelerator，logger_type 从外部传入
        # self.accelerator = Accelerator(
        #         gradient_accumulation_steps=gradient_accumulation_steps,
        #         log_with=logger_type
        # )
        if logger_type == "tensorboard":
            for cb in callbacks: # type: ignore
                if hasattr(cb, "logging_dir"):
                    logging_dir = cb.logging_dir # type: ignore
                    break
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=logger_type,
                project_dir=logging_dir # 如果用tensorboard，则需要指定project_dir，否则会报错
            )
        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=logger_type
            )
        # 2. 准备所有组件, 这一步耗时是最长的 1.3B 15s以上
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        self.val_dataloader = val_dataloader  # 保存验证集dataloader
        self.num_epochs = num_epochs
        self.callbacks = callbacks if callbacks is not None else []
        self.global_step = 0
        self.epoch_id = 0
        self.loss = None

    def _call_callbacks(self, event_name):
        """一个统一调用所有回调函数的辅助方法"""
        for callback in self.callbacks:
            getattr(callback, event_name)(self)

    def validate(self):
        if self.val_dataloader is None:
            print("[Warning] No val_dataloader provided, skip validation.")
            return None
        self.model.eval()
        total_depth_loss = 0
        total_normal_loss = 0
        count = 0
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

        with torch.no_grad():
            for data in tqdm(self.val_dataloader, disable=not self.accelerator.is_main_process, desc=f"Epoch {self.epoch_id}: Validating"):
                if data is None:
                    continue
                inference_params["input_videos"] = [
                    data["normals"].to(self.model.pipe.device).to(self.model.pipe.torch_dtype),
                ]
                inference_params["control_video"] = data["rgbs"].to(self.model.pipe.device).to(self.model.pipe.torch_dtype) # dataloader的collate_fn会自动加上batch
                inference_params["height"] = data.get("height", 480)
                inference_params["width"] = data.get("width", 832)
                inference_params["num_frames"] = data.get("num_frames", 81)
                inference_params["num_input_videos"] = len(inference_params["input_videos"])
                # 执行推理
                try:
                    video = self.model.pipe(**inference_params)
                    normals = video["video_1"]
                    normal_predict_list = [to_tensor(normal) for normal in normals]
                    normal_predict_tensor = torch.stack(normal_predict_list,dim=1)[None, ...]
                    normal_predict_tensor = normal_predict_tensor.to(self.model.pipe.device).to(self.model.pipe.torch_dtype)*2 -1
                    loss_normal = F.mse_loss(normal_predict_tensor, data["normals"].to(self.model.pipe.device).to(self.model.pipe.torch_dtype))
                    total_normal_loss += loss_normal
                    if count == 0:
                        self.video = video
                    count += 1
                except Exception as e:
                    print(f"❌ 推理失败: {e}")
                
        avg_normal_loss = total_normal_loss / max(count, 1)
        print(f"[Validation] Epoch {self.epoch_id}: val_normal_loss={avg_normal_loss:.6f}")
        self.model.train()
        return avg_normal_loss

    def train(self):
        self._call_callbacks("on_train_begin")

        for epoch_id in range(self.num_epochs):
            self.epoch_id = epoch_id
            self._call_callbacks("on_epoch_begin")
            
            # 使用accelerator.main_process_<y_bin_401>来包装tqdm，确保只在主进程显示进度条
            pbar = tqdm(self.dataloader, disable=not self.accelerator.is_main_process, desc=f"Epoch {epoch_id+1}/{self.num_epochs}")
            
            for data in pbar:
                self._call_callbacks("on_step_begin")
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    self.loss = self.model(data) # 将loss保存在self中，供callback使用
                    self.accelerator.backward(self.loss)
                    self.optimizer.step()               
                    self.scheduler.step()
                
                self.global_step += 1
                if self.accelerator.is_main_process:
                    pbar.set_postfix({"loss": float(self.loss.detach().cpu().item())}) # 在进度条上显示loss
                self._call_callbacks("on_step_end")
            
            # 每个epoch结束后做一次验证
            self.val_normal_loss = self.validate() # type: ignore
            self._call_callbacks("on_epoch_end")

        self._call_callbacks("on_train_end")