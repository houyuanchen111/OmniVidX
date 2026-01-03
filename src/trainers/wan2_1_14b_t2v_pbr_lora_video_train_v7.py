"""
这边是wan2.1-t2v-1.3b-control-normal-depth-train的trainer
可训练self.attn
"""

import torch
from pathlib import Path
import warnings
from tqdm import tqdm
from PIL import Image
from typing import Optional
from accelerate import Accelerator,DeepSpeedPlugin
from accelerate.state import AcceleratorState
import av,os
from ..trainers.util import Callback
from torchvision.io import write_video
import numpy as np
import matplotlib
from PIL import Image
import torchvision
import imageio
# 下面两个callback是需要定制化的，便于控制
class ModelCheckpointCallback_wan2_1_14b_t2v_pbr_lora_video_train_v7(Callback):     
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
    
    def _save_resume_weights(self, trainer, checkpoint_dir: Path):
        """
        保存resume权重
        """
        print("--- 正在保存resume权重... ---")
        # 确保 checkpoint_dir 是一个字符串，因为一些旧版本的库可能需要
        checkpoint_dir_str = str(checkpoint_dir)
        trainer.accelerator.save_state(checkpoint_dir_str)
        print(f"✅ 权重已保存至 -> {checkpoint_dir_str}")
     
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
        epoch_checkpoint_dir = self.output_path / f"epoch_{trainer.epoch_id}"
        if trainer.accelerator.is_main_process:
            print(f"\n--- End of Epoch {trainer.epoch_id} ---")
            print(f"准备在主进程上创建目录: {epoch_checkpoint_dir}")
            epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Epoch {trainer.epoch_id}: 正在保存推理用权重...")
            self._save_trainable_weights(trainer, epoch_checkpoint_dir)
            print("目录准备就绪。即将开始保存状态...")
        # print(f"进程 {trainer.accelerator.process_index}: 正在调用 _save_resume_weights...")
        # self._save_resume_weights(trainer, epoch_checkpoint_dir)
        # trainer.accelerator.wait_for_everyone()
        # if trainer.accelerator.is_main_process:
        #     print(f"✅ Epoch {trainer.epoch_id} 检查点已在所有进程上保存完毕。")

class TensorboardLoggingCallback_wan2_1_14b_t2v_pbr_lora_video_train_v7(Callback):    
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
                {"val_loss": trainer.val_loss},
                step=trainer.global_step,
            )
    
    def on_train_end(self, trainer):
        # tensorboard的SummaryWriter会自动关闭，无需手动处理
        pass

class Trainer_wan2_1_14b_t2v_pbr_lora_video_train_v7:
    def __init__(
            self, 
            model, 
            optimizer, 
            video_dataloader, 
            scheduler, 
            num_epochs, 
            gradient_accumulation_steps, 
            logger_type=None, 
            callbacks: Optional[list[Callback]] = None,
            resume_from_deepspeed: Optional[str] = None,
            val_dataloader=None, # 新增验证集dataloader参数
            frame_dataloader=None, # 新增frame数据集dataloader参数
            t2RAIN_dataloader=None, # 新增t2RAIN数据集dataloader参数
            frame_R2AIN_dataloader=None # 新增frame_R2AIN数据集dataloader参数
            ):
        
        if logger_type == "tensorboard":
            for cb in callbacks: # type: ignore
                if hasattr(cb, "logging_dir"):
                    logging_dir = cb.logging_dir # type: ignore
                if hasattr(cb, "output_path"):
                    self.output_path = cb.output_path # type: ignore
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=logger_type,
                project_dir=logging_dir # 如果用tensorboard，则需要指定project_dir，否则会报错
            )
        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                log_with=logger_type,
                #deepspeed_plugin=plugin
            )
        if video_dataloader.batch_size == None:
            state = AcceleratorState()
            if state.deepspeed_plugin is not None:
                state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = video_dataloader.batch_sampler.batch_size
                # 可选：打印确认
                print("DS micro batch size set to:",
                    state.deepspeed_plugin.deepspeed_config.get('train_micro_batch_size_per_gpu'))
        # 2. 准备所有组件, 这一步耗时是最长的 1.3B 15s以上
        # 准备所有训练用的dataloader（不包括val_dataloader）
        prepare_list = [model, optimizer, video_dataloader, scheduler]
        
        if frame_dataloader is not None:
            prepare_list.append(frame_dataloader)
        if t2RAIN_dataloader is not None:
            prepare_list.append(t2RAIN_dataloader)
        if frame_R2AIN_dataloader is not None:
            prepare_list.append(frame_R2AIN_dataloader)
        
        prepared = self.accelerator.prepare(*prepare_list) # 包装模型
        
        # 解包并赋值
        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.video_dataloader = prepared[2]
        self.scheduler = prepared[3]
        
        idx = 4
        if frame_dataloader is not None:
            self.frame_dataloader = prepared[idx]
            idx += 1
        else:
            self.frame_dataloader = None
            
        if t2RAIN_dataloader is not None:
            self.t2RAIN_dataloader = prepared[idx]
        else:
            self.t2RAIN_dataloader = None
        if frame_R2AIN_dataloader is not None:
            self.frame_R2AIN_dataloader = prepared[idx]
            idx += 1
        else:
            self.frame_R2AIN_dataloader = None
        if resume_from_deepspeed is not None:
            try:
                print(f"正在从checkpoint {resume_from_deepspeed} 恢复训练")
                self.accelerator.load_state(resume_from_deepspeed)
            except Exception as e:
                raise ValueError(f"从checkpoint {resume_from_deepspeed} 恢复训练失败: {e}")

            
        self.val_dataloader = val_dataloader  # 保存验证集dataloader
        self.num_epochs = num_epochs
        self.callbacks = callbacks if callbacks is not None else []
        self.global_step = 0
        self.epoch_id = 0
        #self.model.pipe.dit.context_embedding.requires_grad = True
        self.loss = None
        
        # 在所有 dataloader 准备好后，定义 training_mode 到 dataloader 的映射
        # 根据你的需求修改这个映射关系
        # _v 后缀使用 video_dataloader 或 t2RAIN_dataloader
        # _i 后缀使用 frame_dataloader
        self.training_mode_to_dataloader = {
            # _v 后缀：视频数据
            "t2RAIN_v": self.video_dataloader,
            "R2AIN_v": self.video_dataloader,
            "A2RIN_v": self.video_dataloader,
            "I2RAN_v": self.video_dataloader,
            "N2RAI_v": self.video_dataloader,
            "RA2IN_v": self.video_dataloader,
            "RI2AN_v": self.video_dataloader,
            "RN2AI_v": self.video_dataloader,
            "AI2RN_v": self.video_dataloader,
            "AN2RI_v": self.video_dataloader,
            "IN2RA_v": self.video_dataloader,
            "AIN2R_v": self.video_dataloader,
            "RIN2A_v": self.video_dataloader,
            "RAN2I_v": self.video_dataloader,
            "RAI2N_v": self.video_dataloader,
            # _i 后缀：图像数据
            # "t2RAIN_i": self.frame_dataloader,
            "R2AIN_i": self.frame_dataloader,
            # "A2RIN_i": self.frame_dataloader,
            # "I2RAN_i": self.frame_dataloader,
            # "N2RAI_i": self.frame_dataloader,
            # "RA2IN_i": self.frame_dataloader,
            # "RI2AN_i": self.frame_dataloader,
            # "RN2AI_i": self.frame_dataloader,
            # "AI2RN_i": self.frame_dataloader,
            # "AN2RI_i": self.frame_dataloader,
            # "IN2RA_i": self.frame_dataloader,
            "AIN2R_i": self.frame_dataloader,
            # "RIN2A_i": self.frame_dataloader,
            # "RAN2I_i": self.frame_dataloader,
            # "RAI2N_i": self.frame_dataloader,
        }
        
        # 存储所有可用的 training_mode 列表（只包含有对应 dataloader 的模式）
        self.available_training_modes = [
            mode for mode, dl in self.training_mode_to_dataloader.items() 
            if dl is not None
        ]
        
        if self.accelerator.is_main_process:
            print(f"✅ 可用的 training_mode: {self.available_training_modes}")
            print(f"✅ training_mode 到 dataloader 的映射已初始化")
    
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

    def _tensor2video(self, tensor: torch.Tensor, file_path: str, fps: int = 15):
        assert tensor.ndim == 4, "tensor must be [c,t,h,w]"
        tensor = tensor.cpu()
        video_tensor = tensor.permute(1, 2, 3, 0) # c t h w -> t h w c       
        video_tensor = (video_tensor * 255).to(torch.uint8)
        print(f"正在保存视频到: {file_path}")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path),exist_ok = True)
        write_video(
            filename=file_path,
            video_array=video_tensor,
            fps=fps,
            video_codec='h264' # 使用常见的 h264 编码器
        )
        print("视频保存成功！")
    
    def _colorize_disparity(self,disparity: np.ndarray, mask: np.ndarray = None, normalize: bool = False, cmap: str = 'Spectral') -> torch.Tensor: # type: ignore
            """
            Args:
                disparity: [H, W] 0-1
                mask: [H, W] 0-1
                normalize: False 因为输入都做了normalize，所以false就行了
                cmap: 'Spectral'
            Returns:
                [C, H, W]
            """
            assert disparity.min() >= 0 and disparity.max() <= 1, "disparity is not in [0,1]"
            if mask is None:
                disparity = np.where(disparity > 0, disparity, np.nan)
            else:
                disparity = np.where((disparity > 0) & mask.astype(np.uint8), disparity, np.nan)
            if normalize:
                min_disp, max_disp = np.nanquantile(disparity, 0.001), np.nanquantile(disparity, 0.99)
                disparity = (disparity - min_disp) / (max_disp - min_disp)
            colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disparity)[..., :3], 0) # type: ignore
            return torch.from_numpy(colored).permute(2,0,1) # C H W
    
    def _colorize_disparity_video(self, disparity: torch.Tensor, mask: torch.Tensor):
        """
        disparity: [c,t,h,w] (-1,1)
        mask: [1,t,h,w] 0-1
        return: [c,t,h,w]
        """
        disparity_ = disparity.mean(dim=0).clone().cpu().float().numpy() # [t,h,w]
        disparity_vis = []
        for i in range(disparity_.shape[0]):
            disparity_vis.append(self._colorize_disparity(disparity_[i] * 0.5 + 0.5))
        disparity_vis = torch.stack(disparity_vis, dim=1).to(disparity.device).to(disparity.dtype) 
        return disparity_vis * mask # C T H W
    
    def _call_callbacks(self, event_name):
        """一个统一调用所有回调函数的辅助方法"""
        for callback in self.callbacks:
            getattr(callback, event_name)(self)
    
    def _get_synced_training_mode(self) -> str:
        """
        在多GPU环境下同步选择 training_mode，确保所有卡使用相同的模式
        
        Returns:
            str: 选择的 training_mode
        """
        import torch.distributed as dist
        
        # 检查是否在分布式环境中
        if dist.is_available() and dist.is_initialized():
            # 只有 rank 0 生成随机数
            if dist.get_rank() == 0:
                # 使用 global_step 作为随机种子的一部分，确保可复现性
                # 但每次仍然随机选择
                flag = torch.rand(1).item()
                mode_idx = int(flag * len(self.available_training_modes))
                selected_mode = self.available_training_modes[mode_idx]
            else:
                # 其他 rank 占位，真正的值会被 broadcast 覆盖
                selected_mode = self.available_training_modes[0]
            
            # 广播字符串到所有rank
            # 将字符串转换为字节进行广播
            mode_bytes = selected_mode.encode('utf-8')
            max_len = 20  # 足够长的模式字符串
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            mode_tensor = torch.zeros(max_len, dtype=torch.uint8, device=device)
            if dist.get_rank() == 0:
                for i, b in enumerate(mode_bytes):
                    mode_tensor[i] = b
            dist.broadcast(mode_tensor, src=0)
            
            # 转换回字符串
            received_bytes = []
            for i in range(max_len):
                if mode_tensor[i] > 0:
                    received_bytes.append(int(mode_tensor[i].item()))
            received_mode = bytes(received_bytes).decode('utf-8')
            return received_mode
        else:
            # 单卡训练时的逻辑
            flag = torch.rand(1).item()
            mode_idx = int(flag * len(self.available_training_modes))
            return self.available_training_modes[mode_idx]
    
    def _get_dataloader_for_mode(self, training_mode: str):
        """
        根据 training_mode 返回对应的 dataloader
        
        Args:
            training_mode: 训练模式字符串
            
        Returns:
            DataLoader: 对应的 dataloader
        """
        dataloader = self.training_mode_to_dataloader.get(training_mode)
        if dataloader is None:
            # 如果指定的模式没有对应的 dataloader，使用默认的 video_dataloader
            if self.accelerator.is_main_process:
                print(f"⚠️ Warning: training_mode '{training_mode}' 没有对应的 dataloader，使用默认的 video_dataloader")
            dataloader = self.video_dataloader
        return dataloader

    def validate(self):
        val_batch = next(iter(self.val_dataloader)) # type: ignore
        # 提取 Matte 任务的数据: [com, pha, fgr, bgr]
        val_prompt = val_batch.get("prompt")[0]
        inference_rgb = val_batch.get("rgb")
        inference_albedo = val_batch.get("albedo")
        inference_irradiance = val_batch.get("irradiance")
        inference_normal = val_batch.get("normal")

        
        self.model.eval()
        count = 0
        
        # ========== 1. t2RAIN: 全生成模式 ==========
        print("\n[Validation] Testing t2RAIN mode...")
        inference_params = {
            # "prompt":"纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
            "prompt": [
            "宇航员骑着独角兽,驰骋在月球表面。", 
            "宇航员骑着独角兽,驰骋在月球表面。", 
            "宇航员骑着独角兽,驰骋在月球表面。", 
            "宇航员骑着独角兽,驰骋在月球表面。", 
            ],
            "negative_prompt":"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "seed": 1,
            "num_inference_steps": 50,
            "cfg_scale": 5.0,
            "cfg_merge": False,
            "height": 480,
            "width": 640,
            "num_frames": 21,  # 改为21帧
            "denoising_strength": 1.0,
            "tiled": True,
            "tiled": True,
            "tile_size": [30, 52],
            "tile_stride": [15, 26],
            "is_inference": True,
            "inference_rgb": None,
            "inference_albedo": None,
            "inference_irradiance": None,
            "inference_normal": None,
            "training_mode": "t2RAIN",
        }   

        video_dict = self.model.pipe(**inference_params)
        os.makedirs(f"{self.output_path}/epoch_{self.epoch_id}/t2RAIN", exist_ok=True)
        
        # 保存各个分支的生成视频
        if "rgb" in video_dict:
            rgb_gen = video_dict["rgb"]  # [C, T, H, W]
            self._tensor2video(rgb_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/t2RAIN/rgb_gen_{count}.mp4")
        if "albedo" in video_dict:
            albedo_gen = video_dict["albedo"]
            self._tensor2video(albedo_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/t2RAIN/albedo_gen_{count}.mp4")
        if "irradiance" in video_dict:
            irradiance_gen = video_dict["irradiance"]
            self._tensor2video(irradiance_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/t2RAIN/irradiance_gen_{count}.mp4")
        if "normal_unit" in video_dict:
            normal_gen = video_dict["normal_unit"]
            self._tensor2video(normal_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/t2RAIN/normal_gen_{count}.mp4")

        # 保存拼接视频（横向拼接所有模态）
        modalities_to_cat = []
        if "rgb" in video_dict:
            modalities_to_cat.append(video_dict["rgb"] * 0.5 + 0.5)
        if "albedo" in video_dict:
            modalities_to_cat.append(video_dict["albedo"] * 0.5 + 0.5)
        if "irradiance" in video_dict:
            modalities_to_cat.append(video_dict["irradiance"] * 0.5 + 0.5)
        if "normal_unit" in video_dict:
            modalities_to_cat.append(video_dict["normal_unit"] * 0.5 + 0.5)
        
        if modalities_to_cat:
            cat_video = torch.cat(modalities_to_cat, dim=-1)  # 在宽度维度拼接
            self._tensor2video(cat_video, f"{self.output_path}/epoch_{self.epoch_id}/t2RAIN/cat_all_{count}.mp4")
 
        # ========== 2. R2AIN mode==========
        print("\n[Validation] Testing N2RAI mode...")

        # 尝试从本地调试视频替换 inference_rgb，视频尺寸为 480x832
        try:
            video_path = "/share/project/cwm/houyuan.chen/diffusion_render_inverse_render/Wan_IR/debug_480_832.mp4"
            if os.path.exists(video_path):
                reader = imageio.get_reader(video_path, mode="I")
                frames = []
                for frame in reader:
                    frame_tensor = torch.from_numpy(np.array(frame, dtype=np.float32))  # H W C, 0-255
                    frame_tensor = frame_tensor / 127.5 - 1.0  # -> [-1, 1]
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # C H W
                    frame_tensor = torchvision.transforms.functional.resize(frame_tensor, [480, 640])  # C H W -> 480x640
                    frame_tensor = frame_tensor.clamp(-1.0, 1.0)
                    frames.append(frame_tensor)
                reader.close()

                if len(frames) > 0:
                    # 保证时间维度为 21 帧
                    if len(frames) >= 21:
                        frames = frames[:21]
                    else:
                        last_frame = frames[-1]
                        frames.extend([last_frame.clone() for _ in range(21 - len(frames))])

                    video_tensor = torch.stack(frames, dim=1)  # C T H W, T=21
                    video_tensor = video_tensor.unsqueeze(0)    # B C T H W
                    inverse_render_inference_rgb = video_tensor.to(inference_rgb.device).to(inference_rgb.dtype)
        
        except Exception as e:
            print(f"[Validation] Failed to load debug video for Inverse_Rendering: {e}")

        if inference_rgb is not None:
            inference_params = {
                # "prompt":"纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
                "prompt": [
                    "",
                    "",
                    "",
                    "",
                ],
                "negative_prompt":"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                "seed": 1,
                "num_inference_steps": 50,
                "cfg_scale": 5.0,
                "cfg_merge": False,
                "height": 480,
                "width": 640,
                "num_frames": 21,  # 改为21帧
                "denoising_strength": 1.0,
                "tiled": True,
                "tiled": True,
                "tile_size": [30, 52],
                "tile_stride": [15, 26],
                "is_inference": True,
                "inference_rgb": inverse_render_inference_rgb,
                "inference_albedo": None,
                "inference_irradiance": None,
                "inference_normal": None,
                "training_mode": "R2AIN",
            }   

            video_dict = self.model.pipe(**inference_params)
            os.makedirs(f"{self.output_path}/epoch_{self.epoch_id}/R2AIN", exist_ok=True)
            
            # 保存预测视频
            if "albedo" in video_dict:
                albedo_gen = video_dict["albedo"]
                self._tensor2video(albedo_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/R2AIN/albedo_gen_{count}.mp4")
            if "irradiance" in video_dict:
                irradiance_gen = video_dict["irradiance"]
                self._tensor2video(irradiance_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/R2AIN/irradiance_gen_{count}.mp4")
            if "normal_unit" in video_dict:
                normal_gen = video_dict["normal_unit"]
                self._tensor2video(normal_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/R2AIN/normal_gen_{count}.mp4")

            # 保存拼接视频 (输入com + 预测的pha, fgr, bgr)
            modalities_to_cat = [inverse_render_inference_rgb[0] * 0.5 + 0.5]  # [C, T, H, W]
            if "albedo" in video_dict:
                modalities_to_cat.append(video_dict["albedo"] * 0.5 + 0.5)
            if "irradiance" in video_dict:
                modalities_to_cat.append(video_dict["irradiance"] * 0.5 + 0.5)
            if "normal_unit" in video_dict:
                modalities_to_cat.append(video_dict["normal_unit"] * 0.5 + 0.5)
            
            
            cat_video = torch.cat(modalities_to_cat, dim=-1)  # 在宽度维度拼接
            self._tensor2video(cat_video, f"{self.output_path}/epoch_{self.epoch_id}/R2AIN/cat_all_{count}.mp4")
        # ========== 3. AN2RI mode: ==========
        print("\n[Validation] Testing AN2RI mode...")
        if inference_albedo is not None and inference_normal is not None:
            inference_params = {
                "prompt": 
                [
                    val_prompt + "赛博朋克霓虹灯光，强烈的蓝紫色和粉红色霓虹灯的室内场景。",
                    val_prompt + "赛博朋克霓虹灯光，强烈的蓝紫色和粉红色霓虹灯的室内场景。",
                    val_prompt + "赛博朋克霓虹灯光，强烈的蓝紫色和粉红色霓虹灯的室内场景。",
                    val_prompt + "赛博朋克霓虹灯光，强烈的蓝紫色和粉红色霓虹灯的室内场景。"
                ],
                
                "negative_prompt":"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                "seed": 1,
                "num_inference_steps": 50,
                "cfg_scale": 5.0,
                "cfg_merge": False,
                "height": 480,
                "width": 640,
                "num_frames": 21,  # 改为21帧
                "denoising_strength": 1.0,
                "tiled": True,
                "tile_size": [30, 52],
                "tile_stride": [15, 26],
                "is_inference": True,
                "training_mode": "AN2RI",
                "inference_rgb": None,
                "inference_albedo": inference_albedo,
                "inference_normal": inference_normal,
                "inference_irradiance": None,
            }   

            video_dict = self.model.pipe(**inference_params)
            os.makedirs(f"{self.output_path}/epoch_{self.epoch_id}/AN2RI", exist_ok=True)
            
            # 保存预测视频
            if "rgb" in video_dict:
                rgb_gen = video_dict["rgb"]
                self._tensor2video(rgb_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/AN2RI/rgb_gen_{count}.mp4")
            if "irradiance" in video_dict:
                irradiance_gen = video_dict["irradiance"]
                self._tensor2video(irradiance_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/AN2RI/irradiance_gen_{count}.mp4")

            # 保存拼接视频 (输入pha, fgr + 预测的com, bgr)
            modalities_to_cat = []
            if "rgb" in video_dict:
                modalities_to_cat.append(video_dict["rgb"] * 0.5 + 0.5)
            modalities_to_cat.append(inference_albedo[0] * 0.5 + 0.5)
            if "irradiance" in video_dict:
                modalities_to_cat.append(video_dict["irradiance"] * 0.5 + 0.5)
            modalities_to_cat.append(inference_normal[0] * 0.5 + 0.5)
            
            cat_video = torch.cat(modalities_to_cat, dim=-1)  # 在宽度维度拼接
            self._tensor2video(cat_video, f"{self.output_path}/epoch_{self.epoch_id}/AN2RI/cat_all_{count}.mp4")
        
        # ========== 4. AIN2R mode: ==========

        if inference_rgb is not None:
            inference_params = {
                # "prompt":"纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
                "prompt": [
                    "",
                    "",
                    "",
                    "",
                ],
                "negative_prompt":"色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                "seed": 1,
                "num_inference_steps": 50,
                "cfg_scale": 5.0,
                "cfg_merge": False,
                "height": 480,
                "width": 640,
                "num_frames": 21,  # 改为21帧
                "denoising_strength": 1.0,
                "tiled": True,
                "tiled": True,
                "tile_size": [30, 52],
                "tile_stride": [15, 26],
                "is_inference": True,
                "inference_rgb": None,
                "inference_albedo": inference_albedo,
                "inference_irradiance": inference_irradiance,
                "inference_normal": inference_normal,
                "training_mode": "AIN2R",
            }   

            video_dict = self.model.pipe(**inference_params)
            os.makedirs(f"{self.output_path}/epoch_{self.epoch_id}/AIN2R", exist_ok=True)
            
            # 保存预测视频
            if "rgb" in video_dict:
                rgb_gen = video_dict["rgb"]
                self._tensor2video(rgb_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/AIN2R/rgb_gen_{count}.mp4")


            # 保存拼接视频 (输入com + 预测的pha, fgr, bgr)
            modalities_to_cat = []  # [C, T, H, W]
            if "rgb" in video_dict:
                modalities_to_cat.append(video_dict["rgb"] * 0.5 + 0.5)
            modalities_to_cat.append(inference_albedo[0] * 0.5 + 0.5)
            modalities_to_cat.append(inference_normal[0] * 0.5 + 0.5)
            modalities_to_cat.append(inference_irradiance[0] * 0.5 + 0.5)
            
            
            cat_video = torch.cat(modalities_to_cat, dim=-1)  # 在宽度维度拼接
            self._tensor2video(cat_video, f"{self.output_path}/epoch_{self.epoch_id}/AIN2R/cat_all_{count}.mp4")
        
        print("\n[Validation] All modes tested successfully!")
        return 0.0
       
    def train(self):
        self._call_callbacks("on_train_begin")

        for epoch_id in range(self.num_epochs):
            self.epoch_id = epoch_id
            self._call_callbacks("on_epoch_begin")
            
            # 为每个 dataloader 创建迭代器字典
            dataloader_iterators = {}
            for mode, dataloader in self.training_mode_to_dataloader.items():
                if dataloader is not None:
                    dataloader_iterators[mode] = iter(dataloader)
            
            # 计算总步数（使用最长的 dataloader 作为参考，或者使用 video_dataloader 的长度）
            if self.video_dataloader is not None:
                max_steps = len(self.video_dataloader)
            else:
                max_steps = max((len(dl) for dl in self.training_mode_to_dataloader.values() if dl is not None), default=10000)
            
            # 使用accelerator.main_process_<y_bin_401>来包装tqdm，确保只在主进程显示进度条
            pbar = tqdm(range(max_steps), disable=not self.accelerator.is_main_process, desc=f"Epoch {epoch_id+1}/{self.num_epochs}")
            
            batch_idx = 0
            consecutive_failures = 0
            max_consecutive_failures = 10  # 最大连续失败次数
            
            while batch_idx < max_steps:
                try:
                    # 在每一步随机选择 training_mode（多卡同步）
                    training_mode = self._get_synced_training_mode()
                    
                    # 根据 training_mode 选择对应的 dataloader 迭代器
                    if training_mode not in dataloader_iterators:
                        # 如果该模式没有对应的迭代器，创建它
                        dataloader = self._get_dataloader_for_mode(training_mode)
                        if dataloader is not None:
                            dataloader_iterators[training_mode] = iter(dataloader)
                    
                    # 从对应的迭代器获取数据
                    data = None
                    try:
                        data = next(dataloader_iterators[training_mode])
                    except StopIteration:
                        # 如果迭代器用完了，重新创建
                        dataloader = self._get_dataloader_for_mode(training_mode)
                        if dataloader is not None:
                            dataloader_iterators[training_mode] = iter(dataloader)
                            try:
                                data = next(dataloader_iterators[training_mode])
                            except StopIteration:
                                # 如果重新创建后仍然为空，说明这个 dataloader 确实没有数据
                                if self.accelerator.is_main_process:
                                    print(f"⚠️ Warning: training_mode '{training_mode}' 的 dataloader 为空，跳过此步")
                                batch_idx += 1
                                pbar.update(1)
                                continue
                        else:
                            # 如果没有可用的 dataloader，跳过这一步
                            if self.accelerator.is_main_process:
                                print(f"⚠️ Warning: training_mode '{training_mode}' 没有对应的 dataloader，跳过此步")
                            batch_idx += 1
                            pbar.update(1)
                            continue
                    
                    if data is None:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            if self.accelerator.is_main_process:
                                print(f"⚠️ 连续 {max_consecutive_failures} 次获取数据失败，可能所有 dataloader 都已用完")
                            break
                        batch_idx += 1
                        pbar.update(1)
                        continue
                    
                    consecutive_failures = 0  # 重置失败计数
                    
                    # 将 training_mode 添加到 data 中，供 model 使用
                    if isinstance(data, dict):
                        data["training_mode"] = training_mode[:-2]
                    else:
                        # 如果 data 不是 dict，可能需要根据实际情况调整
                        if self.accelerator.is_main_process:
                            print(f"⚠️ Warning: data 不是 dict 类型，无法添加 training_mode")
                    
                    self._call_callbacks("on_step_begin")
                    with self.accelerator.accumulate(self.model):
                        self.optimizer.zero_grad()
                        self.loss = self.model(data) # 将loss保存在self中，供callback使用
                        
                        # 检查loss有效性，如果无效则跳过这个batch
                        if not torch.isfinite(self.loss):
                            print(f"⚠️ Batch {batch_idx + 1}: nan or inf loss detected, skipping...")
                            batch_idx += 1
                            pbar.update(1)
                            continue
                        
                        self.accelerator.backward(self.loss)
                        self.optimizer.step()               
                        self.scheduler.step()
                    
                    self.global_step += 1
                    if self.accelerator.is_main_process:
                        pbar.set_postfix({
                            "loss": float(self.loss.detach().cpu().item()),
                            "mode": training_mode
                        }) # 在进度条上显示loss和training_mode
                    self._call_callbacks("on_step_end")
                    
                    batch_idx += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Batch {batch_idx + 1} 处理出错: {e}")
                    if self.accelerator.is_main_process:
                        import traceback
                        traceback.print_exc()
                    # 继续处理下一个batch
                    batch_idx += 1
                    pbar.update(1)
                    continue
            
            # 跳过验证推理，仅执行保存权重的回调
            if self.num_epochs == 100:
                if self.epoch_id % 3 == 0: # 每9个epoch做一次验证
                    if self.accelerator.is_main_process:
                        try:
                            if self.epoch_id % 9 == 0:
                                self.val_loss = self.validate() # type: ignore
                            else:
                                self.val_loss = 0.0
                        except Exception as e:
                            print(f"❌ 验证推理失败: {e}")
                            import traceback
                            traceback.print_exc()
                            self.val_loss = 0.0
                    else:
                        self.val_loss = 0.0  # 非主进程设置默认值
                    self.accelerator.wait_for_everyone()
                    self._call_callbacks("on_epoch_end")
            elif self.num_epochs == 1000:
                if self.epoch_id % 99 == 0: # 每99个epoch做一次验证
                    if self.accelerator.is_main_process:
                        try:
                            self.val_loss = self.validate() # type: ignore
                        except Exception as e:
                            print(f"❌ 验证推理失败: {e}")
                            import traceback
                            traceback.print_exc()
                            self.val_loss = 0.0
                    else:
                        self.val_loss = 0.0  # 非主进程设置默认值
                if self.epoch_id % 11 == 0:
                    self.val_loss = 0.0
                    self.accelerator.wait_for_everyone()
                    self._call_callbacks("on_epoch_end")
                

        self._call_callbacks("on_train_end")

    def _compute_modality_loss(self, video_dict, data, modality_idx, mask, normal_gt, norm_normal_gt):
        """
        计算指定模态的损失
        
        Args:
            video_dict: 预测结果字典
            data: 真实数据字典
            modality_idx: 模态索引
            mask: 掩码张量
            normal_gt: 真实法线图
            norm_normal_gt: 法线图范数
            
        Returns:
            损失值或None（如果计算失败）
        """
        # 定义模态配置
        MODALITY_CONFIG = {
            0: {"key": "albedo", "name": "albedo"},
            1: {"key": "basecolor", "name": "basecolor"},
            2: {"key": "depth", "name": "depth"},
            3: {"key": "material", "name": "material"},
            4: {"key": "normal_unit", "name": "normal", "special": "normal"}
        }
        
        if modality_idx not in MODALITY_CONFIG:
            print(f"⚠️ 不支持的模态索引: {modality_idx}")
            return None
            
        config = MODALITY_CONFIG[modality_idx]
        modality_key = config["key"]
        
        # 检查预测结果是否存在
        if modality_key not in video_dict:
            print(f"⚠️ 预测结果中缺少模态: {modality_key}")
            return None
            
        try:
            # 获取预测和真实值
            predict = video_dict[modality_key].to(self.model.pipe.device) # [c,t,h,w]
            gt = data[config["name"]].to(self.model.pipe.device)[0]
            
            # 特殊处理法线图
            if config.get("special") == "normal":
                gt = gt / (norm_normal_gt + 1e-6)
            
            # 计算MSE损失
            if mask.dtype == "torch.bool":
                mask = mask.float().to(predict.device).to(predict.dtype)
            loss = (((predict - gt) * mask) ** 2).mean()
            return loss
            
        except Exception as e:
            print(f"❌ 计算模态 {modality_idx} ({config['name']}) 损失时出错: {e}")
            return None

    def _save_validation_videos(self, video_dict, data, rgbs_input_cond, epoch_losses, count, modality_idx):
        """
        保存验证视频，包括预测结果、真实值和对比视频
        
        Args:
            video_dict: 预测结果字典
            data: 真实数据字典
            rgbs_input_cond: 输入RGB条件
            epoch_losses: epoch损失字典
            count: 样本计数
        """
        # 定义模态配置
        MODALITY_CONFIG = {
            0: {"name": "albedo", "type": "single"},
            1: {"name": "basecolor", "type": "single"},
            2: {"name": "depth", "type": "single"},
            3: {"name": "material", "type": "dual", "keys": ["roughness", "metallic"]},
            4: {"name": "normal", "type": "single", "special": "normal"}
        }
        
        try:
            modality_idx = modality_idx
            if modality_idx not in MODALITY_CONFIG:
                print(f"⚠️ 不支持的模态索引: {modality_idx}")
                return
                
            config = MODALITY_CONFIG[modality_idx]
            modality_name = config["name"]
            
            # 获取损失值
            loss_value = epoch_losses.get(f"modality_{modality_idx}", 0.0)
            
            if config["type"] == "single":
                # 单模态处理
                self._save_single_modality_video(
                    video_dict, data, rgbs_input_cond, 
                    modality_name, modality_idx, count, loss_value
                )
                
            elif config["type"] == "dual":
                # 双模态处理（如material包含roughness和metallic）
                self._save_dual_modality_video(
                    video_dict, data, rgbs_input_cond,
                    modality_name, config["keys"], count, loss_value
                )
                
        except Exception as e:
            print(f"❌ 保存验证视频时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_single_modality_video(self, video_dict, data, rgbs_input_cond, 
                                   modality_name, modality_idx, count, loss_value):
        """保存单模态验证视频"""
        # 获取预测和真实值
        if modality_idx == 4:  # normal
            predict_tensor = video_dict["normal_unit"]
            normal_gt = data["normal"][0]
            norm_normal_gt = torch.norm(normal_gt, dim=0, keepdim=True)
            gt_tensor = normal_gt / (norm_normal_gt + 1e-6)
        else:
            modality_key = modality_name
            predict_tensor = video_dict[modality_key]
            gt_tensor = data[modality_key][0]
        
        # 转换到[0,1]范围
        predict_vis = predict_tensor * 0.5 + 0.5
        gt_vis = gt_tensor * 0.5 + 0.5
        input_vis = rgbs_input_cond * 0.5 + 0.5
        
        # 确保所有张量都在同一个device上
        target_device = predict_vis.device
        input_vis = input_vis.to(target_device)
        gt_vis = gt_vis.to(target_device)
        
        # 保存单独的视频
        self._tensor2video(predict_vis, f"{self.output_path}/epoch_{self.epoch_id}/{modality_name}_predict_{count}.mp4")
        self._tensor2video(gt_vis, f"{self.output_path}/epoch_{self.epoch_id}/{modality_name}_gt_{count}.mp4")
        
        # 保存对比视频（输入、预测、真实值并排）
        comparison_video = torch.cat([input_vis, predict_vis, gt_vis], dim=-1)
        self._tensor2video(comparison_video, f"{self.output_path}/epoch_{self.epoch_id}/{modality_name}_predict_gt_{count}_{loss_value:.4f}.mp4")
    
    def _save_dual_modality_video(self, video_dict, data, rgbs_input_cond,
                                 modality_name, modality_keys, count, loss_value):
        """保存双模态验证视频（如material的roughness和metallic）"""
        for key in modality_keys:
            # 获取预测和真实值
            predict_tensor = video_dict[key]
            gt_tensor = data[key][0]
            
            # 转换到[0,1]范围
            predict_vis = predict_tensor * 0.5 + 0.5
            gt_vis = gt_tensor * 0.5 + 0.5
            input_vis = rgbs_input_cond * 0.5 + 0.5
            
            # 确保所有张量都在同一个device上
            target_device = predict_vis.device
            input_vis = input_vis.to(target_device)
            gt_vis = gt_vis.to(target_device)
            
            # 保存单独的视频
            self._tensor2video(predict_vis, f"{self.output_path}/epoch_{self.epoch_id}/{modality_name}_{key}_predict_{count}.mp4")
            self._tensor2video(gt_vis, f"{self.output_path}/epoch_{self.epoch_id}/{modality_name}_{key}_gt_{count}.mp4")
            
            # 保存对比视频
            comparison_video = torch.cat([input_vis, predict_vis, gt_vis], dim=-1)
            self._tensor2video(comparison_video, f"{self.output_path}/epoch_{self.epoch_id}/{modality_name}_{key}_predict_gt_{count}_{loss_value:.4f}.mp4")

