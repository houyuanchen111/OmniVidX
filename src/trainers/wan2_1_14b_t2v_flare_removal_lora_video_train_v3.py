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
class ModelCheckpointCallback_wan2_1_14b_t2v_flare_removal_lora_video_train_v3(Callback):     
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

class TensorboardLoggingCallback_wan2_1_14b_t2v_flare_removal_lora_video_train_v3(Callback):    
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

class Trainer_wan2_1_14b_t2v_flare_removal_lora_video_train_v3:
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
            resume_from_deepspeed: Optional[str] = None,
            val_dataloader=None # 新增验证集dataloader参数
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
        if dataloader.batch_size == None:
            state = AcceleratorState()
            if state.deepspeed_plugin is not None:
                state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = dataloader.batch_sampler.batch_size
                # 可选：打印确认
                print("DS micro batch size set to:",
                    state.deepspeed_plugin.deepspeed_config.get('train_micro_batch_size_per_gpu'))
        # 2. 准备所有组件, 这一步耗时是最长的 1.3B 15s以上
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            model, optimizer, dataloader, scheduler
        ) # 包装模型
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

    def validate(self):
        val_batch = next(iter(self.val_dataloader)) # type: ignore
        # 提取 Flare Removal 任务的数据: [com, flare, clean]
        com = val_batch.get("com")[None, ...]
        flare = val_batch.get("flare")[None, ...]
        clean = val_batch.get("clean")[None, ...]
        
        self.model.eval()
        count = 0
        
        # ========== 1. Composition: 以 (clean, flare) 为条件，预测 com ==========
        print("\n[Validation] Testing Composition mode...")
        inference_params = {
            "prompt": [
                "",
                "",
                "",
            ],
            "negative_prompt": "",
            "seed": 1,
            "num_inference_steps": 50,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "height": 432,
            "width": 768,
            "num_frames": 21,
            "denoising_strength": 1.0,
            "tiled": True,
            "tile_size": [30, 52],
            "tile_stride": [15, 26],
            "is_inference": True,
            "training_mode": "Composition",
            "inference_clean": None if clean is None else clean[0],
            "inference_flare": None if flare is None else flare[0],
        }
        video_dict = self.model.pipe(**inference_params)
        os.makedirs(f"{self.output_path}/epoch_{self.epoch_id}/Composition", exist_ok=True)
        
        # 保存预测 com
        if "com" in video_dict:
            com_gen = video_dict["com"]  # [C, T, H, W]
            self._tensor2video(com_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/Composition/com_gen_{count}.mp4")
        
        # 保存拼接视频（条件 clean、flare 与预测 com 并排）
        videos_to_cat = []
        if clean is not None:
            videos_to_cat.append(clean[0][0] * 0.5 + 0.5)
        if flare is not None:
            videos_to_cat.append(flare[0][0] * 0.5 + 0.5)
        if "com" in video_dict:
            videos_to_cat.append(video_dict["com"]* 0.5 + 0.5)
        if videos_to_cat:
            cat_video = torch.cat(videos_to_cat, dim=-1)
            self._tensor2video(cat_video, f"{self.output_path}/epoch_{self.epoch_id}/Composition/cat_all_{count}.mp4")
        
        # ========== 2. FlareRemoval: 以 com 为条件，同时预测 flare 与 clean ==========
        print("\n[Validation] Testing FlareRemoval mode...")
        inference_params = {
            "prompt": [
                "",
                "",
                ""
            ],
            "negative_prompt": "",
            "seed": 1,
            "num_inference_steps": 50,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "height": 432,
            "width": 768,
            "num_frames": 21,
            "denoising_strength": 1.0,
            "tiled": True,
            "tile_size": [30, 52],
            "tile_stride": [15, 26],
            "is_inference": True,
            "training_mode": "FlareRemoval",
            "inference_com": None if com is None else com[0],
        }
        video_dict = self.model.pipe(**inference_params)
        os.makedirs(f"{self.output_path}/epoch_{self.epoch_id}/FlareRemoval", exist_ok=True)
        
        # 保存预测 flare 和 clean
        if "flare" in video_dict:
            flare_gen = video_dict["flare"]
            self._tensor2video(flare_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/FlareRemoval/flare_gen_{count}.mp4")
        if "clean" in video_dict:
            clean_gen = video_dict["clean"]
            self._tensor2video(clean_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/FlareRemoval/clean_gen_{count}.mp4")
        
        # 保存拼接视频（条件 com 与预测 flare、clean 并排）
        videos_to_cat = []
        if com is not None:
            videos_to_cat.append(com[0][0] * 0.5 + 0.5)
        if "flare" in video_dict:
            videos_to_cat.append(video_dict["flare"] * 0.5 + 0.5)
        if "clean" in video_dict:
            videos_to_cat.append(video_dict["clean"] * 0.5 + 0.5)
        if videos_to_cat:
            cat_video = torch.cat(videos_to_cat, dim=-1)
            self._tensor2video(cat_video, f"{self.output_path}/epoch_{self.epoch_id}/FlareRemoval/cat_all_{count}.mp4")
        
        # ========== 3. Generation: 从纯噪声生成所有三个分支 ==========
        print("\n[Validation] Testing Generation mode...")
        inference_params = {
            "prompt": [
                val_batch["prompt"][0] + "Lighting effect: " + val_batch["flare_prompt"][0],
                val_batch["flare_prompt"][0],
                val_batch["prompt"][0]
            ],
            "negative_prompt": "",
            "seed": 1,
            "num_inference_steps": 50,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "height": 432,
            "width": 768,
            "num_frames": 21,
            "denoising_strength": 1.0,
            "tiled": True,
            "tile_size": [30, 52],
            "tile_stride": [15, 26],
            "is_inference": True,
            "training_mode": "Generation",
        }
        video_dict = self.model.pipe(**inference_params)
        os.makedirs(f"{self.output_path}/epoch_{self.epoch_id}/Generation", exist_ok=True)
        
        # 保存生成的三个分支
        if "com" in video_dict:
            com_gen = video_dict["com"]
            self._tensor2video(com_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/Generation/com_gen_{count}.mp4")
        if "flare" in video_dict:
            flare_gen = video_dict["flare"]
            self._tensor2video(flare_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/Generation/flare_gen_{count}.mp4")
        if "clean" in video_dict:
            clean_gen = video_dict["clean"]
            self._tensor2video(clean_gen * 0.5 + 0.5, f"{self.output_path}/epoch_{self.epoch_id}/Generation/clean_gen_{count}.mp4")
        
        # 保存拼接视频（生成的 com、flare、clean 并排）
        videos_to_cat = []
        if "com" in video_dict:
            videos_to_cat.append(video_dict["com"] * 0.5 + 0.5)
        if "flare" in video_dict:
            videos_to_cat.append(video_dict["flare"] * 0.5 + 0.5)
        if "clean" in video_dict:
            videos_to_cat.append(video_dict["clean"] * 0.5 + 0.5)
        if videos_to_cat:
            cat_video = torch.cat(videos_to_cat, dim=-1)
            self._tensor2video(cat_video, f"{self.output_path}/epoch_{self.epoch_id}/Generation/cat_all_{count}.mp4")
        
        print("\n[Validation] Composition, FlareRemoval & Generation tested successfully!")
        return 0.0
       
    def train(self):
        self._call_callbacks("on_train_begin")

        for epoch_id in range(self.num_epochs):
            self.epoch_id = epoch_id
            self._call_callbacks("on_epoch_begin")
            
            # 使用accelerator.main_process_<y_bin_401>来包装tqdm，确保只在主进程显示进度条
            pbar = tqdm(self.dataloader, disable=not self.accelerator.is_main_process, desc=f"Epoch {epoch_id+1}/{self.num_epochs}")
            
            for batch_idx, data in enumerate(pbar):
                try:
                    self._call_callbacks("on_step_begin")
                    with self.accelerator.accumulate(self.model):
                        self.optimizer.zero_grad()
                        self.loss = self.model(data) # 将loss保存在self中，供callback使用
                        
                        # 检查loss有效性，如果无效则跳过这个batch
                        if not torch.isfinite(self.loss):
                            print(f"⚠️ Batch {batch_idx + 1}: nan or inf loss detected, skipping...")
                            continue
                        
                        self.accelerator.backward(self.loss)
                        self.optimizer.step()               
                        self.scheduler.step()
                    
                    self.global_step += 1
                    if self.accelerator.is_main_process:
                        pbar.set_postfix({"loss": float(self.loss.detach().cpu().item())}) # 在进度条上显示loss
                    self._call_callbacks("on_step_end")
                    
                except Exception as e:
                    print(f"Batch {batch_idx + 1} 处理出错: {e}")
                    if self.accelerator.is_main_process:
                        import traceback
                        traceback.print_exc()
                    # 继续处理下一个batch
                    continue
            
            # 跳过验证推理，仅执行保存权重的回调
            if self.num_epochs == 100:
                if self.epoch_id % 9 == 0: # 每9个epoch做一次验证
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

