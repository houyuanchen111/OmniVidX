import torch
class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift # 非线性去噪, shift大于1,前期多，后期少
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step: # 为了结果更加鲁棒
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep): # timestep 越小越不加噪
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        elif isinstance(timestep, (list, tuple)):
            timestep = torch.tensor(timestep)

        # 支持 timestep 是 list 或 tensor，其中不同元素对应不同模态的时间步
        if isinstance(timestep, torch.Tensor) and timestep.dim() > 0:
            # timestep 是向量，每个元素对应不同模态的时间步
            if timestep.dim() == 1:
                # timestep: (num_modalities,)
                num_modalities = timestep.shape[0]
                if original_samples.shape[0] != num_modalities:
                    raise ValueError(f"timestep 的长度 ({num_modalities}) 应该等于 original_samples 的第一维大小 ({original_samples.shape[0]})")

                # 为每个模态计算对应的 sigma
                timestep_ids = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
                sigmas = self.sigmas[timestep_ids]  # shape: (num_modalities,)

                # 扩展 sigmas 的维度以匹配 original_samples 和 noise 的形状
                # original_samples 和 noise 的形状: (num_modalities, C, T, H, W)
                sigmas = sigmas.view(num_modalities, *([1] * (original_samples.dim() - 1))).to(original_samples.device).to(original_samples.dtype)

                sample = (1 - sigmas) * original_samples + sigmas * noise
                return sample, sigmas
            else:
                raise ValueError(f"timestep 张量维度应该为 1，当前维度为 {timestep.dim()}")
        else:
            # timestep 是标量，保持原有逻辑（向后兼容）
            timestep_id = torch.argmin((self.timesteps - timestep).abs())
            sigma = self.sigmas[timestep_id]
            sample = (1 - sigma) * original_samples + sigma * noise
            return sample, sigma
    

    def training_target(self, sample, noise, timestep):
        # 和timestep无关
        target = noise - sample
        return target
    

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
