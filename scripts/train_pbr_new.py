import argparse,os,yaml
from datetime import datetime
from torch.utils.data import DataLoader 
from registry import (
    DATASET_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY, 
    SCHEDULER_REGISTRY, CALLBACK_REGISTRY,TRAINER_REGISTRY
)
from src.dataset.util import collate_fn, SingleDatasetBatchSampler, DistributedSingleDatasetBatchSampler # 这一行没那么优雅，先将就着
now = datetime.now().strftime("%Y%m%d_%H%M%S9+1320")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/wan2_1_14b_t2v_pbr_lora_video_w_image_w_prompt_train_v6.yaml", help="Path to the main training config file.")
    args = parser.parse_args()

    # load yaml
    with open(args.config, 'r') as f: # load yaml as a dict
        config = yaml.safe_load(f)
    if 'environment_variables' in config and config['environment_variables']:
        print("--- 正在根据配置文件设置环境变量 ---")
        for key, value in config['environment_variables'].items():
            # 注意：环境变量的值必须是字符串
            str_value = str(value)
            os.environ[key] = str_value
            print(f"✅已设置: {key} = {str_value}")
        print("------------------------------------")
    # 实例化
    print("--- 正在根据配置构建训练组件 ---")

    def build_dataset_and_dataloader(dataset_config, dataloader_config, dataset_name_prefix=""):
        """
        构建数据集和DataLoader的辅助函数
        
        Args:
            dataset_config: 数据集配置字典
            dataloader_config: DataLoader配置字典
            dataset_name_prefix: 数据集名称前缀（用于日志）
        
        Returns:
            dataset, dataloader
        """
        import torch.distributed as dist
        from torch.utils.data import ConcatDataset
        
        if dataset_config['name'] == 'ComposedDataset':
            # 组合数据集
            datasets = []
            for dataset_item in dataset_config["params"]["datasets"]:
                dataset_class = DATASET_REGISTRY[dataset_item["name"]]
                datasets.append(dataset_class(**dataset_item["params"]))
                print(f"✅ {dataset_name_prefix}子数据集 '{dataset_item['name']}' 已创建")
            
            dataset = ConcatDataset(datasets)
            print(f"✅ {dataset_name_prefix}组合数据集 '{dataset_config['name']}' 已创建，总大小: {len(dataset)}")
            
            # 使用 SingleDatasetBatchSampler
            batch_sampler = SingleDatasetBatchSampler(
                    dataset, 
                    batch_size=dataloader_config['batch_size'], 
                    shuffle=dataloader_config['shuffle'], 
                    drop_last=dataloader_config['drop_last']
            )
            print(f"✅ {dataset_name_prefix}使用 SingleDatasetBatchSampler")
        
            dataloader = DataLoader(
                dataset, 
                collate_fn=collate_fn, 
                batch_sampler=batch_sampler, 
                num_workers=dataloader_config['num_workers'], 
                pin_memory=dataloader_config['pin_memory'], 
                persistent_workers=dataloader_config['persistent_workers']
            )
            print(f"✅ {dataset_name_prefix}DataLoader 已创建")
        else:
            # 单个数据集
            dataset_class = DATASET_REGISTRY[dataset_config['name']]
            dataset = dataset_class(**dataset_config['params'])
            print(f"✅ {dataset_name_prefix}数据集 '{dataset_config['name']}' 已创建，大小: {len(dataset)}")
            
            dataloader = DataLoader(
                dataset, 
                collate_fn=collate_fn, 
                batch_size=dataloader_config['batch_size'],
                shuffle=dataloader_config.get('shuffle', False),
                drop_last=dataloader_config.get('drop_last', False),
                num_workers=dataloader_config.get('num_workers', 0),
                pin_memory=dataloader_config.get('pin_memory', False),
                persistent_workers=dataloader_config.get('persistent_workers', False)
            )
            print(f"✅ {dataset_name_prefix}DataLoader 已创建")
        
        return dataset, dataloader

    # 构建 video_dataset 和 video_dataloader
    video_dataset, video_dataloader = build_dataset_and_dataloader(
        config['video_dataset'], 
        config['dataloader'],
        dataset_name_prefix="[video] "
    )

    # 构建 frame_dataset 和 frame_dataloader
    frame_dataset, frame_dataloader = build_dataset_and_dataloader(
        config['frame_dataset'], 
        config['dataloader'],
        dataset_name_prefix="[frame] "
    )

    # 构建 t2RAIN_dataset 和 t2RAIN_dataloader
    t2RAIN_dataset, t2RAIN_dataloader = build_dataset_and_dataloader(
        config['t2RAIN_dataset'], 
        config['dataloader'],
        dataset_name_prefix="[t2RAIN] "
    )
    # 构建 frame_R2AIN_dataset 和 frame_R2AIN_dataloader
    frame_R2AIN_dataset, frame_R2AIN_dataloader = build_dataset_and_dataloader(
        config['frame_R2AIN_dataset'], 
        config['dataloader'],
        dataset_name_prefix="[frame_R2AIN] "
    )
    # 构建 val_dataset 和 val_dataloader
    val_dataset, val_dataloader = build_dataset_and_dataloader(
        config['val_dataset'], 
        config['val_dataloader'],
        dataset_name_prefix="[val] "
    )
    # 构建模型 (这里的'model'指的是您的'WanTrainingModule')
    model_class = MODEL_REGISTRY[config['model']['name']] # 假设您也注册了训练模块
    model = model_class(**config['model']['params'])
    print(f"✅ 模型 '{config['model']['name']}' 已创建")

    # 构建优化器
    optimizer_class = OPTIMIZER_REGISTRY[config['optimizer']['name']]
    
    # 由于未知原因，lr变成了str，这里执行str2float
    # ===============================================================
    lr_value = config['optimizer']['params']['lr']
    if isinstance(lr_value, str):
        config['optimizer']['params']['lr'] = float(lr_value)
    # ===============================================================
  
    optimizer = optimizer_class(model.parameters(), **config['optimizer']['params'])
    print(f"✅ 优化器 '{config['optimizer']['name']}' 已创建")

    # 构建学习率调度器
    scheduler_class = SCHEDULER_REGISTRY[config['scheduler']['name']]
    # CosineAnnealingLR 需要传入 optimizer 和 T_max

    # 由于未知原因，eta_min变成了str，这里执行str2float
    eta_min_value = config['scheduler']['params']['eta_min']
    if isinstance(eta_min_value, str):
        config['scheduler']['params']['eta_min'] = float(eta_min_value)
    scheduler_params = config['scheduler']['params']
    # 计算总的训练步数（使用video_dataloader作为主要训练数据）
    scheduler_params['T_max'] = config['trainer_config']['num_epochs'] * len(video_dataloader)
    scheduler = scheduler_class(optimizer, **scheduler_params)
    print(f"✅ 调度器 '{config['scheduler']['name']}' 已创建")

    # 构建回调列表
    callbacks = []
    # callbacks的保存路径现在和时间相关了
    if 'callbacks' in config:
        for cb_config in config['callbacks']:
            cb_class = CALLBACK_REGISTRY[cb_config['name']]
            # 如果log_dir和output_path是字符串，则添加时间戳
            params = cb_config.get('params', {})
            if 'logging_dir' in params and isinstance(params['logging_dir'], str):
                params['logging_dir'] = params['logging_dir'] + f"_{now}"
            if 'output_path' in params and isinstance(params['output_path'], str):
                params['output_path'] = params['output_path'] + f"_{now}"
            callbacks.append(cb_class(**cb_config.get('params', {})))
    print(f"✅ 回调已创建: {[type(cb).__name__ for cb in callbacks]}")

    # 启动
    print("\n--- 组件构建完毕，正在初始化Trainer ---")
    # 每个任务，不同的trainer
    trainer_class = TRAINER_REGISTRY[config['trainer_name']]  
    trainer = trainer_class(
        model=model,
        optimizer=optimizer,
        video_dataloader=video_dataloader,  # 使用video_dataloader作为主要训练数据
        scheduler=scheduler,
        callbacks=callbacks,
        val_dataloader=val_dataloader,
        # 传递额外的dataloader供训练器使用
        frame_dataloader=frame_dataloader,
        t2RAIN_dataloader=t2RAIN_dataloader,
        frame_R2AIN_dataloader=frame_R2AIN_dataloader,
        **config['trainer_config'] # 传入num_epochs等通用训练参数
    )

    print("\n--- 开始训练！ ---")
    trainer.train()

if __name__ == "__main__":
    main()