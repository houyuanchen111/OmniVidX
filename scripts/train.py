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
    parser.add_argument('--config', type=str, default="configs/wan2_1_fun_1_3b_control_material_lora_image_train_v0.yaml", help="Path to the main training config file.")
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

    # 构建数据集
    if config['dataset']['name'] == 'ComposedDataset':
        import torch.distributed as dist
        from torch.utils.data import ConcatDataset
        datasets = []
        for dataset_name in config["dataset"]["params"]["datasets"]:
            dataset_class = DATASET_REGISTRY[dataset_name["name"]]
            datasets.append(dataset_class(**dataset_name["params"]))
            print(f"✅ 数据集 '{dataset_name}' 已创建")
        dataset = ConcatDataset(datasets)
        print(f"✅ concat数据集 '{config['dataset']['name']}' 已创建")
        
        # # 检测是否在分布式环境中，选择合适的采样器
        # use_distributed = dist.is_available() and dist.is_initialized()
        # if use_distributed:
        #     batch_sampler = DistributedSingleDatasetBatchSampler(
        #         dataset, 
        #         batch_size=config['dataloader']['batch_size'], 
        #         shuffle=config['dataloader']['shuffle'], 
        #         drop_last=config['dataloader']['drop_last']
        #     )
        #     print(f"✅ 使用 DistributedSingleDatasetBatchSampler (确保每个step所有rank数据来自同一数据集)")
        # else:
        batch_sampler = SingleDatasetBatchSampler(
            dataset, 
            batch_size=config['dataloader']['batch_size'], 
            shuffle=config['dataloader']['shuffle'], 
            drop_last=config['dataloader']['drop_last']
        )
        print(f"✅ 使用 SingleDatasetBatchSampler")
        
        dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_sampler=batch_sampler, num_workers=config['dataloader']['num_workers'], pin_memory=config['dataloader']['pin_memory'], persistent_workers=config['dataloader']['persistent_workers'])
        print(f"✅ DataLoader 已创建")
        # for i, (data, labels) in enumerate(dataloader):
        #     print(f"批次 {i}:")
        #     print(f"  标签: {labels.tolist()}")
            
        #     # # 检查批次内的所有标签是否都相同
        #     # is_homogeneous = len(torch.unique(labels)) == 1
        #     # print(f"  批次内数据来源是否统一? {'是' if is_homogeneous else '否'}")
            
        #     # if i == 4: # 只看前5个批次
        #     #     break
    else:
        dataset_class = DATASET_REGISTRY[config['dataset']['name']]
        dataset = dataset_class(**config['dataset']['params'])
        print(f"✅ 数据集 '{config['dataset']['name']}' 已创建")
        dataloader = DataLoader(dataset, collate_fn=collate_fn, **config['dataloader'])
        print(f"✅ DataLoader 已创建")  
 
    val_dataset_class = DATASET_REGISTRY[config['val_dataset']['name']]
    val_dataset = val_dataset_class(**config['val_dataset']['params'])
    print(f"✅ 验证数据集 '{config['val_dataset']['name']}' 已创建")
    print(f"验证数据集大小: {len(val_dataset)}")
    # 构建DataLoader
  

    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, **config['val_dataloader']) # dataloader 参数需要做出区分，因为pipe.__call__()不支持batch输入
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
    scheduler_params['T_max'] = config['trainer_config']['num_epochs'] * len(dataloader)
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
        dataloader=dataloader,
        scheduler=scheduler,
        callbacks=callbacks,
        val_dataloader=val_dataloader,
        **config['trainer_config'] # 传入num_epochs等通用训练参数
    )

    print("\n--- 开始训练！ ---")
    trainer.train()

if __name__ == "__main__":
    main()