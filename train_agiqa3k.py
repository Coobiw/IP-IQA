import os
import shutil
from pathlib import Path

from tqdm import tqdm
import warnings

import argparse
from omegaconf import OmegaConf

import random
import numpy as np
import torch
import torch.distributed as dist

from ipiqa.common.dist_utils import (
    init_distributed_mode,
    main_process,
)
from trainer import Trainer
from ipiqa.processors import load_processor
from ipiqa.datasets.agiqa_datasets import AGIQA3k
from ipiqa.common.registry import registry
from ipiqa.common.logger import setup_logger
from ipiqa.tasks import setup_task

from ipiqa.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
    ConstantLRScheduler,
)  # 加入到注册表里，不用直接使用（由于是from的import形式，optim.py里的所有类都会加入注册表，所以实际上import一个也可以）

import pandas as pd

warnings.filterwarnings('ignore')

def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_config(args):
    cfg_path = Path(args.cfg_path)
    assert cfg_path.suffix == '.yaml', 'config file must be .yaml file'
    config = OmegaConf.load(cfg_path)
    init_distributed_mode(config.run)
    return config

def get_transforms(config) -> dict:
    dataset_cfg = config.dataset

    transforms = {}
    transforms['train'] = load_processor(**dataset_cfg.transform_train)
    transforms['val'] = load_processor(**dataset_cfg.transform_val)

    return transforms

def get_datasets(config,transforms) -> dict:
    def agiqa3k_split_fn(info):
        count = int(0.8 * 300)

        indices = np.random.permutation(300)

        train_info = []
        val_info = []

        for i in range(info.shape[0]):
            image_name = info.iloc[i, 0][:-4]
            image_name_split = image_name.split("_")
            idx = int(image_name_split[-1])

            if idx in indices[:count]:
                train_info.append(i)
            else:
                val_info.append(i)

        train_info = info.iloc[train_info]
        val_info = info.iloc[val_info]

        return train_info, val_info

    dataset_cfg = config.dataset

    datasets = {}
    data_info = dataset_cfg.data_path
    vis_root = dataset_cfg.vis_root
    data_info = pd.read_excel(data_info)

    train_info, val_info = agiqa3k_split_fn(data_info)

    datasets["train"] = AGIQA3k(train_info,transforms['train'],vis_root)
    datasets['val'] = AGIQA3k(val_info,transforms['val'],vis_root)

    return datasets

def get_model(config):
    model_cfg = config.model
    print(registry.list_models())
    model_cls = registry.get_model_class(model_cfg.arch)
    return model_cls.from_config(model_cfg)

def main(config):
    transforms = get_transforms(config)
    datasets = get_datasets(config,transforms)
    model = get_model(config)
    task = setup_task(config)
    job_id = now()

    trainer = Trainer(config,model,datasets,task,job_id)
    return trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path',type=str)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--num_cv',type=int,default=1)
    args = parser.parse_args()

    seed_everything(args.seed)
    config = get_config(args)

    setup_logger()

    metric_lst = []
    results = {}
    for i in range(args.num_cv):
        metric_lst.append(main(config))

    print(metric_lst)

    key_lst = ["agg_metrics",'qual_agg','qual_PLCC','qual_SROCC','qual_KROCC','align_agg','align_PLCC','align_SROCC','align_KROCC']
    value_lst = [0] * len(key_lst)
    l = len(key_lst)

    for i in range(l):
        cur_key = key_lst[i]
        value_lst[i] = sum([metric[cur_key] for metric in metric_lst])
        results[cur_key] = value_lst[i] / args.num_cv

    print(results)