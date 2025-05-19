"""Training and evaluation"""

import hydra
import os
import numpy as np
import run_train
import utils
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict
import yaml


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    
    # Load dataset-specific config
    with open('dataset_config.yaml', 'r') as file:
        dataset_configs = yaml.safe_load(file)
    dataset_config = dataset_configs[cfg.dataset_name]

    # Update training size if provided via environment variable
    if cfg.train_size is not None:
        dataset_config['train_size'] = int(cfg.train_size)
        
    if cfg.n_iters is not None:
        dataset_config['n_iters'] = int(cfg.n_iters)
    
    # Get actual dataset size for directory naming
    train_size = dataset_config.get('train_size', 'full')
    
    # Update work directory to include dataset size
    work_dir = cfg.work_dir.replace('_sizefull', f'_size{train_size}')
    
    # Update cfg with dataset-specific settings
    cfg.data.train = dataset_config['train']
    cfg.data.valid = dataset_config['valid']
    cfg.training.n_iters = dataset_config['n_iters']
    cfg.work_dir = work_dir
    
    
    ngpus = cfg.ngpus
    if "load_dir" in cfg:
        hydra_cfg_path = os.path.join(cfg.load_dir, ".hydra/hydra.yaml")
        hydra_cfg = OmegaConf.load(hydra_cfg_path).hydra

        cfg = utils.load_hydra_config_from_run(cfg.load_dir)
        
        work_dir = cfg.work_dir
        utils.makedirs(work_dir)
    else:
        hydra_cfg = HydraConfig.get()
        work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
        utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

	# Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    #logger = utils.get_logger(os.path.join(work_dir, "logs")) # modified by foobar
    logger = utils.get_logger(os.path.join(work_dir, "logs"),debug=True)

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    try:
        mp.set_start_method("forkserver")
        mp.spawn(run_train.run_multiprocess, args=(ngpus, cfg, port), nprocs=ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()