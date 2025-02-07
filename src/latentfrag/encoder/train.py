import argparse
from argparse import Namespace
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import torch
import pytorch_lightning as pl
import yaml

from latentfrag.encoder.models.lightning_modules import FragEmbed
from latentfrag.encoder.utils.misc import set_deterministic


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not args.debug:
        # check if same run started less than 96h ago (cluster restarts jobs)
        for run_name in Path(config['logdir']).iterdir():
            if run_name.is_dir() and run_name.name.startswith(config['run_name']):
                old_datetime = ('_').join(run_name.name.split('_')[-2:])
                # check if older than 96 h
                if datetime.strptime(old_datetime, '%Y%m%d_%H%M%S') > datetime.now() - timedelta(hours=96):
                    args.resume = run_name.name
                    break

    config["run_name"] = f'{config["run_name"]}_{start_time}'
    ckpt_path = None

    if args.resume is not None:
        config["run_name"] = args.resume
        ckpt_path = Path(config['logdir'], args.resume, 'checkpoints', 'last.ckpt')
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']
        config = merge_configs(config, resume_config)
        print(f'Resuming experiment {config["run_name"]}', flush=True)
    else:
        print(f'Starting experiment {config["run_name"]}', flush=True)

    if args.debug:
        config['run_name'] = 'debug'
        config['wandb_params']['mode'] = 'disabled'
        config['trainer_params']['enable_progress_bar'] = True
        config['num_workers'] = 0
        # torch.autograd.set_detect_anomaly(True)

    out_dir = Path(config['logdir'], config['run_name'])
    set_deterministic(config['seed'])

    if config['mode'] == 'frag_embed':
        pl_module = FragEmbed(
            surface_encoder=config['surface_encoder'],
            surface_encoder_params=config[f"{config['surface_encoder']}_params"],
            ligand_encoder=config['ligand_encoder'],
            ligand_encoder_params=config[f"{config['ligand_encoder']}_params"],
            lr=config['lr'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            dataset_params=config['dataset_params'],
            surface_parameterization=config['surface'],
            surface_params=config['surface_params'],
            loss_params=config['loss_params'],
            eval_strategies=config['eval_strategies'],
        )
    else:
        raise NotImplementedError()

    logger = pl.loggers.WandbLogger(
        save_dir=config['logdir'],
        project='dmasif-ligand',
        group=config['wandb_params']['group'],
        name=config['run_name'],
        id=config['run_name'],
        resume='must' if args.resume is not None else False,
        entity=config['wandb_params']['entity'],
        mode=config['wandb_params']['mode'],
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        # auto_insert_metric_name=False,
        save_top_k=5,
        save_last=True,
        mode="min",
    )

    n_gpus = config['gpus']
    default_strategy = 'auto' if pl.__version__ >= '2.0.0' else None

    print('Initializing the trainer')
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=('gpu' if n_gpus > 0 else 'cpu'),
        devices=(n_gpus if n_gpus > 0 else 'auto'),
        strategy=('ddp' if n_gpus > 1 else default_strategy),
        **config['trainer_params'],
    )
    if args.resume is None and n_gpus <= 1 and not args.debug:
        print('Get baseline performance')
        trainer.validate(model=pl_module, ckpt_path=ckpt_path)

    print('Start training')
    trainer.fit(model=pl_module, ckpt_path=ckpt_path)

    # # run test set
    # result = trainer.test(ckpt_path='best')
