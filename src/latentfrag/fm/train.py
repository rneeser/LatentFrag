import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import yaml

from latentfrag.fm.models.lightning import LatentFrag
from latentfrag.fm.utils.gen_utils import (set_deterministic,
                                        disable_rdkit_logging,
                                        dict_to_namespace,
                                        namespace_to_dict)


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        arg_dict[key] = dict_to_namespace(value)

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__

        if isinstance(value, dict):
            # update dictionaries recursively
            value = merge_configs(config[key], value)

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
    p.add_argument('--overfit', action='store_true')
    p.add_argument('--const_zt', action='store_true')
    args = p.parse_args()

    set_deterministic(seed=42)
    disable_rdkit_logging()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)

    # TODO write resume if job restated by cluster
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']
        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)
    # for backward compatibility
    if not hasattr(args, 'rotate'):
        args.rotate = False
    if not hasattr(args, 'connect'):
        args.connect = True

    if args.debug:
        print('DEBUG MODE')
        args.wandb_params.mode = 'disabled'
        args.train_params.enable_progress_bar = True
        args.train_params.num_workers = 0

    if args.overfit:
        print('OVERFITTING MODE')

    if args.const_zt:
        print('CONSTANT ZT MODE')

    out_dir = Path(args.train_params.logdir, args.run_name)
    args.eval_params.outdir = out_dir

    pl_module = LatentFrag(
        pocket_representation=args.pocket_representation,
        train_params=args.train_params,
        loss_params=args.loss_params,
        eval_params=args.eval_params,
        predictor_params=args.predictor_params,
        simulation_params=args.simulation_params,
        coarse_stage=args.coarse_stage,
        rotate=args.rotate,
        connect=args.connect,
        debug=args.debug,
        overfit=args.overfit,
        const_zt=args.const_zt,
    )

    logger = pl.loggers.WandbLogger(
        save_dir=args.train_params.logdir,
        project='LatentFrag',
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume='must' if args.resume is not None else False,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoints_root_dir = Path(out_dir, 'checkpoints')
    checkpoints_valid_dir = Path(checkpoints_root_dir, 'valid_and_connected')
    checkpoints_valid_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoints_root_dir,
            filename="best_train_epoch_{epoch:03d}",
            monitor="loss/train",
            save_top_k=1,
            save_last=True,
            mode="min",
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        ),
    ]

    # For learning rate logging
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=args.train_params.n_epochs,
        logger=logger,
        callbacks=checkpoint_callbacks,
        enable_progress_bar=args.train_params.enable_progress_bar,
        check_val_every_n_epoch=args.eval_params.eval_epochs,
        num_sanity_val_steps=args.train_params.num_sanity_val_steps,
        accumulate_grad_batches=args.train_params.accumulate_grad_batches,
        accelerator='gpu' if args.train_params.gpus > 0 else 'cpu',
        devices=args.train_params.gpus if args.train_params.gpus > 0 else 'auto',
        strategy=('ddp' if args.train_params.gpus > 1 else None),
        log_every_n_steps=1,
        detect_anomaly=False,
    )

    logger.experiment.config.update({'as_dict': namespace_to_dict(args)}, allow_val_change=True)
    trainer.fit(model=pl_module, ckpt_path=ckpt_path)

    # # run test set
    # result = trainer.test(ckpt_path='best')
