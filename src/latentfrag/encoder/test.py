import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl

from latentfrag.encoder.models.lightning_modules import FragEmbed


parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
parser.add_argument('--device', type=str, default=None)
args = parser.parse_args()

if args.device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = args.device

# Load model
model = FragEmbed.load_from_checkpoint(args.checkpoint, map_location=device, strict=False)

run_name = args.checkpoint.parent.parent.name

logger = pl.loggers.WandbLogger(
    project='dmasif-ligand',
    name=f'test_{run_name}',
    id=run_name,
    save_dir=args.checkpoint.parent.parent,
    group='FragEmbed',
    mode='online')

# Run test
trainer = pl.Trainer(gpus=0 if device == 'cpu' else 1,
                      logger=logger,)
trainer.test(model=model)
