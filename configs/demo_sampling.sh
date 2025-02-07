#!/bin/bash -l

python src/latentfrag/fm/sample.py \
    --checkpoint /path/to/LatentFrag/checkpoints/fm.ckpt \
    --outdir /path/to/out_dir \
    --n_samples 10 \
    --frag_library /path/to/LatentFrag/FragsEmbed/fragments_min_8/library \
    --library_sdf /path/to/LatentFrag/FragsEmbed/fragments_min_8/fragments_ev.sdf \
    --pdb demo_data/1MVC_A.pdb \
    --msms_bin /path/to/msms.x86_64Linux2.2.6.1 \
    --trained_encoder /path/to/LatentFrag/checkpoints/encoder.ckpt  \
    --reference_ligand demo_data/1MVC_BM6.sdf \
    --sample_gt_size