#!/bin/bash -l

python src/latentfrag/fm/sample.py \
    --checkpoint /path/to/checkpoints/fm.ckpt \
    --outdir /path/to/outdir \
    --n_samples 2 \
    --frag_library /path/to/fragments_min_8/library \
    --library_sdf /path/tod/fragments_min_8/fragments_ev.sdf \
    --pdb demo_data/1MVC_A.pdb \
    --msms_bin /path/to/msms.x86_64Linux2.2.6.1 \
    --trained_encoder /path/to/checkpoints/encoder.ckpt  \
    --reference_ligand demo_data/1MVC_BM6.sdf \
    --sample_gt_size \
    --skip_protonation