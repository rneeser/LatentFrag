#!/bin/bash -l

python src/latentfrag/fm/sample.py \
    --checkpoint /data/rneeser/LatentFrag/checkpoints/fm.ckpt \
    --outdir /home/rebecca/code/LatentFrag/tests/reference_outputs/after \
    --n_samples 2 \
    --frag_library /data/rneeser/LatentFrag/FragsEmbed/fragments_min_8/library \
    --library_sdf /data/rneeser/LatentFrag/FragsEmbed/fragments_min_8/fragments_ev.sdf \
    --pdb demo_data/1MVC_A.pdb \
    --msms_bin /home/rebecca/tools/MSMS/msms.x86_64Linux2.2.6.1 \
    --trained_encoder /data/rneeser/LatentFrag/checkpoints/encoder.ckpt  \
    --reference_ligand demo_data/1MVC_BM6.sdf \
    --sample_gt_size \
    --skip_protonation