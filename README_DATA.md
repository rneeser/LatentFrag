# Data processing

## Preparation for training of both encoder and generative model

1. Query PDB

    Define a list of relevant PDB files on [rcsb.org/search](https://www.rcsb.org/search) and download to `$DATA/pdb_ids.txt`

    ```
    QUERY:
        ( Experimental Method = "X-RAY DIFFRACTION" OR Experimental Method = "ELECTRON MICROSCOPY" OR Experimental Method = "SOLID-STATE NMR" OR Experimental Method = "SOLUTION NMR" )
        AND Refinement Resolution <= 3
        AND Entry Polymer Types = "Protein (only)"
        AND Number of Distinct Non-polymer Entities >= 1
    ```

2.  Download PDB files all relevant PDBs to `DATA=/path/to/datadir`:

    ```shell
    mkdir -p $DATA/pdb
    bash latentfrag/encoder/data/batch_download.sh -f $DATA/pdb_ids.txt -O $DATA/pdb -p
    ```

3. Download precomputed 30% identity clusters:

    ```shell
    wget https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt -O $DATA/clusters-by-entity-30.txt
    ```

4. Download SMILES of all ligands from PDB

    ```shell
    wget http://ligand-expo.rcsb.org/dictionaries/Components-smiles-oe.smi -O $DATA/ligand_expo_smiles.smi
    ```

5. Split PDB files in protein chains and ligands, clean and filter them, define interactions

    ```shell
    python -W ignore latentfrag/encoder/data/process_pdb.py \
                    --pdb_dir $DATA/pdb \
                    --ligand_expo_smiles $DATA/ligand_expo_smiles.smi \
                    --out_chains_dir $DATA/chains \
                    --out_ligands_dir $DATA/ligands \
                    --out_interactions_table $DATA/interactions.csv \
                    --enable_resume
    ```

6. Download [HoloProt data and splits](https://drive.google.com/file/d/1o0_0OM_2PykzQTXCYagdJA2w4zoE4AUt/view?usp=sharing) – save to `$DATA/holoprot_splits`

7. Get mapping between PDB chain and entity ids (for clustering) for processed PDB files

    ```shell
    python latentfrag/encoder/data/build_chain_entity_mapping.py $DATA/chains $DATA/chain2entity.json
    ```

8. Protonate proteins

    ```shell
    python latentfrag/encoder/data/protonate.py --in_dir $DATA/chains --out_dir $DATA/chains_protonated --reduce $REDUCE
    ```

## Encoder

Define some paths in the data directory:

```shell
LIG_DATA=$DATA/ligands
PDB_DATA=$DATA/chains_protonated
PROJ_DIR=$DATA/FragsEmbed
FRAG_DIR=$PROJ_DIR/fragments_min_8
PLIP_DIR=$PROJ_DIR/interaction_fp_min8
```

1. Fragment ligands

    ```shell
    python latentfrag/encoder/data/create_fragments_ev.py \
    --ligands $LIG_DATA \
    --chains $PDB_DATA \
    --out_dir $FRAG_DIR \
    --radius 8 \
    --min_num_atoms 8
    ```

2. Compute interaction profiles

    This will take a while as all interactions need to be computed.

    ```shell
    python latentfrag/encoder/data/interaction_fp.py \
    --output_dir $PLIP_DIR \
    --sdf_file $FRAG_DIR/fragments.sdf \
    --csv_file $FRAG_DIR/index.csv \
    --pdb_dir $PDB_DATA
    ```

3. Get non-covalent interactions per chain

    This will take a while as all interactions need to be computed.

    ```shell
    python latentfrag/encoder/data/extract_ncis.py $PLIP_DIR --out $PLIP_DIR/chain_ncis
    ```

4. Filter fragments

    Remove fragment if it has:
    - not supported elements
    - more than 20 heavy atoms
    - more than 500 Da
    - a maximal ring size of more than 8
    - matching patterns of phosphates and alkyl chains of at least for Carbons
    - no interactions based on PLIP profile

    ```shell
    python latentfrag/encoder/data/postfiltering_fragments.py \
    --csvpath $FRAG_DIR/index.csv \
    --interaction_dir $PLIP_DIR \
    --out $FRAG_DIR/index_filtered.csv
    ```

5. Split into training, validation and test sets

    Splitting is done based on 30% sequence identity clusters.

    ```shell
    python latentfrag/encoder/data/train_test_split_frag.py \
    --pdb_interactions $FRAG_DIR/index_filtered.csv \
    --pdb_entities $DATA/chain2entity.json \
    --holoprot_splits $DATA/holoprot_splits/metadata/identity30_split.json \
    --clusters $DATA/clusters-by-entity-30.txt \
    --output_train $FRAG_DIR/index_train.csv \
    --output_valid $FRAG_DIR/index_valid.csv \
    --output_test $FRAG_DIR/index_test.csv \
    > $FRAG_DIR/train_test_split.log
    ```

6. Get negative fragment pairs based on Tanimoto similarity

    ```shell
    python latentfrag/encoder/data/negative_frag_pair_tanimoto.py --datadir $FRAG_DIR
    ```

## Create fragment library from trained encoder

```shell
MODEL_PATH="<path/to/best/trained/model>"

python latentfrag/encoder/data/create_embedding_library.py \
--datadir $FRAG_DIR \
--out $FRAG_DIR/library \
--ckpt $MODEL_PATH \
--data_source index_filtered_all.csv
```

## Generative model

Define some paths in the data directory:

```shell
LIG_DATA=$DATA/ligands
PDB_DATA=$DATA/chains_protonated
PROJ_DIR=$DATA/FragsEmbed
FRAG_DIR=$PROJ_DIR/fragments_min_8
PLIP_DIR=$PROJ_DIR/interaction_fp_min8
```

1. Split data into train/val/test based on encoder

    ```shell
    python latentfrag/fm/data/pdb_split.py \
    --pdb_interactions $FRAG_DIR/index_filtered_all.csv \
    --pdb_entities $DATA/chain2entity.json \
    --holoprot_splits $DATA/holoprot_splits/metadata/identity30_split.json \
    --clusters $DATA/clusters-by-entity-30.txt \
    --output_dir $FRAG_DIR
    ```

2. Process the data

    This can take a while as all surfaces need to be computed. Define the MSMS bin path `MSMS_BIN=/path/to/msms.x86_64Linux2.2.6.1`.

    ```shell
    EMBEDDER_CKPT="<path/to/best/trained/model>"

    python data/process_pdb_dataset.py $FRAG_DIR $DATA \
    --pocket surface \
    --embedder_ckpt $EMBEDDER_CKPT \
    --min_frag_size 8 \
    --msms_bin $MSMS_BIN
    ```
