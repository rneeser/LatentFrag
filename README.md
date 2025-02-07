# LatentFrag

Structure-based fragment identification in latent space.

This work proposes a protein-fragment encoder that is trained contrastively between protein surfaces and fragments. This latent representation is next used in a generative model based on flow matching where we sample fragments in this learned latent space and their centroids conditioned on the pocket surface.

## Installation

```bash
conda create --name latentfrag python=3.7 -y
conda activate latentfrag

conda install cmake==3.27.4 -y
conda install cudatoolkit=11.3 -y
conda install cudatoolkit-dev=11.7 -c conda-forge -y
conda install cudnn=8.2.1 -y

conda install -c conda-forge openbabel=3.0.0

pip install --upgrade pip
pip cache purge

pip install open3d==0.9.0
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric==2.3.1 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install pykeops==1.5

pip install pandas matplotlib jupyter pytorch-lightning==1.8.0 wandb==0.13.4 rdkit-pypi==2022.9.5 biopython==1.78 pdb-tools==2.5.0 ProDy==2.4.0
pip install plyfile==0.7.2 pyvtk==0.5.18
pip install --no-index torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
conda install -c conda-forge imageio
pip install posebusters==0.2.12
pip install fcd==1.2
pip install useful-rdkit-utils

cd LatentFrag
pip install -e .
```

Test the installation (mainly for pykeops):

```bash
python test_setup.py
```

### Install reduce

Reduce is needed to protonate the protein structures. Follow instructions from [here](https://github.com/rlabduke/reduce) to install reduce.

### Install MSMS

MSMS is needed to compute protein surfaces. Follow instructions from [here](https://ccsb.scripps.edu/msms/downloads/) to install MSMS.

For example for linux unpack the downloaded binary like this:

```bash
tar zxvf msms_i86_64Linux2_2.6.1.tar.gz
```

### Install PLIP

PLIP is needed to compute protein-ligand interactions. Find more information [here](https://plip-tool.biotec.tu-dresden.de/plip-web/plip/index).

```bash
git clone https://github.com/pharmai/plip.git
cd plip
python setup.py install
sudo apt install plip
# check if plipcmd works
plipcmd -h
```

## Data

Download the data from [here](https://figshare.com/s/c28e5d7f9d318305614e). We provide files necessary for inference and model checkpoints for both the encoder and the flow matching model. In order to generate the data for training, please follow the instructions in the `README_DATA.md`.

## Usage

### Encoder

#### Encode fragments

To compute the latent encodings of fragments, a sdf file containing a single fragments, a folder of sdf files or a SMILES string can be provided. For example:

```bash
python src/latentfrag/encoder/predict_frag.py --sdf_file demo_data/1MVC_BM6.sdf --checkpoint /path/to/encoder.ckpt --out_dir /path/to/out_dir
```

#### Encode a protein surface

To compute the latent encodings of a protein surface, a pdb file is needed. The script will compute the protein surface using MSMS and then compute the latent encoding (save under the variable `desc`). Ideally the protein gets protonated using reduced before.

```bash
python src/latentfrag/encoder/predict_protein.py --pdb_file demo_data/1MVC_A.pdb --checkpoint /path/to/encoder.ckpt --out_dir /path/to/out_dir --msms_bin /path/to/msms.x86_64Linux2.2.6.1
```

#### Training encoder

To train the encoder, a example config file is provided in `configs/train_encoder_example.yml`. Please adapt the data paths accordingly.

```bash
python src/latentfrag/encoder/train.py --config configs/train_encoder_example.yml
```

### Generative modeling

#### Fragment identification with Flow Matching

We provide a processed PDB file and the corresponding reference ligand for a sampling demo (folder `demo_data`). There are different ways to define how many fragment per sample to generate:

- by number of reference fragments (`--sample_gt_size`): providing a reference ligand is necessary
- by chosen number (`--num_nodes`)
- by histogram based on pocket size (`--size_histogram`)

The pocket is defined either by a predefined radius around the reference ligand (`--reference_ligand`) or by defining pocket residues (`--residue_ids`). The following script defines the pocket nd the sampled nodes by the ligand. Sampled fragments are not docked and thus are placed with arbitrary orientation at the predicted centroid. Please adjust the paths in the bash script accordingly.

```bash
cd LatentFrag
conda activate latentfrag
configs/demo_sampling.sh
```

#### Training FM

To train the flow matching model, a example config file is provided in `configs/train_flow_matching_example.yml`. Please adapt the data paths accordingly.

```bash
python src/latentfrag/flow_matching/train.py --config configs/train_flow_matching_example.yml
```
