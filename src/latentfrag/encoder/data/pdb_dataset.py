import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from rdkit import Chem, RDLogger
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Batch

from latentfrag.encoder.utils.data import featurize_ligand, deduplicate_and_combine, find_nearby_points_and_merge

RDLogger.DisableLog('rdApp.*')


class FragDataset(Dataset):
    def __init__(
            self,
            datadir: str,
            split: str,
            surface_fn: callable,
            pdb_dir: str,
            nci_dir: str,
            thresh: float,
            subset: int = -1,
            use_ev: bool = False,
            frag_only_loss: str = None,
            predict_nci: bool = False,
            processed_dir: str = None,
            preprocessing: bool = False,
            ):
        self.surface_fn = surface_fn
        self.split = split

        self.pdb_dir = pdb_dir
        self.nci_dir = nci_dir
        self.preprocessing = preprocessing
        thresh_str = str(thresh).replace('.', '_')
        self.processed_dir = processed_dir if processed_dir is not None \
            else Path(datadir, f'processed_thresh{thresh_str}')
        if frag_only_loss is not None:
            self.processed_dir = Path(str(self.processed_dir) + f'_{frag_only_loss}')

        self.thresh = thresh
        self.use_ev = use_ev
        self.frag_only_loss = frag_only_loss
        self.predict_nci = predict_nci

        # load interaction csv
        csv_file = Path(datadir, f"index_{split}.csv")
        table = pd.read_csv(csv_file)

        if self.frag_only_loss is not None:
            if self.frag_only_loss == 'tanimoto':
                self.dissim_uuids = np.load(Path(datadir, f'dissimilar_uuids_{split}.npy'), allow_pickle=True).item()
            else:
                NotImplementedError(f"Loss {frag_only_loss} not implemented")

        if subset > 0:
            # sample subset based on pdb and chain pairs
            table = table.head(subset)
            grouped = table.groupby(['pdb', 'chain'])
            sorted_groups = grouped['num_atoms'].sum().sort_values()
            remove = sorted_groups[sorted_groups == 1]
            table = table[~table.set_index(['pdb', 'chain']).index.isin(remove.index)]
            self.processed_dir = Path(str(self.processed_dir) + f'_subset{subset}')

        table = table[
            ['uuid', 'pdb', 'chain', 'cluster', 'split']]

        self.clusters = {}
        self.uuid2smi = {}
        for uuid, pdbid, chain, cluster, split  in tqdm(table.values):

            cluster = int(float(cluster))
            pdbid = pdbid.upper()
            pdb_file = Path(self.pdb_dir, f"{pdbid}_{chain}.pdb")
            if not cluster in self.clusters:
                self.clusters[cluster] = defaultdict(set)
            self.clusters[cluster][pdb_file].add(uuid)

        self.cluster_ids = list(self.clusters.keys())

        if self.use_ev:
            frag_fn = 'fragments_ev.sdf'
        else:
            frag_fn = 'fragments.sdf'
        self.sdf_all = Chem.SDMolSupplier(str(Path(datadir, frag_fn)), sanitize=True)

    def __len__(self):
        return len(self.clusters)

    def _select_from_cluster(self, cluster):
        sel_idx = random.randint(0, len(cluster.keys()) - 1)
        return list(cluster.items())[sel_idx]

    def _process_item(self, pdb_file, uuid):
        # Compute surface point cloud and pocket labels
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("", str(pdb_file))[0]

        sample = self.surface_fn(model)
        sample['name'] = pdb_file.name

        sample["batch"] = torch.zeros(len(sample['xyz']), dtype=int)

        if sample['seq_length'] > 1024:
            raise RuntimeError("Sequence length too long")

        ligand_coords = []
        ligand_types = []
        ligand_bonds = []
        ligand_bond_types = []
        ligand_mol_feats = []
        used_uuids = []
        for idx in uuid:
            rdmol = self.sdf_all[idx]

            ligand = featurize_ligand(rdmol, use_ev=self.use_ev)

            # assert there is contact (between surface (not atom xyz) and ligand)
            if torch.all(torch.cdist(sample['xyz'].cpu(), ligand['xyz']) > self.thresh):
                # skip this fragment
                continue

            used_uuids.append(idx)

            ligand_coords.append(ligand['xyz'])
            ligand_types.append(ligand['types'])
            ligand_bonds.append(ligand['bonds'])
            ligand_mol_feats.append(ligand['mol_feats'])
            ligand_bond_types.append(ligand['bond_types'])

        if len(ligand_coords) == 0:
            raise RuntimeError("no ligand found")

        idx_offset = [0]
        for coord in ligand_coords:
            idx_offset.append(idx_offset[-1] + len(coord))

        sample["ligand_xyz"] = torch.cat(ligand_coords, dim=0)
        sample["ligand_types"] = torch.cat(ligand_types, dim=0)
        sample["ligand_bonds"] = torch.cat(
            [x + idx_offset[i] for i, x in enumerate(ligand_bonds)], dim=1)
        sample["ligand_bond_types"] = torch.cat(ligand_bond_types)
        sample["ligand_mol_feats"] = torch.cat(ligand_mol_feats)
        sample["ligand_batch"] = torch.zeros(len(sample["ligand_xyz"]))
        sample["ligand_sizes"] = torch.tensor([len(x) for x in ligand_coords])
        sample["ligand_mask"] = torch.arange(
            len(ligand_coords)).repeat_interleave(sample["ligand_sizes"])
        sample["processed_uuids"] = used_uuids

        # sanity check: bonds should only connect atoms in the same molecule
        assert torch.all(sample["ligand_mask"][sample["ligand_bonds"][0]] ==
                         sample["ligand_mask"][sample["ligand_bonds"][1]])

        if self.frag_only_loss is not None:
            neg_ligand_coords = []
            neg_ligand_types = []
            neg_ligand_bonds = []
            neg_ligand_bond_types = []
            neg_ligand_mol_feats = []
            for uuid_i in used_uuids:
                neg_uuids = self.dissim_uuids[uuid_i]
                neg_uuid = random.choice(neg_uuids)
                rdmol = self.sdf_all[int(neg_uuid)]
                neg_ligand = featurize_ligand(rdmol, use_ev=self.use_ev)

                neg_ligand_coords.append(neg_ligand['xyz'])
                neg_ligand_types.append(neg_ligand['types'])
                neg_ligand_bonds.append(neg_ligand['bonds'])
                neg_ligand_mol_feats.append(neg_ligand['mol_feats'])
                neg_ligand_bond_types.append(neg_ligand['bond_types'])

            idx_offset = [0]
            for coord in neg_ligand_coords:
                idx_offset.append(idx_offset[-1] + len(coord))

            sample["neg_ligand_xyz"] = torch.cat(neg_ligand_coords, dim=0)
            sample["neg_ligand_types"] = torch.cat(neg_ligand_types, dim=0)
            sample["neg_ligand_bonds"] = torch.cat(
                [x + idx_offset[i] for i, x in enumerate(neg_ligand_bonds)], dim=1)
            sample["neg_ligand_bond_types"] = torch.cat(neg_ligand_bond_types)
            sample["neg_ligand_batch"] = torch.zeros(len(sample["neg_ligand_xyz"]))
            sample['neg_ligand_sizes'] = torch.tensor([len(x) for x in neg_ligand_coords])
            sample["neg_ligand_mask"] = torch.arange(
                len(neg_ligand_coords)).repeat_interleave(sample["neg_ligand_sizes"])
            sample['neg_ligand_mol_feats'] = torch.cat(neg_ligand_mol_feats)

        return sample

    def __getitem__(self, idx):
        try:
            cluster = self.clusters[self.cluster_ids[idx]]
            pdb_file, uuid = self._select_from_cluster(cluster)
            processed_cluster_path = Path(self.processed_dir, f"cluster_{self.cluster_ids[idx]}.pt")
            if not self.preprocessing or not processed_cluster_path.exists():
                item = self._process_item(pdb_file, uuid)
            else:
                processed_cluster = torch.load(processed_cluster_path)
                item = processed_cluster[pdb_file.stem]

            item = self.postprocess(item, pdb_file)

        except (KeyError, ValueError, RuntimeError, FileNotFoundError) as e:
            # except (KeyError, ValueError, FileNotFoundError) as e:
            print(type(e).__name__, e)
            # replace bad item with a random one
            rand_idx = random.randint(0, len(self) - 1)
            item = self[rand_idx]

        return item

    def postprocess(self, sample, pdb_file):
        nci_coords = []
        nci_ohe = []
        uuid = sample['processed_uuids']
        for i, idx in enumerate(uuid):
            # for now predict only from protein side
            if self.predict_nci:
                nci_file = Path(self.nci_dir, f"{pdb_file.stem}_{idx}.pt")
                nci = torch.load(nci_file)
                nci_coords.append(nci['prot_coords'])
                nci_ohe.append(nci['nci_ohe'])

        if self.predict_nci:
            nci_coords = torch.cat(nci_coords, dim=0)
            nci_ohe = torch.cat(nci_ohe, dim=0)
            nci_coords_dedup, nci_ohe_dedup = deduplicate_and_combine(nci_coords, nci_ohe)
            nci_feats, nci_mask = find_nearby_points_and_merge(nci_coords_dedup, nci_ohe_dedup, sample['xyz'], 2.0)
            sample['nci_ohe'] = nci_feats
            sample['nci_mask'] = nci_mask

        return sample


    @classmethod
    def collate_fn(cls, batch):

        out = {}
        for prop in batch[0].keys():

            # PyTorch Geometric objects
            if isinstance(batch[0][prop], Data):
                out[prop] = Batch.from_data_list([x[prop] for x in batch])

            elif prop == 'name':
                out[prop] = [x[prop] for x in batch]

            elif 'batch' in prop:
                out[prop] = torch.cat([i * torch.ones(len(x[prop]), dtype=int)
                                       for i, x in enumerate(batch)], dim=0)

            elif prop == "ligand_bonds":
                idx_offset = torch.tensor([0] + [sum(x["ligand_sizes"]) for x in batch]).cumsum(0)
                out[prop] = torch.cat([x[prop] + idx_offset[i] for i, x in enumerate(batch)], dim=1)

            elif prop == "neg_ligand_bonds":
                idx_offset = torch.tensor([0] + [sum(x["neg_ligand_sizes"]) for x in batch]).cumsum(0)
                out[prop] = torch.cat([x[prop] + idx_offset[i] for i, x in enumerate(batch)], dim=1)

            elif prop == "ligand_mask":
                idx_offset = torch.tensor([0] + [len(x["ligand_sizes"]) for x in batch]).cumsum(0)
                out[prop] = torch.cat([x[prop] + idx_offset[i] for i, x in enumerate(batch)])

            elif prop == "neg_ligand_mask":
                idx_offset = torch.tensor([0] + [len(x["neg_ligand_sizes"]) for x in batch]).cumsum(0)
                out[prop] = torch.cat([x[prop] + idx_offset[i] for i, x in enumerate(batch)])

            elif prop == "nci_mask":
                idx_offset = torch.tensor([0] + [len(x["xyz"]) for x in batch]).cumsum(0)
                out[prop] = torch.cat([x[prop] + idx_offset[i] for i, x in enumerate(batch)])

            elif prop == "residue_ids" and 'esm' in batch[0].keys():
                idx_offset = torch.tensor([0] + [len(x['esm']) for x in batch]).cumsum(0)
                out[prop] = torch.cat([x[prop] + idx_offset[i] for i, x in enumerate(batch)], dim=0)

            elif prop == "processed_uuids":
                continue

            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out

    def preprocess(self):

        if not self.processed_dir.exists():
            self.processed_dir.mkdir(exist_ok=True)

        for cluster_id, cluster in tqdm(self.clusters.items()):
            outfile = Path(self.processed_dir, f"cluster_{cluster_id}.pt")

            if outfile.exists():
                continue

            processed = {}
            for pdb_file, ligand_names in cluster.items():
                try:
                    processed[pdb_file.stem] = self._process_item(pdb_file, ligand_names)
                except (RuntimeError, FileNotFoundError) as e:
                    print(e)
                    continue

            # Save result
            torch.save(processed, outfile)
