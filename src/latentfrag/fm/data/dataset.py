
import random
import warnings
import torch
from torch.utils.data import Dataset

from latentfrag.fm.data.data_utils import TensorDict, collate_entity


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, pt_path, coarse_stage=True, ligand_transform=None, pocket_transform=None, catch_errors=False, unique_pockets=False):

        self.ligand_transform = ligand_transform
        self.pocket_transform = pocket_transform
        self.catch_errors = catch_errors

        self.data = torch.load(pt_path)

        if unique_pockets:
            # deduplicate data based on pocket name
            pocket_names = self.data['pockets']['name']
            pocket_names_unique = list(set(pocket_names))
            pocket_indices = [pocket_names.index(pocket_name) for pocket_name in pocket_names_unique]
            pocket_indices.sort()

            data_dedup = {}
            for key in ['ligands', 'pockets']:
                data_dedup[key] = {}
                for subkey in self.data[key].keys():
                    data_dedup[key][subkey] = [self.data[key][subkey][i] for i in pocket_indices]

            self.data = data_dedup

        # add number of nodes for convenience
        for entity in ['ligands', 'pockets']:
            if entity == 'ligands':
                self.prefix = 'coarse_'
                self.data[entity][f'{self.prefix}size'] = \
                    torch.tensor([len(x) for x in self.data[entity][f'{self.prefix}x']])
                self.data[entity][f'{self.prefix}n_bonds'] = \
                    torch.tensor([len(x) for x in self.data[entity][f'{self.prefix}bond_one_hot']])

            self.data[entity]['size'] = torch.tensor([len(x) for x in self.data[entity]['x']])
            self.data[entity]['n_bonds'] = torch.tensor([len(x) for x in self.data[entity]['bond_one_hot']])

    def __len__(self):
        return len(self.data['ligands']['name'])

    def __getitem__(self, idx):
        data = {}
        data['ligand'] = {key: val[idx] for key, val in self.data['ligands'].items()}
        data['pocket'] = {key: val[idx] for key, val in self.data['pockets'].items()}
        try:
            if self.ligand_transform is not None:
                data['ligand'] = self.ligand_transform(data['ligand'])
            if self.pocket_transform is not None:
                data['pocket'] = self.pocket_transform(data['pocket'])
        except (RuntimeError, ValueError) as e:
            if self.catch_errors:
                warnings.warn(f"{type(e).__name__}('{e}') in data transform. "
                              f"Returning random item instead")
                # replace bad item with a random one
                rand_idx = random.randint(0, len(self) - 1)
                return self[rand_idx]
            else:
                raise e
        return data

    @staticmethod
    def collate_fn(batch_pairs, prefix='', ligand_transform=None):

        out = {}
        for entity in ['ligand', 'pocket']:
            batch = [x[entity] for x in batch_pairs]

            if entity == 'ligand' and ligand_transform is not None:
                max_size = max(x[f'{prefix}size'].item() for x in batch)
                # TODO: might have to remove elements from batch if processing fails, warn user in that case
                batch = [ligand_transform(x, max_size=max_size) for x in batch]

            out[entity] = TensorDict(**collate_entity(batch))

        return out
