from typing import Tuple, List, Union
import tempfile
import subprocess
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, KekulizeException, AtomKekulizeException
import torch
import torch.nn.functional as F
import numpy as np

from latentfrag.fm.data.data_utils import TensorDict
from latentfrag.fm.utils.gen_utils import namespace_to_dict
from latentfrag.fm.analysis.interactions_plip import INTERACTION_LIST, prepare_mol, combine_mol_pdb, run_plip, read_plip

from latentfrag.encoder.utils.data import featurize_ligand
from latentfrag.encoder.models.lightning_modules import FragEmbed

GNINA_TRANSLATIONS = {
    'Affinity': 'vina_score',
    'minimizedAffinity': 'vina_score',
    'CNNaffinity': 'gnina_score',
    'CNNscore': 'cnn_score',
    'RMSD': 'minimized_rmsd'
}


class SimpleFragmentEvaluator:
    ID = 'frag_metrics'

    def __call__(self,
                 preds: Tuple,
                 targets: Tuple,
                 smiles_preds: List[str] = None,
                 smiles_targets: List[str] = None) -> dict:
        """
        Args:
            preds: tuple containing the predicted fragments embeddings and coordinates
            targets: ltuple containing the target fragments embeddings and coordinates
            smiles_preds: list of SMILES strings of the predicted fragments
            smiles_targets: list of SMILES strings of the target fragments

        Returns:
            metrics (dict): dictionary of metrics
        """
        RDLogger.DisableLog('rdApp.*')
        self.check_format(preds)
        self.check_format(targets)
        results = self.evaluate(preds, targets, smiles_preds, smiles_targets)
        return self.add_id(results)

    def add_id(self, results):
        if self.ID is not None:
            return {f'{self.ID}.{key}': value for key, value in results.items()}
        else:
            return results

    def evaluate(self, preds, targets, smile_preds=None, smile_targets=None):
        # TODO get rot_vec
        if smile_targets is not None:
            mols_targets = [Chem.MolFromSmiles(smile) for smile in smile_targets]
            mols_preds = [Chem.MolFromSmiles(smile) for smile in smile_preds]

        # get index for closest predicted fragments to target fragments
        dists = torch.cdist(preds[0], targets[0])
        closest_geom_idx = dists.argmin(dim=1)

        # get index for most similiar target fragments to predicted fragments
        similar_latent_idx = [torch.argmax(F.cosine_similarity(pred.unsqueeze(0), targets[1])).item() for pred in preds[1]]

        # sort preds according to closest_pred_idx
        com_targets_geom = targets[0][closest_geom_idx]
        emb_targets_geom = targets[1][closest_geom_idx]
        mols_targets_geom = [mols_targets[i] for i in closest_geom_idx]
        com_pred = preds[0]
        emb_pred = preds[1]

        # sort targets according to similar_pred_idx
        com_targets_latent = targets[0][similar_latent_idx]
        emb_targets_latent = targets[1][similar_latent_idx]
        mols_targets_latent = [mols_targets[i] for i in similar_latent_idx]

        # calculate metrics
        rmsd_com_geom = dists.min(dim=1).values
        rmsd_com_latent = [self.rmsd(com1,com2) for com1, com2 in zip(com_targets_latent, com_pred)]
        cossim_embedding_geom = F.cosine_similarity(emb_targets_geom, emb_pred, dim=1)
        cossim_embedding_latent = F.cosine_similarity(emb_targets_latent, emb_pred, dim=1)
        tanimoto_geom = [self.tanimoto(mol_pred, mol_gt) for mol_pred, mol_gt in zip(mols_targets_geom, mols_preds)]
        tanimoto_latent = [self.tanimoto(mol_pred, mol_gt) for mol_pred, mol_gt in zip(mols_targets_latent, mols_preds)]
        # rotation_difference = self.rotation_diff(emb_preds, emb_gt)
        diff_num_frags = abs(com_targets_geom.shape[0] - com_pred.shape[0])

        results = {
            'RMSD_COM_avg': rmsd_com_geom.mean().item(),
            'RMSD_COM_min': rmsd_com_geom.min().item(),
            'RMSD_COM_latent_avg': (sum(rmsd_com_latent) / len(rmsd_com_latent)).item(),
            'RMSD_COM_latent_min': min(rmsd_com_latent).item(),
            'cossim_embedding_avg': cossim_embedding_geom.mean().item(),
            'cossim_embedding_max': cossim_embedding_geom.max().item(),
            'cossim_embedding_latent_avg': cossim_embedding_latent.mean().item(),
            'cossim_embedding_latent_max': cossim_embedding_latent.max().item(),
            'tanimoto_geom_avg': sum(tanimoto_geom) / len(tanimoto_geom),
            'tanimoto_geom_max': max(tanimoto_geom),
            'tanimoto_latent_avg': sum(tanimoto_latent) / len(tanimoto_latent),
            'tanimoto_latent_max': max(tanimoto_latent),
            'diff_num_fragments': diff_num_frags
        }

        return results

    @staticmethod
    def check_format(fragments):
        assert isinstance(fragments, tuple)
        assert len(fragments) == 2
        assert isinstance(fragments[0], torch.Tensor)
        assert isinstance(fragments[1], torch.Tensor)

    @staticmethod
    def rmsd(coord_a, coord_b):
        return torch.sqrt(torch.mean(torch.sum((coord_a-coord_b)**2, axis=0)))

    @staticmethod
    def tanimoto(mol_a, mol_b):
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @property
    def dtypes(self):
        return self.add_id(self._dtypes)

    @property
    def _dtypes(self):
        return {
            'RMSD_COM_avg': float,
            'RMSD_COM_min': float,
            'RMSD_COM_latent_avg': float,
            'RMSD_COM_latent_min': float,
            'cossim_embedding_avg': float,
            'cossim_embedding_max': float,
            'cossim_embedding_latent_avg': float,
            'cossim_embedding_latent_max': float,
            'tanimoto_geom_avg': float,
            'tanimoto_geom_max': float,
            'tanimoto_latent_avg': float,
            'tanimoto_latent_max': float,
            'diff_num_fragments': int,
        }

class ComplexFragmentEvaluator:
    """
    Evaluator comparing sampled fragments to all possible fragments in the pocket
    instead of only the corresponding ones of the data point.
        This only makes sense when training in connect = False mode.
    """
    ID = 'frag_metrics_complex'

    def __init__(self,
                 frag_encoder: str,
                 use_ev: bool = False,
                 frag_encoder_params: dict = None,
                 ref_dir: str = None,
                 ) -> None:
        self.frag_encoder = frag_encoder
        self.use_ev = use_ev
        self.ref_dir = ref_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset_params = namespace_to_dict(frag_encoder_params.dataset_params)
        surface_params = namespace_to_dict(frag_encoder_params.surface_params)
        self.model = FragEmbed.load_from_checkpoint(frag_encoder,
                                                    map_location=self.device,
                                                    dataset_params=dataset_params,
                                                    surface_params=surface_params)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self,
                    frags_pred_path: Path,
                    frags_ref_path: Path = None) -> dict:
        RDLogger.DisableLog('rdApp.*')
        frags_pred = Chem.SDMolSupplier(str(frags_pred_path))[0]
        frags_pred = Chem.GetMolFrags(frags_pred, asMols=True, sanitizeFrags=False)
        if frags_ref_path is None:
            fn = frags_pred_path.parent.stem + '_frags.sdf'
            frags_ref_path = [f for f in Path(self.ref_dir).iterdir() if f.name.endswith(fn)][0]
            frags_ref = Chem.SDMolSupplier(str(frags_ref_path))[0]
            frags_ref = Chem.GetMolFrags(frags_ref, asMols=True, sanitizeFrags=False)

        results, sim_corr = self.evaluate(frags_pred, frags_ref)
        return self.add_id(results), sim_corr

    def add_id(self, results):
        if self.ID is not None:
            return {f'{self.ID}.{key}': value for key, value in results.items()}
        else:
            return results

    def evaluate(self, frags_pred, frags_ref):
        # get geometric mean of fragments ("center of mass")
        com_pred = torch.tensor(np.array([self.get_com(mol) for mol in frags_pred]))
        com_gt = torch.tensor(np.array([self.get_com(mol) for mol in frags_ref]))
        # get latent embeddings of fragments
        latent_pred = torch.concat(
            [self.get_embedding(mol)['desc_global'].detach().cpu() for mol in frags_pred])
        latent_gt = torch.concat(
            [self.get_embedding(mol)['desc_global'].detach().cpu() for mol in frags_ref])

        # RMSDs
        rmsds = torch.cdist(com_pred, com_gt)
        min_dists, min_dist_indices = torch.min(rmsds, dim=1)
        rmsds_long = rmsds.view(-1)
        # Cosine similarities
        cossims = F.cosine_similarity(latent_pred.unsqueeze(1), latent_gt, dim=-1)
        max_cossims, max_cossims_indices = torch.max(cossims, dim=1)
        cossims_long = cossims.view(-1)

        # closest fragments in cross metric
        dists_latent = rmsds[torch.arange(rmsds.shape[0]), max_cossims_indices]
        cossims_euclidean = cossims[torch.arange(cossims.shape[0]), min_dist_indices]

        # metrics
        dists_avg_euclidean = min_dists.mean().item()
        dists_avg_latent = dists_latent.mean().item()
        cossims_avg_latent = max_cossims.mean().item()
        cossims_avg_euclidean = cossims_euclidean.mean().item()

        results = {
            'RMSD_avg_euclidean': dists_avg_euclidean,
            'RMSD_avg_latent': dists_avg_latent,
            'cossim_avg_latent': cossims_avg_latent,
            'cossim_avg_euclidean': cossims_avg_euclidean
        }

        return results, (rmsds_long, cossims_long)

    @staticmethod
    def get_com(mol):
        return mol.GetConformer().GetPositions().mean(axis=0)

    def get_embedding(self, mol):
        frag_feat = featurize_ligand(mol, self.use_ev)

        frag_feat = TensorDict(**frag_feat)
        frag_feat['batch'] = torch.zeros(len(frag_feat['xyz']), dtype=int)
        frag_feat = frag_feat.to(self.device)
        frag_feat['desc'], frag_feat['desc_global'] = self.model.ligand_encoder(
                frag_feat["xyz"], frag_feat["types"], frag_feat["batch"],
                frag_feat["bonds"], frag_feat["bond_types"], frag_feat["mol_feats"], return_global=True)
        return frag_feat

    @property
    def dtypes(self):
        return self.add_id(self._dtypes)

    @property
    def _dtypes(self):
        return {
            'RMSD_avg_euclidean': float,
            'RMSD_avg_latent': float,
            'cossim_avg_latent': float,
            'cossim_avg_euclidean': float
        }


class FragNCIEvaluator:
    ID = 'frag_nci'

    def __init__(self,
                 gnina: str = 'gnina',
                 plip_exec: str = 'plipcmd',
                 reference_ligand_dir: str = None,
                 state: str = 'dock_whole'):
        self.gnina = gnina
        self.plip_exec = plip_exec
        self.reference_ligand_dir = reference_ligand_dir
        self.state = state
        assert self.state in ['docked', 'dock_whole', 'dock_partial'], 'Invalid state'

    @property
    def default_profile(self):
        int_profile = {i: 0 for i in INTERACTION_LIST}
        docking_profile = {i: None for i in ['vina_score',
                                             'gnina_score',
                                             'frag_rmsd',
                                             'minimized_rmsd',
                                             'com_rmsd',
                                             'cnn_score',
                                             'vina_efficiency',
                                             'gnina_efficiency']}
        return {**int_profile, **docking_profile}

    def __call__(self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path]):
        RDLogger.DisableLog('rdApp.*')
        self.check_format(molecule, protein)
        results = self.evaluate(molecule, protein)
        return self.add_id(results)

    def add_id(self, results):
        if self.ID is not None:
            if isinstance(results, List):
                new_results = []
                for elem in results:
                    new_results.append({f'{self.ID}_{self.state}.{key}': value for key, value in elem.items()})
                return new_results
            else:
                return {f'{self.ID}_{self.state}.{key}': value for key, value in results.items()}
        else:
            return results

    def evaluate(self, molecules, protein):
        if isinstance(molecules, (str, Path)):
            docked_fn = Path(molecules).parent / f'{Path(molecules).stem}_docked.sdf'
        else:
            docked_fn = None
        molecules = self.load_molecules(molecules)
        all_profiles = []
        for n, mol in enumerate(molecules):
            if self.state != 'docked' and docked_fn is not None:
                docked_fn_idx = docked_fn.parent / f'{docked_fn.stem}_{n}_{self.state}.sdf'
            else:
                docked_fn_idx = None
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    gnina_scores, docked_molecule = self.dock_molecule(mol, protein, tmpdir, save_fn=docked_fn_idx)
                except Exception as e:
                    profile = self.default_profile
                    docked_molecule = None
                if docked_molecule is not None:
                    try:
                        profile = self.get_nci_profile(docked_molecule, protein, tmpdir)
                    except Exception as e:
                        profile = self.default_profile
                        print(f'Failed to evaluate molecule {n}: {e}')
                    profile.update(gnina_scores)
            profile['idx'] = n
            profile['sdf_docked_fn'] = str(docked_fn_idx) if docked_fn_idx is not None else None
            profile['pdb_fn'] = str(protein)
            all_profiles.append(profile)

        return all_profiles

    def get_nci_profile(self, molecule, protein, tmpdir):
        profile = self.default_profile
        mol_pdb = prepare_mol(molecule, Path(tmpdir))
        complex_pdb = Path(tmpdir, 'complex.pdb')
        combine_mol_pdb(mol_pdb, protein, str(complex_pdb))
        plip_out = run_plip(complex_pdb, Path(tmpdir), plip_exec=str(self.plip_exec))
        profile = read_plip(plip_out / 'report.xml')
        return profile

    def dock_molecule(self, molecule, protein, tmpdir, save_fn=None):
        molecule_fn = self.save_molecule(molecule, sdf_path=Path(tmpdir, 'molecule.sdf'))
        out_mol = Path(tmpdir, 'docked_molecule.sdf')

        if self.state == 'docked':
            args = '--minimize'
            save_fn = out_mol
        elif self.state == 'dock_whole':
            condition_1 = lambda f: f.name.startswith(Path(protein).parent.stem[:6])
            condition_2 = lambda f: f.suffix == '.sdf'
            condition_3 = lambda f: 'frags' not in f.name
            ref_ligand = [f for f in Path(self.reference_ligand_dir).iterdir() if condition_1(f) and condition_2(f) and condition_3(f)][0]
            args = f'--autobox_ligand {ref_ligand}'
        elif self.state == 'dock_partial':
            args = f'--autobox_ligand {str(molecule_fn)} --autobox_extend 1'

        if not save_fn.exists() or self.state == 'docked':
            gnina_cmd = f'{self.gnina} -r {str(protein)} -l {str(molecule_fn)} {args} -o {str(out_mol)} --seed 42'
            gnina_out = subprocess.run(gnina_cmd, shell=True, capture_output=True, text=True)

        gnina_scores = {'vina_score': None,
                        'gnina_score': None,
                        'frag_rmsd': None,
                        'com_rmsd': None,
                        'cnn_score': None,
                        'minimized_rmsd': None,}

        if not save_fn.exists():
            docked_mol = Chem.SDMolSupplier(str(out_mol))[0]
            self.save_molecule(docked_mol, save_fn)
        else:
            docked_mol = Chem.SDMolSupplier(str(save_fn))[0]

        all_gnina_scores = docked_mol.GetPropsAsDict()
        for key, value in all_gnina_scores.items():
            gnina_score_key = GNINA_TRANSLATIONS.get(key, None)
            if gnina_score_key is not None:
                gnina_scores[gnina_score_key] = value

        if gnina_scores['minimized_rmsd'] is None:
            gnina_scores['minimized_rmsd'] = self.get_rmsd(molecule, docked_mol, align=True)
        gnina_scores['frag_rmsd'] = self.get_rmsd(molecule, docked_mol)

        ori_com = self.get_com(molecule)
        docked_com = self.get_com(docked_mol)
        gnina_scores['com_rmsd'] = self.get_rmsd(ori_com, docked_com)

        n_atoms = docked_mol.GetNumAtoms()

        # Additionally computing ligand efficiency
        try:
            gnina_scores['vina_efficiency'] = gnina_scores['vina_score'] / n_atoms if n_atoms > 0 else None
        except:
            gnina_scores['vina_efficiency'] = None
        try:
            gnina_scores['gnina_efficiency'] = gnina_scores['gnina_score'] / n_atoms if n_atoms > 0 else None
        except:
            gnina_scores['gnina_efficiency'] = None
        return gnina_scores, docked_mol

    @staticmethod
    def check_format(molecule, protein):
        assert isinstance(molecule, (str, Path, Chem.Mol)), 'Supported molecule types: str, Path, Chem.Mol'
        assert protein is None or isinstance(protein, (str, Path)), 'Supported protein types: str'

    @staticmethod
    def load_molecules(molecule):
        if isinstance(molecule, (str, Path)):
            suppl = Chem.SDMolSupplier(str(molecule), sanitize=False)
            molecule = [m for m in suppl]
            if len(molecule) == 1:
                molecule = molecule[0]
        else:
            molecule = Chem.Mol(molecule)

        if not isinstance(molecule, List):
            molecules = list(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False))
        return molecules

    @staticmethod
    def save_molecule(molecule, sdf_path):
        if isinstance(molecule, (str, Path)):
            return molecule

        with Chem.SDWriter(str(sdf_path)) as w:
            try:
                w.write(molecule)
            except (RuntimeError, ValueError) as e:
                if isinstance(e, (KekulizeException, AtomKekulizeException)):
                    w.SetKekulize(False)
                    w.write(molecule)
                    w.SetKekulize(True)
                else:
                    w.write(Chem.Mol())
                    print('[FragNCIEvaluator] Error when saving the molecule')

        return sdf_path

    @staticmethod
    def get_rmsd(mol_a, mol_b, align=False):
        if isinstance(mol_a, Chem.Mol):
            mol_b_copy = Chem.Mol(mol_b)
            Chem.Kekulize(mol_b_copy)
            if align:
                return Chem.rdMolAlign.AlignMol(mol_a, mol_b_copy)
            return Chem.rdMolAlign.CalcRMS(mol_a, mol_b_copy)
        elif isinstance(mol_a, np.ndarray):
            return np.sqrt(np.mean(np.sum((mol_a - mol_b)**2, axis=0)))
        else:
            raise ValueError('Invalid input type')

    @staticmethod
    def get_com(mol):
        return mol.GetConformer().GetPositions().mean(axis=0)

    @property
    def dtypes(self):
        return self.add_id(self._dtypes)

    @property
    def _dtypes(self):
        types_dict = {'vina_score': float,
                'gnina_score': float,
                'frag_rmsd': float,
                'minimized_rmsd': float,
                'com_rmsd': float,
                'cnn_score': float,
                'vina_efficiency': float,
                'gnina_efficiency': float,
                'idx': int,
                'sdf_docked_fn': str,
                'pdb_fn': str}
        # add interaction types
        for interaction in INTERACTION_LIST:
            types_dict[interaction] = int

        return types_dict
