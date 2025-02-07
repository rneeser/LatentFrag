
import tempfile
from pathlib import Path
import subprocess
import sys
import os
from copy import deepcopy

from tqdm import tqdm
from rdkit import Chem, RDConfig
import numpy as np
from rdkit.Chem import (Descriptors,
                        Crippen,
                        Lipinski,
                        QED,
                        DataStructs,
                        MolSanitizeException,)


sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # type: ignore


class CategoricalDistribution:
    EPS = 1e-10

    def __init__(self, histogram_dict, mapping):
        histogram = np.zeros(len(mapping))
        for k, v in histogram_dict.items():
            histogram[mapping[k]] = v

        # Normalize histogram
        self.p = histogram / histogram.sum()
        self.mapping = deepcopy(mapping)

    def kl_divergence(self, other_sample):
        sample_histogram = np.zeros(len(self.mapping))
        for x in other_sample:
            # sample_histogram[self.mapping[x]] += 1
            sample_histogram[x] += 1

        # Normalize
        q = sample_histogram / sample_histogram.sum()

        return -np.sum(self.p * np.log(q / (self.p + self.EPS) + self.EPS))


class GaussianDistribution:
    def __init__(self, dist):
        self.mean_p = dist[0,:]
        self.std_p = dist[1,:]

    def kl_divergence(self, other_sample):
        mean_q = other_sample.mean(axis=0)
        std_q = other_sample.std(axis=0)
        return np.sum(np.log(std_q / self.std_p) + \
                       (self.std_p ** 2 + (self.mean_p - mean_q) ** 2) / (2 * std_q ** 2) - 0.5)


def check_mol(rdmol):
    """
    See also: https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    """
    if rdmol is None:
        return 'is_none'

    _rdmol = Chem.Mol(rdmol)
    try:
        Chem.SanitizeMol(_rdmol)
        return 'valid'
    except ValueError as e:
        assert isinstance(e, MolSanitizeException)
        return type(e).__name__


class MolecularProperties:

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = sascorer.calculateScore(rdmol)
        # return round((10 - sa) / 9, 2)  # from pocket2mol
        return sa

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        logp = Crippen.MolLogP(rdmol)
        rule_4 = (logp >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    @staticmethod
    def calculate_mw(rdmol):
        return Descriptors.ExactMolWt(rdmol)

    @staticmethod
    def calculate_heavyatoms(rdmol):
        return rdmol.GetNumHeavyAtoms()

    @classmethod
    def calculate_diversity(cls, pocket_mols):
        if len(pocket_mols) < 2:
            return 0.0

        div = 0
        total = 0
        for i in range(len(pocket_mols)):
            for j in range(i + 1, len(pocket_mols)):
                div += 1 - cls.similarity(pocket_mols[i], pocket_mols[j])
                total += 1
        return div / total

    @staticmethod
    def similarity(mol_a, mol_b):
        # fp1 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_a, 2, nBits=2048, useChirality=False)
        # fp2 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_b, 2, nBits=2048, useChirality=False)
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def evaluate_pockets(self, pocket_rdmols, verbose=False):
        """
        Run full evaluation
        Args:
            pocket_rdmols: list of lists, the inner list contains all RDKit
                molecules generated for a pocket
        Returns:
            QED, SA, LogP, Lipinski (per molecule), and Diversity (per pocket)
        """

        for pocket in pocket_rdmols:
            for mol in pocket:
                Chem.SanitizeMol(mol)  # only evaluate valid molecules

        all_qed = []
        all_sa = []
        all_logp = []
        all_lipinski = []
        all_mw = []
        all_heavyatoms = []
        per_pocket_diversity = []
        for pocket in tqdm(pocket_rdmols):
            all_qed.append([self.calculate_qed(mol) for mol in pocket])
            all_sa.append([self.calculate_sa(mol) for mol in pocket])
            all_logp.append([self.calculate_logp(mol) for mol in pocket])
            all_lipinski.append([self.calculate_lipinski(mol) for mol in pocket])
            all_mw.append([self.calculate_mw(mol) for mol in pocket])
            all_heavyatoms.append([self.calculate_heavyatoms(mol) for mol in pocket])
            per_pocket_diversity.append(self.calculate_diversity(pocket))

        qed_flattened = [x for px in all_qed for x in px]
        sa_flattened = [x for px in all_sa for x in px]
        logp_flattened = [x for px in all_logp for x in px]
        mw_flattened = [x for px in all_mw for x in px]
        heavyatoms_flattened = [x for px in all_heavyatoms for x in px]
        lipinski_flattened = [x for px in all_lipinski for x in px]

        if verbose:
            print(f"{sum([len(p) for p in pocket_rdmols])} molecules from "
                  f"{len(pocket_rdmols)} pockets evaluated.")
            print(f"QED: {np.mean(qed_flattened):.3f} \pm {np.std(qed_flattened):.2f}")
            print(f"SA: {np.mean(sa_flattened):.3f} \pm {np.std(sa_flattened):.2f}")
            print(f"LogP: {np.mean(logp_flattened):.3f} \pm {np.std(logp_flattened):.2f}")
            print(f"Lipinski: {np.mean(lipinski_flattened):.3f} \pm {np.std(lipinski_flattened):.2f}")
            print(f"Molecular weight: {np.mean(mw_flattened):.3f} \pm {np.std(mw_flattened):.2f}")
            print(f"Heavy atoms: {np.mean(heavyatoms_flattened):.3f} \pm {np.std(heavyatoms_flattened):.2f}")
            print(f"Diversity: {np.mean(per_pocket_diversity):.3f} \pm {np.std(per_pocket_diversity):.2f}")

        return all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity

    def __call__(self, rdmols):
        """
        Run full evaluation and return mean of each property
        Args:
            rdmols: list of RDKit molecules
        Returns:
            Dictionary with mean QED, SA, LogP, Lipinski, and Diversity values
        """

        if len(rdmols) < 1:
            return {'QED': 0.0, 'SA': 0.0, 'LogP': 0.0, 'Lipinski': 0.0,
                    'Diversity': 0.0}

        _rdmols = []
        for mol in rdmols:
            try:
                Chem.SanitizeMol(mol)  # only evaluate valid molecules
                _rdmols.append(mol)
            except ValueError as e:
                print("Tried to analyze invalid molecule")
        rdmols = _rdmols

        qed = np.mean([self.calculate_qed(mol) for mol in rdmols])
        sa = np.mean([self.calculate_sa(mol) for mol in rdmols])
        logp = np.mean([self.calculate_logp(mol) for mol in rdmols])
        lipinski = np.mean([self.calculate_lipinski(mol) for mol in rdmols])
        mw = np.mean([self.calculate_mw(mol) for mol in rdmols])
        heavyatoms = np.mean([self.calculate_heavyatoms(mol) for mol in rdmols])
        diversity = self.calculate_diversity(rdmols)

        return {'QED': qed, 'SA': sa, 'LogP': logp, 'Lipinski': lipinski, 'Molecular weight': mw,
                'Heavy atoms': heavyatoms, 'Diversity': diversity}


class MolecularMetrics:
    def __init__(self, connectivity_thresh=1.0):
        self.connectivity_thresh = connectivity_thresh

    @staticmethod
    def is_valid(rdmol):
        if rdmol.GetNumAtoms() < 1:
            return False

        _mol = Chem.Mol(rdmol)
        try:
            Chem.SanitizeMol(_mol)
        except ValueError:
            return False

        return True

    def is_connected(self, rdmol):

        if rdmol.GetNumAtoms() < 1:
            return False

        mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True)

        largest_frag = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())
        if largest_frag.GetNumAtoms() / rdmol.GetNumAtoms() >= self.connectivity_thresh:
            return True
        else:
            return False

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = sascorer.calculateScore(rdmol)
        # return round((10 - sa) / 9, 2)  # from pocket2mol
        return sa

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        logp = Crippen.MolLogP(rdmol)
        rule_4 = (logp >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    @staticmethod
    def calculate_mw(rdmol):
        return Descriptors.ExactMolWt(rdmol)

    @staticmethod
    def calculate_heavyatoms(rdmol):
        return rdmol.GetNumHeavyAtoms()

    def __call__(self, rdmol):
        valid = self.is_valid(rdmol)

        if valid:
            Chem.SanitizeMol(rdmol)

        connected = None if not valid else self.is_connected(rdmol)
        qed = None if not valid else self.calculate_qed(rdmol)
        sa = None if not valid else self.calculate_sa(rdmol)
        logp = None if not valid else self.calculate_logp(rdmol)
        mw = None if not valid else self.calculate_mw(rdmol)
        heavyatoms = None if not valid else self.calculate_heavyatoms(rdmol)
        lipinski = None if not valid else self.calculate_lipinski(rdmol)

        return {
            'valid': valid,
            'connected': connected,
            'qed': qed,
            'sa': sa,
            'logp': logp,
            'lipinski': lipinski,
            'mw': mw,
            'heavyatoms': heavyatoms
        }


class Diversity:
    @staticmethod
    def similarity(fp1, fp2):
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def get_fingerprint(self, mol):
        # fp = AllChem.GetMorganFingerprintAsBitVect(
        #     mol, 2, nBits=2048, useChirality=False)
        fp = Chem.RDKFingerprint(mol)
        return fp

    def __call__(self, pocket_mols):

        if len(pocket_mols) < 2:
            return 0.0

        pocket_fps = [self.get_fingerprint(m) for m in pocket_mols]

        div = 0
        total = 0
        for i in range(len(pocket_fps)):
            for j in range(i + 1, len(pocket_fps)):
                div += 1 - self.similarity(pocket_fps[i], pocket_fps[j])
                total += 1

        return div / total
