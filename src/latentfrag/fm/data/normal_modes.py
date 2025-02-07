import warnings
import numpy as np
import prody
prody.confProDy(verbosity='none')
from prody import parsePDB, ANM


def pdb_to_normal_modes(pdb_file, num_modes=5, nmax=5000):
    """
    Compute normal modes for a PDB file using an Anisotropic Network Model (ANM)
    http://prody.csb.pitt.edu/tutorials/enm_analysis/anm.html (accessed 01/11/2023)
    """
    protein = parsePDB(pdb_file, model=1).select('calpha')

    if len(protein) > nmax:
        warnings.warn("Protein is too big. Returning zeros...")
        eig_vecs = np.zeros((len(protein), 3, num_modes))

    else:
        # build Hessian
        anm = ANM('ANM analysis')
        anm.buildHessian(protein, cutoff=15.0, gamma=1.0)

        # calculate normal modes
        anm.calcModes(num_modes, zeros=False)

        # only use slowest modes
        eig_vecs = anm.getEigvecs()  # shape: (num_atoms * 3, num_modes)
        eig_vecs = eig_vecs.reshape(len(protein), 3, num_modes)
        # eig_vals = anm.getEigvals()  # shape: (num_modes,)

    nm_dict = {}
    for atom, nm_vec in zip(protein, eig_vecs):
        chain = atom.getChid()
        resi = atom.getResnum()
        name = atom.getName()
        nm_dict[(chain, resi, name)] = nm_vec.T

    return nm_dict
