import subprocess
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import tempfile

from latentfrag.fm.utils.constants import vdw_radii_msms


def pdb_to_xyzr(struct, xyzr_file, filter=lambda atom: True):
    """
    xyzr files contain one line per atom with:
        x-coord y-coord z-coord radius
    """

    with open(xyzr_file, 'w') as f:
        for atom in struct.get_atoms():
            if filter(atom):
                x, y, z = atom.get_coord()
                # TODO: add context-dependent radii
                #  (e.g. based on chemical bonds within the amino acid)
                f.write(f"{x:.04f} {y:.04f} {z:.04f} {vdw_radii_msms[atom.element]}\n")


def parse_msms_vertices(vert_file):
    """
    Returns:
        coords: float, (n, 3)
        normals: float, (n, 3)
    """
    with open(vert_file, 'r') as f:
        content = f.readlines()
        n = int(content[2].split()[0])
        coords = np.zeros([n, 3])
        normals = np.zeros([n, 3])
        for i, line in enumerate(content[3:]):
            x, y, z, u, v, w = map(float, line.split()[:6])
            coords[i] = [x, y, z]
            normals[i] = [u, v, w]

    return coords, normals


def parse_msms_faces(face_file):
    """
    Returns:
        faces: int, (m, 3)
    """
    with open(face_file, 'r') as f:
        content = f.readlines()
        n = int(content[2].split()[0])
        faces = np.zeros([n, 3], dtype=int)
        for idx, line in enumerate(content[3:]):
            i, j, k = map(int, line.split()[:3])
            faces[idx] = [i - 1, j - 1, k - 1]

    return faces


def pdb_to_points_normals(pdb, msms_bin, resolution=3.0, filter=None,
                          return_faces=False):
    """
    Use MSMS software to compute surface point clouds.
    Documentation: https://ccsb.scripps.edu/msms/documentation/
    :param pdb: PDB file or BioPython object
    :param resolution: triangulation density [vertex/Angstrom^2]
    :return: coords, normals
    """

    if isinstance(pdb, Path):
        # Read the input structure
        parser = PDBParser(QUIET=True)
        pdb = parser.get_structure('', str(pdb))

    # temporary files
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_stem = Path(tmpdirname, f'msms')
        tmp_xyzr = tmp_stem.with_suffix('.xyzr')
        tmp_vert = tmp_stem.with_suffix('.vert')
        tmp_face = tmp_stem.with_suffix('.face')

        # process
        if filter is None:
            pdb_to_xyzr(pdb, tmp_xyzr)
        else:
            pdb_to_xyzr(pdb, tmp_xyzr, filter=filter)

        out = subprocess.run(f'{msms_bin} -density {resolution} '
                             f'-hdensity 3.0 -probe 1.5 -if {tmp_xyzr} '
                             f'-of {tmp_stem}', shell=True, capture_output=True)

        if 'MSMS terminated normally' not in str(out.stdout):
            raise RuntimeError(f"MSMS didn't terminate normally. "
                               f"Output message:\n{out}")

        # read results
        coords, normals = parse_msms_vertices(tmp_vert)
        if return_faces:
            faces = parse_msms_faces(tmp_face)

    return coords, normals, faces if return_faces else coords, normals


def default_filter(res):
    return is_aa(res)