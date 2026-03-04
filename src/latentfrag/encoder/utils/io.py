from rdkit import Chem


def read_xyz_file(path):
    with open(path, 'r') as f:
        atoms = [line.split() for line in f.readlines()[2:]]

    types = [a[0] for a in atoms]
    coords = [(float(a[1]), float(a[2]), float(a[3])) for a in atoms]

    return types, coords


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)


def pointcloud_to_pdb(filename, coords, feature=None, show_as='HETATM'):

    assert len(coords) <= 99999
    assert show_as in {'ATOM', 'HETATM'}
    if feature is not None:
        assert feature.min() >= 0 and feature.max() <= 100

    out = ""
    for i in range(len(coords)):
        # format:
        # 'ATOM'/'HETATM' <atom serial number> <atom name>
        # <alternate location indicator> <residue name> <chain>
        # <residue sequence number> <code for insertions of residues>
        # <x> <y> <z> <occupancy> <b-factor> <segment identifier>
        # <element symbol> <charge>
        feature_i = 0.0 if feature is None else feature[i]
        out += f"{show_as:<6}{i:>5} {'SURF':<4}{' '}{'ABC':>3} {'A':>1}{1:>4}" \
               f"{' '}   {coords[i, 0]:>8.3f}{coords[i, 1]:>8.3f}" \
               f"{coords[i, 2]:>8.3f}{1.0:>6}{feature_i:>6.2f}      {'':>4}" \
               f"{'H':>2}{'':>2}\n"

    with open(filename, 'w') as f:
        f.write(out)
