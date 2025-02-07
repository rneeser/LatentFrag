import os
import numpy as np
from pymol import cmd, stored
from pymol.cgo import SPHERE, NORMAL, COLOR, BEGIN, END, TRIANGLES, VERTEX, LINEWIDTH, LINES


colorDict = {
    "sky": [COLOR, 0.0, 0.76, 1.0],
    "sea": [COLOR, 0.0, 0.90, 0.5],
    "yellowtint": [COLOR, 0.88, 0.97, 0.02],
    "hotpink": [COLOR, 0.90, 0.40, 0.70],
    "greentint": [COLOR, 0.50, 0.90, 0.40],
    "blue": [COLOR, 0.0, 0.0, 1.0],
    "green": [COLOR, 0.0, 1.0, 0.0],
    "yellow": [COLOR, 1.0, 1.0, 0.0],
    "orange": [COLOR, 1.0, 0.5, 0.0],
    "red": [COLOR, 1.0, 0.0, 0.0],
    "black": [COLOR, 0.0, 0.0, 0.0],
    "white": [COLOR, 1.0, 1.0, 1.0],
    "gray": [COLOR, 0.9, 0.9, 0.9],
}


def bwr_gradient(vals):
    """ Blue-white-red gradient """
    max = np.max(vals)
    min = np.min(vals)

    # normalize values
    vals -= (max + min) / 2
    if min >= max:
        vals *= 0
    else:
        vals *= 2 / (max - min)

    blue_vals = np.copy(vals)
    blue_vals[vals >= 0] = 0
    blue_vals = abs(blue_vals)
    red_vals = vals
    red_vals[vals < 0] = 0
    r = 1.0 - blue_vals
    g = 1.0 - (blue_vals + red_vals)
    b = 1.0 - red_vals
    return np.stack(([COLOR] * len(vals), r, g, b)).T


def load_npy(npy_file, dotSize=0.2):

    # data must contain keys 'xyz', 'normals' and 'faces'
    # optionally 'features' can contain another dictionary defining
    # features for each vertex as key-value pairs
    data = np.load(npy_file, allow_pickle=True).item()

    basename = os.path.splitext(os.path.basename(npy_file))[0]
    group_names = ""

    verts = data['xyz']
    normals = data['normals']
    faces = data['faces'].astype(int)
    features = data['features']

    # Draw vertices
    obj = []
    for vert in verts:
        colorToAdd = colorDict['white']
        obj.extend(colorToAdd)
        obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])

    name = basename + "_coords"
    group_names += name
    cmd.load_cgo(obj, name, 1.0)

    # Draw normals
    obj = []
    for v_idx in range(len(verts)):
        colorToAdd = colorDict["white"]
        x1, y1, z1 = verts[v_idx]
        nx, ny, nz = normals[v_idx]
        x2, y2, z2 = x1 + nx, y1 + ny, z1 + nz

        obj.extend([LINEWIDTH, 2.0])
        obj.extend([BEGIN, LINES])
        obj.extend(colorToAdd)
        obj.extend([VERTEX, x1, y1, z1])
        obj.extend([VERTEX, x2, y2, z2])
        obj.append(END)

    name = basename + "_normals"
    group_names += " "
    group_names += name
    cmd.load_cgo(obj, name, 1.0)

    # Draw triangles (faces)
    obj = []
    for tri in faces:
        pairs = [[tri[0], tri[1]], [tri[0], tri[2]], [tri[1], tri[2]]]
        colorToAdd = colorDict["gray"]
        for pair in pairs:
            vert1 = verts[pair[0]]
            vert2 = verts[pair[1]]
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
    name = basename + "_mesh"
    group_names += " "
    group_names += name
    cmd.load_cgo(obj, name, 1.0)

    # Draw features
    for feat_name, feat_val in features.items():
        obj = []
        colors = bwr_gradient(feat_val)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(colors[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(colors[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(colors[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = basename + "_" + feat_name
        group_names += " "
        group_names += name
        cmd.load_cgo(obj, name, 1.0)

        # group the resulting objects
        cmd.group(basename + '_surface', group_names)


# ------------------------------------------------------------------------------
cmd.extend("loadnpy", load_npy)
