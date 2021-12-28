import numpy as np


def vec(list):
    """Handy shorthand to make a single-precision float array."""
    return np.array(list, dtype=np.float32)


def normalize(v):
    """Return a unit vector in the direction of the vector v."""
    return v / np.linalg.norm(v)


def from_srgb(img_srgb):
    return np.where(img_srgb > 0.04045, ((img_srgb + 0.055) / 1.055)**2.4, img_srgb / 12.92).astype(np.float32)


def to_srgb(img):
    img_clip = np.clip(img, 0, 1)
    return np.where(img > 0.0031308, (1.055 * img_clip**(1/2.4) - 0.055), 12.92 * img_clip)


def from_srgb8(img_srgb8):
    return from_srgb(img_srgb8 / 255.0)


def to_srgb8(img):
    return np.clip(np.round(255.0 * to_srgb(img)), 0, 255).astype(np.uint8)


def read_obj(f):
    """Read a file in the Wavefront OBJ file format.

    Argument is an open file.
    Returns a tuple of NumPy arrays: (indices, positions, normals, uvs).
    """

    # position, normal, uv, and face data in the order they appear in the file
    f_posns = []
    f_normals = []
    f_uvs = []
    f_faces = []

    # set of unique index combinations that appear in the file
    verts = set()

    # read file
    for words in (line.split() for line in f.readlines()):
        if words[0] == 'v':
            f_posns.append([float(s) for s in words[1:]])
        elif words[0] == 'vn':
            f_normals.append([float(s) for s in words[1:]])
        elif words[0] == 'vt':
            f_uvs.append([float(s) for s in words[1:]])
        elif words[0] == 'f':
            f_faces.append(words[1:])
            for w in words[1:]:
                verts.add(w)

    # there is one vertex for each unique index combo; number them
    vertmap = dict((s, i) for (i, s) in enumerate(sorted(verts)))

    # collate the vertex data for each vertex
    posns = [None] * len(vertmap)
    normals = [None] * len(vertmap)
    uvs = [None] * len(vertmap)
    for k, v in vertmap.items():
        w = k.split('/')
        posns[v] = f_posns[int(w[0]) - 1]
        if len(w) > 1 and w[1]:
            uvs[v] = f_uvs[int(w[1]) - 1]
        if len(w) > 2 and w[2]:
            normals[v] = f_normals[int(w[2]) - 1]

    # set up faces using our ordering
    inds = [[vertmap[k] for k in f] for f in f_faces]

    # convert all to NumPy arrays with the right datatypes
    return (
        np.array(inds, dtype=np.int32),
        np.array(posns, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(uvs, dtype=np.float32)
    )


def read_obj_triangles(f):
    """Read a file in the Wavefront OBJ file format and convert to separate triangles.

    Argument is an open file.
    Returns an array of shape (n, 3, 3) that has the 3D vertex positions of n triangles.
    """

    (i, p, n, t) = read_obj(f)
    return p[i, :]
