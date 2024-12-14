import os
import sys
import numpy as np

from meshparty import skeleton_io

if __name__ == "__main__":
    skeletonName = "648518346349540053_skeleton.h5"
    skeletonNumpyName = "648518346349540053_skeleton_label.npy"
    cacheFolder = "/srv/public/datasets/VIS-L23/smoothed_skeletons_v185"
    path1 = "/srv/public/datasets/VIS-L23/smoothed_skeletons_v185/" + skeletonName
    path2 = "/srv/public/datasets/VIS-L23/smoothed_skeletons_v185/" + skeletonNumpyName

    vertices, edges, meta, mesh_to_skel_map, vertex_properties, root = skeleton_io.read_skeleton_h5_by_part(path1)    
    sk = skeleton_io.read_skeleton_h5(path1)
    labels = np.load(path2)
    
    print(vertices, vertices.shape)

    radius = np.array(vertex_properties["rs"] + [0])
    print("vertex labels", np.unique(labels))

    print(len(vertex_properties["rs"]))
    print(root)
    print(meta)

    print(labels, labels.shape)

    skeleton_io.export_to_swc(sk, "/tmp/foo.swc", labels, radius=radius, xyz_scaling=1)