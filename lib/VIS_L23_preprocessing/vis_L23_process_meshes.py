import os
import sys
import numpy as np
import glob 

import vis_L23_util as util

import vtk
from meshparty import trimesh_io
from meshparty import trimesh_vtk

if __name__ == "__main__":
    dataFolder = sys.argv[1]
    inputFolder = os.path.join(dataFolder, "layer23_v185")
    outputFolder = os.path.join(dataFolder, "morphologies_stl")

    util.makeCleanDir(outputFolder)

    files = glob.glob(os.path.join(inputFolder, "*.h5"))
    
    for filename in files:
        print(filename)
        neuronId = int(os.path.basename(filename).split(".")[0])
        outfileName = os.path.join(outputFolder, "{}.stl".format(neuronId))

        meshmeta = trimesh_io.MeshMeta()
        mesh = meshmeta.mesh(filename)
        vtk_polydata = trimesh_vtk.trimesh_to_vtk(mesh.vertices, mesh.faces, None)    

        writer_stl = vtk.vtkSTLWriter()
        writer_stl.SetInputData(vtk_polydata)
        writer_stl.SetFileName(outfileName)
        writer_stl.Write()