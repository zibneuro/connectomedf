import os
import glob
import numpy as np

import h01_util_geometry
import h01_constants
import h01_util


def writeSynapsesPerCube(outfolder, synapsesPerCube):
    for cubeId, synapses in synapsesPerCube.items():
        filename = os.path.join(outfolder, "{}.csv".format(cubeId))

        if(not os.path.exists(filename)):
            with open(filename, "w") as f:
                f.write("pre_id,post_id,pre_celltype,post_celltype,pre_compartment,post_compartment,exc_inh_classification\n")
        
        with open(filename, "a") as f:
            for synapse in synapses:
                f.write("{},{},{},{},{},{},{}\n".format(*synapse[3:]))


def writeCubesMeta(filename, cubeId_origin):
    cubeIds = list(cubeId_origin.keys())
    cubeIds.sort()
    with open(filename, "w") as f:
        f.write("cube_id,origin_x,origin_y,origin_z\n")
        for cubeId in cubeIds:
            origin = cubeId_origin[cubeId]
            f.write("{},{},{},{}\n".format(cubeId, *origin))


def gridSynapses(gridDescriptor, boundsDescriptor, classifiedConnectionsOnly = False):        
    inputFolder = os.path.join(h01_util.getBaseFolder(), "synapses", "postprocessed")
    outputFolder = os.path.join(h01_util.getBaseFolder(), "synapses", "gridded_{}_{}".format(boundsDescriptor, gridDescriptor))    
    h01_util.makeCleanDir(outputFolder)
    cubesMetaFile = os.path.join(h01_util.getBaseFolder(), "meta", "cubes_{}.csv".format(gridDescriptor))

    h01_util_geometry.setGridSize(gridDescriptor)
    boxMin, boxMax = h01_constants.getSubvolume(boundsDescriptor)
    gridBounds = h01_util_geometry.getShiftedGridBounds(boxMin, boxMax)

    synapseFiles = glob.glob(os.path.join(inputFolder, "*.csv"))
    allCubeIds = set()
    for synapseFile in synapseFiles:        
        print(synapseFile)

        synapses = np.loadtxt(synapseFile, skiprows=1, delimiter=",").astype(int)
        # 0 x,
        # 1 y,
        # 2 z,
        # 3 pre_id,
        # 4 post_id,
        # 5 pre_celltype,
        # 6 post_celltype,
        # 7 pre_compartment,
        # 8 post_compartment,
        # 9 exc_inh_classification

        positions = synapses[:, 0:3]
        cubeIds = h01_util_geometry.getCubeIdsFromShiftedGridBounds(gridBounds, positions)

        synapsesPerCube = {}
        for i in range(0, synapses.shape[0]):
            preCelltype = synapses[i,5]
            postCelltype = synapses[i,6]
            isClassifiedConnection = preCelltype != -1 and postCelltype != -1
            if(not classifiedConnectionsOnly or isClassifiedConnection):
                cubeId = cubeIds[i]
                allCubeIds.add(cubeId)
                if(cubeId not in synapsesPerCube):
                    synapsesPerCube[cubeId] = []            
                synapsesPerCube[cubeId].append(synapses[i,:])

        writeSynapsesPerCube(outputFolder, synapsesPerCube)

    cubeId_origin = {}
    for cubeId in allCubeIds:
        cubeId_origin[cubeId] = h01_util_geometry.getCubeOriginForCubeId(gridBounds, cubeId)
    writeCubesMeta(cubesMetaFile, cubeId_origin)



if __name__ == "__main__":
    #gridSynapses("25000-25000-25000", "complete-volume")
    gridSynapses("100000-100000-87500", "complete-volume", classifiedConnectionsOnly=True)
