import os
import sys
import numpy as np
import pandas as pd

import vis_L23_constants as constants
import vis_L23_util as util


class Gridder():
    def __init__(self):
        self.eps = 1 # nanometer
        self.defaultGridSize = np.array([50000,50000,50000])

    def setPositions(self, xyz):
        self.positions = xyz
        self.min = np.min(self.positions, axis=0)
        self.max = np.max(self.positions, axis=0)
        self.positions_zero_based = self.positions - self.min
        self.extent = np.max(self.positions_zero_based, axis=0)        
        print(self.extent)


    def applyGrid(self, gridSize, sampleFile=None, maxSampleSize=100000):
        gridSize = np.array(gridSize)
        indices = np.floor_divide(self.positions_zero_based, gridSize)
        base = np.max(indices) + 1
        baseFactors = np.array([base**2,base,1])
        indicesScalar = np.dot(indices, baseFactors)         
        indicesUnique, inverseIndices = np.unique(indicesScalar, return_inverse=True)        
        indicesRelabelled = np.arange(indicesUnique.size)[inverseIndices]                        
        
        if(sampleFile is not None):
            n = self.positions.shape[0]
            sampleIndices = np.random.choice(np.arange(n), np.min([n, maxSampleSize]))            
            samples = np.hstack([self.positions[sampleIndices,:],indicesRelabelled[sampleIndices].reshape((-1,1))])
            np.savetxt(sampleFile, samples, fmt="%d")

        self.gridMasks = {} # indexRelabelled -> mask (synapses flat)
        for indexRelabelled in range(0, indicesUnique.size):         
            #print(indexRelabelled)   
            self.gridMasks[indexRelabelled] = np.nonzero(indicesRelabelled == indexRelabelled)
            #if(self.gridMasks[indexRelabelled][0].size > 1):
            #    print("#", self.gridMasks[indexRelabelled][0].size)
        

def loadNeuronIdMapping(somaFile):
    mapping = {}
    with open(somaFile) as f:
        lines = f.readlines()
        for line in lines[1:]:
            neuronId = int(line.rstrip().split(",")[3])
            mappedNeuronId = int(line.rstrip().split(",")[4])
            mapping[neuronId] = mappedNeuronId
    return mapping


if __name__ == "__main__":
    dataFolder = sys.argv[1]
    synapsesBaseFolder = os.path.join(dataFolder, "synapses")
    util.makeCleanDir(synapsesBaseFolder)    
    gridsize = 5000
    synapsesFolder = os.path.join(synapsesBaseFolder, "gridded_{}-{}-{}".format(gridsize, gridsize, gridsize))
    util.makeCleanDir(synapsesFolder)

    mapping = loadNeuronIdMapping(os.path.join(dataFolder, "meta", "soma.csv"))

    gridder = Gridder()
    
    synapses = pd.read_csv(os.path.join(dataFolder, "synapses_flat.csv"), delimiter=",")    
    synapsesFlat = synapses.to_numpy(int)
    positions = synapses[["x", "y", "z"]].to_numpy(int)
    synapseData = synapses[["pre_id", "post_id", "pre_celltype", "post_celltype", "post_compartment"]].to_numpy(int)
    print(synapseData.shape)
    gridder.setPositions(positions)
                
    gridder.applyGrid([gridsize, gridsize, gridsize])
    
    synapsesPerCube = {}
    for cubeId, mask in gridder.gridMasks.items():
        synapseDataCurrent = synapseData[mask]
        for i in range(0, synapseDataCurrent.shape[0]):
            preId = synapseDataCurrent[i, 0]
            postId = synapseDataCurrent[i, 1]
            preCelltype = synapseDataCurrent[i, 2]
            postCelltype = synapseDataCurrent[i, 3]
            postCompartment = synapseDataCurrent[i, 4]
            if(preCelltype != -1 and postCelltype != -1):
                if(cubeId not in synapsesPerCube):
                   synapsesPerCube[cubeId] = []
                synapsesPerCube[cubeId].append((mapping[preId], mapping[postId], preCelltype, postCelltype, postCompartment))
    
    for cubeId, synapses in synapsesPerCube.items():
        filename = os.path.join(synapsesFolder, "{}.csv".format(cubeId))
        with open(filename, "w") as f:
            f.write("pre_id,post_id,pre_celltype,post_celltype,pre_compartment,post_compartment\n")
            for synapse in synapses:
                f.write("{},{},{},{},{},{}\n".format(synapse[0], synapse[1], synapse[2], synapse[3], constants.getCompartmentId("AXON"), synapse[4]))