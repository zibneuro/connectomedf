import os
import sys
import numpy as np
import pandas as pd

import vis_L23_util as util


def filterConnections(connections, ids):
    connectionsFiltered = {}
    idsFiltered = set()
    for key, count in connections.items():
        if(key[0] in ids and key[1] in ids):
            connectionsFiltered[key] = count
            idsFiltered.add(key[0])
            idsFiltered.add(key[1])
    return connectionsFiltered, idsFiltered


def loadTuningCurves(filename):
    tuningCurves = {}
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            parts = lines[i].rstrip().split(",")
            neuronId = int(parts[0])
            curve = np.zeros(16)
            for k in range(0,16):
                curve[k]= float(parts[k+1])
            tuningCurves[neuronId] = curve    
    return tuningCurves


if __name__ == "__main__":
    dataFolder = "/srv/public/datasets/VIS-L23/"
    functionalDataFolder = os.path.join(dataFolder, "data_Microns_L23", "211019_vignette_functional_analysis_data")

    connections = util.loadConnectionsPycPyc(os.path.join(functionalDataFolder, "pyc_pyc_subgraph.csv"))
    tuningCurves = loadTuningCurves(os.path.join(dataFolder, "functional_data", "tuning_curves.csv"))
    #print(np.unique(list(connections.values()), return_counts=True))

    for postfix in ["", "_osi"]:

        idsFunctional = util.loadIds(os.path.join(dataFolder, "functional_data", "ids_functional{}".format(postfix)), sortedList=True)

        connectionsFiltered, idsFiltered = filterConnections(connections, idsFunctional)
        print(np.unique(list(connectionsFiltered.values()), return_counts=True))
        print(len(idsFiltered))        

        observations = []
        for neuronId in idsFunctional:
            observations.append(tuningCurves[neuronId])

        observationMatrix = np.array(observations)
        correlationMatrix = np.corrcoef(observationMatrix)
            
        np.savetxt(os.path.join(dataFolder, "functional_data", "correlation_tuning_curve{}".format(postfix)), correlationMatrix)

        connectivityMatrix = np.zeros_like(correlationMatrix)
        for i in range(0, len(idsFunctional)):
            for j in range(0, len(idsFunctional)):
                preId = idsFunctional[i]
                postId = idsFunctional[j]
                key = (preId, postId)
                if(key in connectionsFiltered):
                    connectivityMatrix[i,j] = connectionsFiltered[key]

        np.savetxt(os.path.join(dataFolder, "functional_data", "connectivity_matrix{}".format(postfix)), connectivityMatrix)    