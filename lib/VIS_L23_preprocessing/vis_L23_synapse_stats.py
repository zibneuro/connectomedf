import os
import sys
import pandas as pd
import numpy as np

import vis_L23_constants as constants


def loadSynapsesProcessed(filename, compartment = None):
    synapses = {}
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            parts = lines[i].rstrip().split(",")            
            synapse = (int(parts[3]), int(parts[4]))
            synapseCompartment = int(parts[7])
            if(compartment is None or synapseCompartment == compartment):
                if(synapse not in synapses):
                    synapses[synapse] = 0
                synapses[synapse] += 1                    
    return synapses


def loadSynapsesPyc(filename):
    synapses = {}
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            parts = lines[i].rstrip().split(",")
            synapse = (int(parts[4]), int(parts[5]))
            if(synapse not in synapses):
                synapses[synapse] = 0
            synapses[synapse] += 1                    
    return synapses


def loadSynapsesAIS(filename):
    synapses = {}
    ais_synapses_df = pd.read_hdf(filename, 'data')    
    ais_synapses = ais_synapses_df[ais_synapses_df["synapse_correct"]]        
    preIds = ais_synapses['pre_pt_root_id'].to_numpy(int)
    postIds = ais_synapses['post_pt_root_id'].to_numpy(int)
    for i in range(0, preIds.size):
        synapse = (preIds[i], postIds[i])
        if(synapse not in synapses):
            synapses[synapse] = 0
        synapses[synapse] += 1                        
    return synapses



if __name__ == "__main__":
    dataFolder = sys.argv[1]

    synapsesProcessed = loadSynapsesProcessed(os.path.join(dataFolder, "synapses_flat.csv"))
    synapsesProcessedAIS = loadSynapsesProcessed(os.path.join(dataFolder, "synapses_flat.csv"), compartment=constants.getCompartmentId("AIS"))
    synapsesPyc = loadSynapsesPyc(os.path.join(dataFolder, "data_Microns_L23", "211019_pyc-pyc_subgraph_v185.csv"))
    synapsesAIS = loadSynapsesAIS(os.path.join(dataFolder, "data_gitrepo_ChandelierL23", "ais_synapse_data_all_v185.h5"))
    clusterSizes, clusterOccurrences = np.unique(list(synapsesProcessed.values()), return_counts = True)    

    print("synapse clusters")
    print(clusterSizes)
    print(clusterOccurrences)

    # assert all proofedited PycPyc connections are preserved
    assert len(synapsesPyc) == 1752    
    assert np.sum(list(synapsesPyc.values())) == 1981

    for neuronPair, numConnections in synapsesPyc.items():
        assert neuronPair in synapsesProcessed        
        assert numConnections == synapsesProcessed[neuronPair]

    # assert all manually curated AIS synapses are preserved
    assert np.sum(list(synapsesAIS.values())) == 1929
    assert np.sum(list(synapsesProcessedAIS.values())) == 1929

    for neuronPair, numConnections in synapsesAIS.items():
        assert neuronPair in synapsesProcessedAIS        
        assert numConnections == synapsesProcessedAIS[neuronPair]
