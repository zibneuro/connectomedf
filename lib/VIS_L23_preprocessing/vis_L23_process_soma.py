import os
import sys
import numpy as np
import pandas as pd

import vis_L23_constants
import vis_L23_util


def loadSomaTable(filename):

    def getCelltypeId(subtype):
        if(subtype == "pyramidal"):
            return vis_L23_constants.getCelltypeId("PYR")
        elif(subtype == "bipolar"):
            return vis_L23_constants.getCelltypeId("INTER-bipolar")
        elif(subtype == "basket"):
            return vis_L23_constants.getCelltypeId("INTER-basket")
        elif(subtype == "chandelier"):
            return vis_L23_constants.getCelltypeId("INTER-chandelier")
        elif(subtype == "martinotti"):
            return vis_L23_constants.getCelltypeId("INTER-martinotti")
        elif(subtype == "neurogliaform"):
            return vis_L23_constants.getCelltypeId("INTER-neurogliaform")
        elif(subtype == "unknown_type"):
            return vis_L23_constants.getCelltypeId("INTER-unknown")
        else:
            raise ValueError(subtype)

    neuronId_props = {}
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            parts = lines[i].rstrip().split(",")
            if(parts[3] in ["e", "i"]):
                neuronId = int(parts[1])
                subtype = parts[7]                
                celltypeId = getCelltypeId(subtype)
                analyzed = bool(int(parts[8]))
                position = np.array([int(parts[9]), int(parts[10]), int(parts[11])])
                neuronId_props[neuronId] = {
                    "position" : position,
                    "celltype_id" : celltypeId,
                    "analyzed" : analyzed,
                    "ais_analyzed" : False,
                    "axon_length" : -1,
                    "dendrite_length" : -1,
                }
    return neuronId_props


def loadPyramidalSubset(filename):
    neuronId_props = {}
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            parts = lines[i].rstrip().split(",")
            neuronId = int(parts[1])
            axonLength = float(parts[4])
            dendriteLength = float(parts[5])
            neuronId_props[neuronId] = {
                "axon_length" : axonLength,
                "dendrite_length" : dendriteLength,
            }
    return neuronId_props


def writeSoma(filename, neuronId_props):    
    with open(filename, "w") as fOut:
        fOut.write("x,y,z,neuron_id,neuron_id_mapped,celltype,analyzed,ais_analyzed,axon_length,dendrite_length\n")
        neuronIds = sorted(neuronId_props.keys())
        neuronIdMapped = 1000
        for neuronId in neuronIds:          
            props = neuronId_props[neuronId]  
            position = props["position"]
            fOut.write("{:.0f},{:.0f},{:.0f},{},{},{},{},{},{:.1f},{:.1f}\n".format(*position, neuronId, neuronIdMapped, 
                props["celltype_id"], int(props["analyzed"]), int(props["ais_analyzed"]), props["axon_length"], props["dendrite_length"]))
            neuronIdMapped += 1


def loadAISSubset(filename):
    df = pd.read_hdf(filename)
    ais_synapses = df[df["synapse_correct"]]        
    postIds = set(ais_synapses["post_pt_root_id"].to_numpy(int))    
    return postIds


if __name__ == "__main__":
    dataFolder = sys.argv[1]
    metaFolder = os.path.join(dataFolder, "meta")

    vis_L23_util.makeCleanDir(metaFolder)

    fileSomaTable = os.path.join(dataFolder, "data_gitrepo_ChandelierL23", "soma_valence_v185.csv")
    filePyramidalSubset = os.path.join(dataFolder, "data_Microns_L23", "soma.csv")
    fileAISSubset = os.path.join(dataFolder, "data_gitrepo_ChandelierL23", "ais_synapse_data_all_v185.h5")
    fileOut = os.path.join(metaFolder, "soma.csv")
    
    somaTable = loadSomaTable(fileSomaTable)
    assert len(somaTable) == 450

    pyramidalSubset = loadPyramidalSubset(filePyramidalSubset)
    assert len(pyramidalSubset) == 363

    aisNeuronIds = loadAISSubset(fileAISSubset)
    assert len(aisNeuronIds) == 153

    for neuronId in pyramidalSubset.keys():
        assert somaTable[neuronId]["analyzed"]
        assert somaTable[neuronId]["celltype_id"] == vis_L23_constants.getCelltypeId("PYR")

    for neuronId in aisNeuronIds:
        somaTable[neuronId]["ais_analyzed"] = True


    for neuronId, props in somaTable.items():
        if(neuronId in pyramidalSubset):
            props["axon_length"] = pyramidalSubset[neuronId]["axon_length"]
            props["dendrite_length"] = pyramidalSubset[neuronId]["dendrite_length"]

    writeSoma(fileOut, somaTable)