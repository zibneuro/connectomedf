import os
import sys
import numpy as np
import pandas as pd

import vis_L23_util as util
import vis_L23_constants as constants


def writeToFile(filename, synapses, somas=None):

    def get_mapped_id(neuron_id, unknown_id = -1):
        if(neuron_id in somas):
            return somas[neuron_id]["neuron_id_mapped"]
        else:
            return unknown_id

    with open(filename, "w") as f:
        f.write("x,y,z,pre_id,post_id,pre_celltype,post_celltype,post_compartment\n")
        if(somas is not None):
            for synapse in synapses:
                pre_id = get_mapped_id(synapse[3])
                post_id = get_mapped_id(synapse[4])                    
                f.write(f"{synapse[0]},{synapse[1]},{synapse[2]},{pre_id},{post_id},{synapse[5]},{synapse[6]},{synapse[7]}\n")
        else:
            for synapse in synapses:                
                f.write("{},{},{},{},{},{},{},{}\n".format(*synapse))


def isSomaProximal(positionSynapse, positionSoma, distanceThreshold = 15000):
    return np.linalg.norm(positionSynapse -  positionSoma) <= distanceThreshold


def getAIS_synapseIds_falsePositives(filename):
    ais_synapses_df = pd.read_hdf(filename, 'data')    
    ais_synapses = ais_synapses_df[ais_synapses_df["synapse_correct"]]        
    synapseIds = set(ais_synapses["id"].to_numpy(int))
    ais_synapses_falsePositives = ais_synapses_df[ais_synapses_df["synapse_correct"] == 0]
    falsePositiveIds = set(ais_synapses_falsePositives["id"].to_numpy(int))
    return synapseIds, falsePositiveIds


def getPycPyc_synapseIds_neuronIds(filename):
    pyc_pyc_df = pd.read_csv(filename, delimiter=",")
    synapseIds = set(pyc_pyc_df["id"].to_numpy(int))
    neuronIds = set(pyc_pyc_df["pre_root_id"].to_numpy(int))
    neuronIds |= set(pyc_pyc_df["post_root_id"].to_numpy(int))
    return synapseIds, neuronIds
    

def getSynapsesAll(filename):
    synapsesAll_df = pd.read_csv(filename, delimiter=",")    
    return synapsesAll_df[["id", "pre_root_id", "post_root_id", "ctr_pt_x_nm", "ctr_pt_y_nm", "ctr_pt_z_nm"]]


def loadSoma(filename):
    df = pd.read_csv(filename, delimiter=",")  
    neuronId_props = {}
    for idx in df.index:
        neuronId_props[df["neuron_id"][idx]] = {
            "neuron_id_mapped" : int(int(df["neuron_id_mapped"][idx])),
            "position" : np.array([int(df["x"][idx]), int(df["y"][idx]), int(df["z"][idx])]),
            "celltype" : int(df["celltype"][idx]),
            "analyzed" : bool(df["analyzed"][idx]),
            "ais_analyzed" : bool(df["ais_analyzed"][idx]),
        }
    return neuronId_props


if __name__ == "__main__":
    dataFolder = sys.argv[1]
    outfile = os.path.join(dataFolder, "synapses.csv")
    outfileFalsePositives = os.path.join(dataFolder, "synapses_flat_false_positives.csv")
    outfolder_synapses = os.path.join(dataFolder, "synapses_per_neuron")
    
    util.makeCleanDir(outfolder_synapses)

    filenameAIS = os.path.join(dataFolder, "data_gitrepo_ChandelierL23", "ais_synapse_data_all_v185.h5")    
    synapseIds_AIS, synapseIds_AIS_falsePositives  = getAIS_synapseIds_falsePositives(filenameAIS)
    #print("ais false positives", len(synapseIds_AIS_falsePositives))

    filenamePycPyc = os.path.join(dataFolder, "data_Microns_L23", "211019_pyc-pyc_subgraph_v185.csv")    
    synapseIds_PycPyc, neuronIds_PycPyc = getPycPyc_synapseIds_neuronIds(filenamePycPyc)
    #print("analyzed pyc neurons", len(neuronIds_PycPyc))

    filenameSoma = os.path.join(dataFolder, "meta", "soma.csv")
    somas = loadSoma(filenameSoma)
    somaIds = set(somas.keys())
    #print("num soma ids", len(somaIds))    
    
    filenameSynapsesAll = os.path.join(dataFolder, "data_Microns_L23", "pni_synapses_v185.csv")    
    synapsesAll_df = getSynapsesAll(filenameSynapsesAll)     
    synapsesAll = synapsesAll_df.to_numpy(int)

    synapseIds_all = set(synapsesAll_df["id"].to_numpy(int))    
    assert len(synapseIds_AIS - synapseIds_all) == 0
    assert len(synapseIds_PycPyc - synapseIds_all) == 0
    
    synapsesPerNeuronOutgoing = {}
    synapsesPerNeuronIncoming = {}
    for somaId in somaIds:
        synapsesPerNeuronOutgoing[somaId] = set()
        synapsesPerNeuronIncoming[somaId] = set()
    
    synapsesProcessed = set()  
    synapsesPruned = set()  
    for i in range(0, synapsesAll.shape[0]):        
        synapseId = synapsesAll[i,0]
        preId = synapsesAll[i,1]
        postId = synapsesAll[i,2]
        positionSynapse = np.array(synapsesAll[i,3:])
        preCelltypeId = -1
        postCelltypeId = -1
        postCompartmentId = -1
        isFalsePositive = False

        # check for false positives based on proofedited subsets of synapses
        if(preId in neuronIds_PycPyc and postId in neuronIds_PycPyc):            
            if(synapseId not in synapseIds_PycPyc):
                isFalsePositive = True
        if(synapseId in synapseIds_AIS_falsePositives):
            isFalsePositive = True
    
        if(preId in somaIds):
            preCelltypeId = somas[preId]["celltype"]
        if(postId in somaIds):
            postCelltypeId = somas[postId]["celltype"]
        if(synapseId in synapseIds_AIS):
            assert somas[postId]["ais_analyzed"]
            postCompartmentId = constants.getCompartmentId("AIS")
        if(postCompartmentId == -1 and postCelltypeId != -1):
            if(isSomaProximal(positionSynapse, somas[postId]["position"])):
                postCompartmentId = constants.getCompartmentId("SOMA")
            else:
                postCompartmentId = constants.getCompartmentId("DENDRITE")
            
        synapseProcessed = (positionSynapse[0], positionSynapse[1], positionSynapse[2], preId, postId, preCelltypeId, postCelltypeId, postCompartmentId)

        if(isFalsePositive):
            synapsesPruned.add(synapseProcessed)
        else:
            synapsesProcessed.add(synapseProcessed)
            #if(preCelltypeId != -1):
            #    synapsesPerNeuronOutgoing[preId].add(synapseProcessed)
            #if(postCelltypeId != -1):
            #    synapsesPerNeuronIncoming[postId].add(synapseProcessed)

    writeToFile(outfile, synapsesProcessed, somas)
    writeToFile(outfileFalsePositives, synapsesPruned)
    
    """
    for somaId, synapses in synapsesPerNeuronOutgoing.items():
        writeToFile(os.path.join(outfolder_synapses, "{}_outgoing.csv".format(somaId)), synapses)
    for somaId, synapses in synapsesPerNeuronIncoming.items():
        writeToFile(os.path.join(outfolder_synapses, "{}_incoming.csv".format(somaId)), synapses)
    """
