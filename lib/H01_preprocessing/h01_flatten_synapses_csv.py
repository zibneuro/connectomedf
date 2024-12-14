import os
import sys
import glob
import json
import multiprocessing as mp
import numpy as np

from . import h01_util
from . import h01_constants


def flattenJson():
    files = glob.glob(os.path.join(h01_util.getBaseFolder(), "synapses", "json","*.json"))
    outfolder = os.path.join(h01_util.getBaseFolder(), "synapses", "flattened_single_file")
    metaFolder = os.path.join(h01_util.getBaseFolder(), "meta")
    outFilename = os.path.join(outfolder, "synapses.csv")
    h01_util.makeCleanDir(outfolder)

    conversionFactors = h01_constants.getLengthConversionFactorsNanometer()
    def getBoundingBoxVolume(bbSize):
        volume = conversionFactors[0] * int(bbSize["x"]) * conversionFactors[1] * int(bbSize["y"]) * conversionFactors[2] * int(bbSize["z"])
        return volume
    def getPositionNanometer(location):
        position = [conversionFactors[0] * int(location["x"]), conversionFactors[1] * int(location["y"]), conversionFactors[2] * int(location["z"])]
        return position

    ids_classified = h01_util.loadIds(os.path.join(metaFolder, "ids_classified"))
    celltype_ids = h01_util.getIdsGroupedByCelltypeId(metaFolder)
    def getCelltypeId(neuronId):            
        if(neuronId not in ids_classified):
            return -1
        else:            
            for celltypeId, ids in celltype_ids.items():
                if(neuronId in ids):
                    return celltypeId
        raise RuntimeError

    with open(outFilename, "w") as fOut:
        fOut.write("x,y,z,pre_id,post_id,pre_celltype,post_celltype,post_compartment\n")

        for filename in files:
            with open(filename) as f:
                line = f.readline().rstrip()            

                counter = 1
                while(line):
                    counter += 1

                    synapseProps = json.loads(line)

                    location = synapseProps["location"]                    
                    prePartner = synapseProps["pre_synaptic_site"]
                    postPartner = synapseProps["post_synaptic_partner"]
                    
                    position = getPositionNanometer(location)                    
                    pre_label = prePartner["class_label"]                          
                    post_id = int(postPartner["id"])                     
                    post_label = postPartner["class_label"]   

                    if(pre_label in ["AXON"] and post_label in ["DENDRITE", "AIS", "SOMA"]):
                        pre_id = int(prePartner["neuron_id"])
                        post_id = int(postPartner["neuron_id"])
                        pre_celltype = getCelltypeId(pre_id)
                        post_celltype = getCelltypeId(post_id)
                        post_compartment = h01_constants.getLabelId(post_label)
                        
                        fOut.write("{},{},{},{},{},{},{},{}\n".format(*position, pre_id, post_id, pre_celltype, post_celltype, post_compartment))
                   
                    line = f.readline()


if __name__ == "__main__":
    flattenJson()