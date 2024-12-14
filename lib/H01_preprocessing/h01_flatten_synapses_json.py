import os
import sys
import glob
import json
import multiprocessing as mp
import numpy as np

from . import h01_util
from . import h01_constants


def flattenBatch(filenames, filenameIds, outfolder):    
    for i in range(0, len(filenameIds)):
        filename = filenames[filenameIds[i]]
        print(filename)

        outfileName = os.path.join(outfolder, os.path.basename(filename).replace(".json", ".csv"))

        conversionFactors = h01_constants.getLengthConversionFactorsNanometer()
        def getBoundingBoxVolume(bbSize):
            volume = conversionFactors[0] * int(bbSize["x"]) * conversionFactors[1] * int(bbSize["y"]) * conversionFactors[2] * int(bbSize["z"])
            return volume
        def getPositionNanometer(location):
            position = [conversionFactors[0] * int(location["x"]), conversionFactors[1] * int(location["y"]), conversionFactors[2] * int(location["z"])]
            return position

        with open(outfileName, "w") as fOut:

            fOut.write("pre_id,pre_neuronId,pre_baseNeuronId,pre_label,post_id,post_neuronId,post_baseNeuronId,post_label,synapse_type,synapse_x,synapse_y,synapse_z,confidence,pre_size,post_size\n")

            with open(filename) as f:
                line = f.readline().rstrip()            

                counter = 1
                while(line):
                    counter += 1

                    synapseProps = json.loads(line)

                    excInh = int(synapseProps["type"])
                    prePartner = synapseProps["pre_synaptic_site"]
                    postPartner = synapseProps["post_synaptic_partner"]
                    location = synapseProps["location"]
                    
                    position = getPositionNanometer(location)
                    confidence = float(synapseProps["type"])

                    pre_id = int(prePartner["id"])
                    if("neuron_id" in prePartner):
                        pre_neuronId = int(prePartner["neuron_id"])                 
                    else:
                        pre_neuronId = -1
                    
                    if("base_neuron_id") in prePartner:
                        pre_baseNeuronId = int(prePartner["base_neuron_id"]) 
                    else:
                        pre_baseNeuronId = -1

                    pre_label = prePartner["class_label"]      

                    pre_volume = getBoundingBoxVolume(prePartner["bounding_box"]["size"])

                    post_id = int(postPartner["id"]) 
                    if("neuron_id" in postPartner):
                        post_neuronId = int(postPartner["neuron_id"])                
                    else:
                        post_neuronId = -1
                    if("base_neuron_id" in postPartner):
                        post_baseNeuronId = int(postPartner["base_neuron_id"]) 
                    else:
                        post_baseNeuronId = -1
                    post_label = postPartner["class_label"]   

                    post_volume = getBoundingBoxVolume(postPartner["bounding_box"]["size"])                 
                    
                    fOut.write("{},{},{},{},".format(pre_id, pre_neuronId, pre_baseNeuronId, pre_label))
                    fOut.write("{},{},{},{},".format(post_id, post_neuronId, post_baseNeuronId, post_label))
                    fOut.write("{},{},{},{},{:.3f},{},{}\n".format(excInh, position[0], position[1], position[2], confidence, pre_volume, post_volume))

                    line = f.readline()


def flattenJson():
    files = glob.glob(os.path.join(h01_util.getBaseFolder(), "synapses", "json","*.json"))
    outfolder = os.path.join(h01_util.getBaseFolder(), "synapses", "flattened_ext")
    h01_util.makeCleanDir(outfolder)

    filenameIds = np.arange(len(files))
    batches = np.array_split(filenameIds, h01_util.getnum_workers())

    processes = []
    for batch in batches:
        p = mp.Process(target=flattenBatch, args=(files, batch, outfolder))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    flattenJson()
    
    
    

        
        



