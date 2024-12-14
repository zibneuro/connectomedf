import os
import sys
import numpy as np
import glob

from . import h01_constants
from . import h01_util
import multiprocessing as mp


def convertPositionsToMicron(positions):
    lowerBounds = h01_constants.getLowerBounds()
    conversionFactors = h01_constants.getLengthConversionFactors()

    positions = positions - lowerBounds
    positions = np.multiply(positions, conversionFactors)
    return positions


def postprocessSynapsesExportedBatch(outfolder, metaFolder, files, fileIndices):
    for fileIdx in fileIndices:
        filename = files[fileIdx]
        print(filename)
        outfilename = os.path.join(outfolder, os.path.basename(filename))

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

        with open(filename) as f:
            lines = f.readlines()

            conversionFactors = h01_constants.getLengthConversionFactorsNanometer()
            
            with open(outfilename, "w") as f:
                f.write("x,y,z,pre_id,post_id,pre_celltype,post_celltype,pre_compartment,post_compartment,exc_inh_classification,pre_size,post_size,radial_dist\n")
                for i in range(1, len(lines)):
                    parts = lines[i].rstrip().split(",")
                    preId = int(parts[1])
                    preLabel = parts[3]
                    postId = int(parts[5])
                    postLabel = parts[7]
                    synapseType = parts[8]
                    #x = conversionFactors[0] * float(parts[9])
                    #y = conversionFactors[1] * float(parts[10])
                    #z = conversionFactors[2] * float(parts[11])
                    x = int(parts[9])
                    y = int(parts[10])
                    z = int(parts[11])
                    position = np.array([x,y,z])
                    radialDist = h01_constants.getRadialDistanceFromCenterPoint(position)
                    preSize = int(parts[13])
                    postSize = int(parts[14])
                    
                    if(preLabel in ["AXON"] and postLabel in ["DENDRITE", "AIS", "SOMA"]):

                        preCelltype = getCelltypeId(preId)
                        postCelltype = getCelltypeId(postId)                      

                        preLabelId = h01_constants.getLabelId(preLabel)
                        postLabelId = h01_constants.getLabelId(postLabel)

                        f.write("{:.0f},{:.0f},{:.0f},{},{},{},{},{},{},{},{},{},{}\n".format(x, y, z, preId, postId, preCelltype, postCelltype, preLabelId, postLabelId, synapseType, preSize, postSize, radialDist))


def postprocessSynapsesExported():    
    files = glob.glob(os.path.join(h01_util.getBaseFolder(), "synapses", "flattened_ext", "*.csv"))

    metaFolder = os.path.join(h01_util.getBaseFolder(), "meta")
    outfolder = os.path.join(h01_util.getBaseFolder(), "synapses", "postprocessed_ext")
    h01_util.makeCleanDir(outfolder)
    
    filenameIds = np.arange(len(files))
    batches = np.array_split(filenameIds, h01_util.getnum_workers())

    processes = []
    for batch in batches:
        p = mp.Process(target=postprocessSynapsesExportedBatch, args=(outfolder, metaFolder, files, batch))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    

if __name__ == "__main__":
    postprocessSynapsesExported()
