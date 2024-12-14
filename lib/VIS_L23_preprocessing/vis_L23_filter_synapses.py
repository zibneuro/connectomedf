import os
import sys

import vis_L23_util as util

if __name__ == "__main__":    
    dataFolder = "/srv/public/datasets/VIS-L23"
    fileIn = os.path.join(dataFolder, "synapses_flat.csv")
    fileOut = os.path.join(dataFolder, "synapses_classified_neurons.csv")

    mapping = util.loadNeuronIdMapping(os.path.join(dataFolder, "meta", "soma.csv"))

    counter = 0
    with open(fileIn) as fIn:
        lines = fIn.readlines()        
        with open(fileOut, "w") as fOut:
            fOut.write(lines[0])
            for i in range(1, len(lines)):
                parts = lines[i].rstrip().split(",")
                preIdOriginal = int(parts[3])
                postIdOriginal = int(parts[4])
                preCelltype = int(parts[5])
                postCelltype = int(parts[6])

                if(preCelltype != -1 and postCelltype != -1):
                    preId = mapping[preIdOriginal]
                    postId = mapping[postIdOriginal]
                    counter += 1
                    print(counter, preCelltype, postCelltype)
                    parts[3] = str(preId)
                    parts[4] = str(postId)
                    fOut.write("{},{},{},{},{},{},{},{}\n".format(*parts))

            

