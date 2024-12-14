from email import header
import os
import sys

import h01_util
import h01_constants

if __name__ == "__main__":

    baseFolder = h01_util.getBaseFolder()
    inputFolder = os.path.join(baseFolder, "synapses", "postprocessed")
    outputFolder = os.path.join(baseFolder, "synapses", "filtered")
    h01_util.makeDir(outputFolder)
    outputfile = os.path.join(outputFolder, "synapses-classified-neurons.csv")
    sampleFactorPost =  20
    outputfilePost = os.path.join(outputFolder, "synapses-classified-post-neurons-mod-{}.csv".format(sampleFactorPost))
    sampleFactorAll =  100
    outputfileAll = os.path.join(outputFolder, "synapses-mod-{}.csv".format(sampleFactorAll))
    outputfilePostCompartment = os.path.join(outputFolder, "synapses-non-default-post-compartment.csv")
    selectedPostCompartments = [h01_constants.getLabelId("AIS"), h01_constants.getLabelId("SOMA")]
    ids_classified = h01_util.loadIds(os.path.join(baseFolder, "meta", "ids_classified"))

    # input format
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
    headerLine = "x,y,z,pre_id,post_id,pre_celltype,post_celltype,pre_compartment,post_compartment,exc_inh_classification\n"

    filenames = h01_util.getFiles(inputFolder)

    filteredLines = []
    filteredLinesPost = []
    filteredLinesAll = []
    filteredLinesPostCompartment = []

    counter = 0    

    numSynapsesTotal = 0
    for filename in filenames:
        counter += 1

        with open(filename) as f:
            lines = f.readlines()            
            for i in range(1,len(lines)):
                numSynapsesTotal += 1
                parts = lines[i].split(",")
                pre_id = int(parts[3])
                post_id = int(parts[4])        
                post_compartment = int(parts[8])
                if(post_id in ids_classified):
                    filteredLinesPost.append(lines[i])
                    if(pre_id in ids_classified):
                        filteredLines.append(lines[i])
                        if(post_compartment in selectedPostCompartments):
                            filteredLinesPostCompartment.append(lines[i])
                if(numSynapsesTotal % sampleFactorAll == 0):
                    filteredLinesAll.append(lines[i])

        print(counter,len(filenames),len(filteredLines), len(filteredLinesPost), len(filteredLinesPostCompartment))
        print("num total", numSynapsesTotal)
    
    with open(outputfile, "w") as f:
        f.write(headerLine)
        for line in filteredLines:
            f.write(line)

    with open(outputfilePost, "w") as f:
        f.write(headerLine)
        for i in range(0,len(filteredLinesPost)):
            if(i % sampleFactorPost == 0):
                f.write(filteredLinesPost[i])

    with open(outputfilePostCompartment, "w") as f:
        f.write(headerLine)
        for line in filteredLinesPostCompartment:
            f.write(line)

    with open(outputfileAll, "w") as f:
        f.write(headerLine)
        for line in filteredLinesAll:
            f.write(line)
