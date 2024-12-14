import os
import numpy as np
import multiprocessing as mp

import h01_util


if __name__ == "__main__":
    gridDescriptor = "100000-100000-87500"
    metaFolder = os.path.join(h01_util.getBaseFolder(), "meta") 
    #cubeIndexFolder = os.path.join(metaFolder, "cube_index")
    cubeIndexFolder = os.path.join(metaFolder, "cube_index_{}".format(gridDescriptor))
    h01_util.makeCleanDir(cubeIndexFolder)

    neuronIds = h01_util.loadIds(os.path.join(metaFolder, "ids_classified"))
    cubeFiles = h01_util.getFiles(os.path.join(h01_util.getBaseFolder(), "synapses", "gridded_complete-volume_{}".format(gridDescriptor)))

    batches = np.array_split(cubeFiles, h01_util.getnum_workers())

    manager = mp.Manager()
    results = manager.dict() # batchIdx -> neuronId -> [(cubeId, pre/post/both), ...]

    def createCubeIndexBatch(results, batchIdx, filenames, neuronIds):
        localResults = {}
        for neuronId in neuronIds:
            localResults[neuronId] = []

        for fileIdx in range(0, len(filenames)):
            print(fileIdx, len(filenames))

            filename = filenames[fileIdx]
            cubeId = int(os.path.basename(filename).split(".")[0])
            synapses = np.atleast_2d(np.loadtxt(filename, skiprows=1, delimiter=",").astype(int))
            preIds = set(synapses[:,0]) 
            postIds = set(synapses[:,1])

            presynaptic = preIds & neuronIds
            postsynaptic = postIds & neuronIds
            both = presynaptic & postsynaptic
            presynapticOnly = presynaptic - both
            postsynapticOnly = postsynaptic - both
            
            # 0 only presynaptic sites
            # 1 only postsynaptic sites
            # 2 pre- and postsynaptic sites
            for neuronId in presynapticOnly:
                localResults[neuronId].append((cubeId, 0))
            for neuronId in postsynapticOnly:
                localResults[neuronId].append((cubeId, 1))
            for neuronId in both:
                localResults[neuronId].append((cubeId, 2))

        results[batchIdx] = localResults

    processes = []
    for i in range(0, len(batches)):
        p = mp.Process(target=createCubeIndexBatch, args=(results, i, batches[i], neuronIds))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    resultsMerged = {}
    for neuronId in neuronIds:
        resultsMerged[neuronId] = []
    for resultsLocal in results.values():
        for neuronId, cubes in resultsLocal.items():
            resultsMerged[neuronId].extend(cubes)

    def writeNeuron(filename, cubes):
        cubes.sort()
        if(cubes):
            with open(filename, "w") as f:
                f.write("cube_id,pre_post_both\n")
                for cube in cubes:
                    f.write("{},{}\n".format(*cube))

    for neuronId, cubes in resultsMerged.items():
        filename =  os.path.join(cubeIndexFolder, "{}.csv".format(neuronId))
        writeNeuron(filename, cubes)


