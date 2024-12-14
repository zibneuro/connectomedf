import os
import glob
import numpy as np
import shutil
import warnings

from . import h01_constants


def getPath(descriptor):
    baseFolder = "/srv/public/datasets/H01"

    if(descriptor == "baseFolder"):
        return baseFolder    
    elif(descriptor == "synapsesFolder"):
        return baseFolder + "/synapses"    
    elif(descriptor == "metaFolder"):
        return baseFolder + "/meta"
    elif(descriptor == "networkFolder"):
        return baseFolder + "/networkCompatibility" 
    else:
        raise ValueError(descriptor)


def getBaseFolder():
    return "/srv/public/datasets/H01"


def getnum_workers():
    num_workers = 10
    return num_workers


def makeCleanDir(dirname):
    if(os.path.exists(dirname)):
        shutil.rmtree(dirname, ignore_errors=False, onerror=None)
    os.mkdir(dirname)


def makeDir(dirname):
    if(not os.path.exists(dirname)):
        os.mkdir(dirname)


def writeSynapses(filename, synapses):
    np.savetxt(filename, synapses, delimiter=",", fmt=["%d", "%d", "%d", "%.3f", "%.3f", "%.3f"])


def getExcInhCount(synapses):
    excInhValues = h01_constants.getExcInhValues()
    numExc = np.count_nonzero(synapses[:, 2] == excInhValues[0])
    numInh = np.count_nonzero(synapses[:, 2] == excInhValues[1])
    return numExc, numInh


def getFiles(folderName, postfix="*.csv"):
    return glob.glob(os.path.join(folderName, postfix))


def writeFeatures(filename, contactSites):
    siteKeys = list(contactSites.keys())
    siteKeys.sort()

    with open(filename, "w") as f:
        for siteKey in siteKeys:
            count = contactSites[siteKey]
            f.write("{},{},{}\n".format(siteKey[0], siteKey[1], count))


def loadFeatures(filename):
    return np.loadtxt(filename, delimiter=",").astype(int).reshape((-1, 3))


def getFeatureFilesPerCube(featuresFolder):
    featureFilesPerCube = {}
    prePostFiles = getFiles(featuresFolder)
    for prePostFile in prePostFiles:
        basename = os.path.basename(prePostFile)
        cubeId = int(basename.split("_")[0])
        if(cubeId not in featureFilesPerCube):
            featureFilesPerCube[cubeId] = {}
        if("pre" in basename):
            featureFilesPerCube[cubeId]["pre"] = prePostFile
        else:
            featureFilesPerCube[cubeId]["post"] = prePostFile

    return featureFilesPerCube


def splitFeaturesExcInh(features):
    excIdx = features[:, 1] == h01_constants.getExcInhValues()[0]
    inhIdx = features[:, 1] == h01_constants.getExcInhValues()[1]
    return features[excIdx, :][:, (0, 2)], features[inhIdx, :][:, (0, 2)]


def getMergedFeatures(features, collapseRows=False):
    if(collapseRows):
        nid_contactSites = {}
        for i in range(0, features.shape[0]):
            nid = features[i, 0]
            numSites = features[i, 2]
            if(nid in nid_contactSites):
                nid_contactSites[nid] += numSites
            else:
                nid_contactSites[nid] = numSites
        n = len(nid_contactSites)
        nids = list(nid_contactSites.keys())
        nids.sort()
        featuresMerged = np.zeros((n, 2), dtype=int)
        for i in range(0, n):
            nid = nids[i]
            featuresMerged[i, 0] = nid
            featuresMerged[i, 1] = nid_contactSites[nid]
        return featuresMerged
    else:
        return features[:, (0, 2)]


def writeConnectedPairs(filename, connectedPairs):
    cellPairs = list(connectedPairs.keys())
    cellPairs.sort()

    with open(filename, "w") as f:
        for cellPair, numSynapses in connectedPairs.items():
            f.write("{},{},{}\n".format(cellPair[0], cellPair[1], numSynapses))


def writeDSC(filename, dscPerPair):
    cellPairs = list(dscPerPair.keys())
    cellPairs.sort()

    with open(filename, "w") as f:
        for cellPair, dsc in dscPerPair.items():
            f.write("{},{},{:.12E}\n".format(cellPair[0], cellPair[1], dsc))


def loadNeuronIds(filename, properties=False):
    idsWithProperties = np.loadtxt(filename, delimiter=",")
    idsAsSet = set(idsWithProperties[:, 0].astype(int))
    if(not properties):
        return idsAsSet
    else:
        propertiesDict = {}
        for i in range(0, len(idsAsSet)):
            neuronId = idsWithProperties[i][0]
            prop1 = idsWithProperties[i][1]
            propertiesDict[neuronId] = (prop1)
        return idsAsSet, propertiesDict


def splitIdsByLayer(ids, properties):
    idsByProperty = {}
    for neuronId in ids:
        propertyValue = properties[neuronId]
        if(propertyValue not in idsByProperty):
            idsByProperty[propertyValue] = set()
        idsByProperty[propertyValue].add(neuronId)
    return idsByProperty


def getCellularConnections(boundsDescriptor, gridDescriptor, ruleName, neuronIds):
    realizationDir = os.path.join(getPath("realizationsFolder"), "{}_{}_{}".format(boundsDescriptor, gridDescriptor, ruleName))

    if(not os.path.exists(realizationDir)):
        raise RuntimeError

    print("cellular connections from:", realizationDir)

    pairwiseConnections = set()
    cubeFiles = getFiles(realizationDir)
    for cubeFile in cubeFiles:

        synapseCounts = np.loadtxt(cubeFile, delimiter=",", usecols=(0, 1)).reshape((-1, 2)).astype(int)
        for i in range(0, synapseCounts.shape[0]):
            preId = synapseCounts[i][0]
            postId = synapseCounts[i][1]
            if(preId in neuronIds and postId in neuronIds):
                cellPair = (preId, postId)
                pairwiseConnections.add(cellPair)

    print("total connections", len(pairwiseConnections))

    return pairwiseConnections


def getSynapseCountsForSelectedNeurons(cubeFile, neuronIds):
    synapsesPerPair = {}

    realization = np.loadtxt(cubeFile, delimiter=",").reshape((-1, 3)).astype(int)
    for i in range(0, realization.shape[0]):
        preId = realization[i][0]
        postId = realization[i][1]
        count = realization[i][2]
        if(preId in neuronIds and postId in neuronIds):
            cellPair = (preId, postId)
            synapsesPerPair[cellPair] = count

    return synapsesPerPair


def getConnectedPostPerPre(connectedPairs):
    postPerPre = {}
    for pair in connectedPairs:
        preId = pair[0]
        postId = pair[1]

        if(preId not in postPerPre):
            postPerPre[preId] = set()
        postPerPre[preId].add(postId)

    return postPerPre


def getMaskForIds(ids, selectedIds):
    n = ids.shape[0]
    mask = np.zeros(n).astype(bool)
    for i in range(0, n):
        mask[i]= ids[i] in selectedIds
    return mask, ~mask


def getMaskForNeuronIds(prePostIds, enabledIds, returnInvertedMask=False):
    n = prePostIds.shape[0]
    mask = np.zeros(n).astype(bool)
    for i in range(0, n):
        preId = prePostIds[i, 0]
        postId = prePostIds[i, 1]
        mask[i] = preId in enabledIds and postId in enabledIds
    if(returnInvertedMask):
        maskInverted = ~mask
        return mask, maskInverted
    else:
        return mask


def getNumPairs(prePostIds):
    preIds = set(prePostIds[:, 0])
    postIds = set(prePostIds[:, 1])
    return len(preIds) * len(postIds) - len(preIds)


def loadIds(filename):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return set(np.loadtxt(filename).astype(int))


def writeIds(filename, ids):
    idsList = list(ids)
    idsList.sort()
    np.savetxt(filename, idsList, fmt="%d")


def getIdsGroupedByCelltypeId(metaFolder):
    celltype_ids = {}
    for layer in h01_constants.getLayers():
        for celltype in h01_constants.getCelltypes():
            ids = loadIds(os.path.join(metaFolder, "ids_{}_{}".format(layer, celltype)))
            celltypeId = h01_constants.getCelltypeId(layer, celltype)
            celltype_ids[celltypeId] = ids
    return celltype_ids


def loadConnections(filename):
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
    synapses = np.loadtxt(filename, skiprows=1, delimiter=",").astype(int)
    connections = {} # (pre_id, pre_celltype) -> (postId, post_celltype) -> post_compartment -> synapse_count
    for i in range(0, synapses.shape[0]):
        synapse = synapses[i, :]
        pre_key = (synapse[3], synapse[5]) # (pre_id, pre_celltype)
        post_key = (synapse[4], synapse[6]) # (post_id, post_celltype)
        post_compartment = synapse[8]
        if(pre_key not in connections):
            connections[pre_key] = {}
        if(post_key not in connections[pre_key]):
            connections[pre_key][post_key] = {}
        if(post_compartment not in connections[pre_key][post_key]):
            connections[pre_key][post_key][post_compartment] = 0
        connections[pre_key][post_key][post_compartment] += 1
    return connections
    

def getConnectionsFromPre(connections, preNeuronId):
    for pre_key, props in connections.items():
        preId = pre_key[0]
        if(preId == preNeuronId):
            return pre_key, props
    raise ValueError(preNeuronId)
