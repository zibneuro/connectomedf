import numpy as np


def getMotifIndex_16():
    return {
        (0,0,0,0,0,0) : 16, # no edges        
        (1,0,0,0,0,0) : 15, # A -> B
        (0,1,0,0,0,0) : 15, # B -> A
        (0,0,1,0,0,0) : 15, # A -> C
        (0,0,0,1,0,0) : 15, # C -> A
        (0,0,0,0,1,0) : 15, # B -> C
        (0,0,0,0,0,1) : 15, # C -> B
        (1,1,0,0,0,0) : 14, # A <-> B
        (0,0,1,1,0,0) : 14, # A <-> C
        (0,0,0,0,1,1) : 14, # B <-> C
        (1,0,0,0,1,0) : 13, # A -> B -> C
        (0,0,1,0,0,1) : 13, # A -> C -> B
        (0,1,1,0,0,0) : 13, # B -> A -> C
        (0,0,0,1,1,0) : 13, # B -> C -> A
        (1,0,0,1,0,0) : 13, # C -> A -> B 
        (0,1,0,0,0,1) : 13, # C -> B -> A 
        (1,0,1,0,0,0) : 12, # A -> B,C
        (0,1,0,0,1,0) : 12, # B -> A,C
        (0,0,0,1,0,1) : 12, # C -> A,B
        (0,0,1,0,1,0) : 11, # A, B -> C
        (1,0,0,0,0,1) : 11, # A, C -> B
        (0,1,0,1,0,0) : 11, # B, C -> A
        (1,1,0,1,0,0) : 10, # A <-> B; C -> A
        (1,1,0,0,0,1) : 10, # A <-> B; C -> B
        (0,1,1,1,0,0) : 10, # A <-> C; B -> A
        (0,0,1,1,1,0) : 10, # A <-> C; B -> C
        (1,0,0,0,1,1) : 10, # B <-> C; A -> B
        (0,0,1,0,1,1) : 10, # B <-> C; A -> C
        (1,1,1,0,0,0) : 9,  # A <-> B; A -> C
        (1,1,0,0,1,0) : 9,  # A <-> B; B -> C
        (1,0,1,1,0,0) : 9,  # A <-> C; A -> B
        (0,0,1,1,0,1) : 9,  # A <-> C; C -> B
        (0,1,0,0,1,1) : 9,  # B <-> C; B -> A
        (0,0,0,1,1,1) : 9,  # B <-> C; C -> A
        (1,1,1,1,0,0) : 8,  # A <-> B; A <-> C
        (1,1,0,0,1,1) : 8,  # A <-> B; B <-> C
        (0,0,1,1,1,1) : 8,  # A <-> C; B <-> C
        (1,0,0,1,1,0) : 7,  # A -> B -> C -> A
        (0,1,1,0,0,1) : 7,  # A -> C -> B -> A
        (1,0,1,0,1,0) : 6,  # A -> B,C; B -> C
        (1,0,1,0,0,1) : 6,  # A -> B,C; C -> B
        (0,1,1,0,1,0) : 6,  # B -> A,C; A -> C
        (0,1,0,1,1,0) : 6,  # B -> A,C; C -> A
        (1,0,0,1,0,1) : 6,  # C -> A,B; A -> B
        (0,1,0,1,0,1) : 6,  # C -> A,B; B -> A
        (1,1,1,0,0,1) : 5,  # A <-> B; A -> C -> B
        (1,1,0,1,1,0) : 5,  # A <-> B; B -> C -> A
        (1,0,1,1,1,0) : 5,  # A <-> C; A -> B -> C
        (0,1,1,1,0,1) : 5,  # A <-> C; C -> B -> A
        (0,1,1,0,1,1) : 5,  # B <-> C; B -> A -> C
        (1,0,0,1,1,1) : 5,  # B <-> C; C -> A -> B
        (1,1,0,1,0,1) : 4,  # A <-> B; C -> A, B
        (0,1,1,1,1,0) : 4,  # A <-> C; B -> A, C
        (1,0,1,0,1,1) : 4,  # B <-> C; A -> B, C
        (1,1,1,0,1,0) : 3,  # A <-> B; A -> C; B -> C
        (1,0,1,1,0,1) : 3,  # A <-> C; A -> B; C -> B
        (0,1,0,1,1,1) : 3,  # B <-> C; B -> A; C -> A
        (1,1,1,1,1,0) : 2,  # A <-> B; A <-> C; B -> C
        (1,1,1,1,0,1) : 2,  # A <-> B; A <-> C; C -> B
        (1,1,1,0,1,1) : 2,  # A <-> B; B <-> C; A -> C
        (1,1,0,1,1,1) : 2,  # A <-> B; B <-> C; C -> A
        (1,0,1,1,1,1) : 2,  # A <-> C; B <-> C; A -> B
        (0,1,1,1,1,1) : 2,  # A <-> C; B <-> C; B -> A
        (1,1,1,1,1,1) : 1   # A <-> B; A <-> C; B <-> C
    }


def assertSumsToOne(probabilities, tolerance = 0.01):
    summed = np.sum(list(probabilities.values()))
    if(abs(summed - 1) > tolerance):
        raise RuntimeError("summed probabilities: {:.12f}".format(summed))


def aggregateProbabilties_16(probabilities_64):
    probabilities_16 = {}
    
    for k in range(1, 17):
        probabilities_16[k] = 0

    motifIndex16 = getMotifIndex_16()
    
    for maskKey, probability in probabilities_64.items():
        motifNumber = motifIndex16[maskKey]
        probabilities_16[motifNumber] += probability

    assertSumsToOne(probabilities_16)

    return probabilities_16



def calcProbabilitiesSinglePreNeuron(postIds, dscPerPreNeuron):
    dscMatched = np.zeros(postIds.size)

    overlappingPostIds = dscPerPreNeuron["postIds"]
    dscValues = dscPerPreNeuron["dscValues"]

    common, idxOverlapping, idxMatched = np.intersect1d(overlappingPostIds, postIds, assume_unique=True, return_indices=True)
    if(common.size):
        dscMatched[idxMatched] = dscValues[idxOverlapping]

    #p = 1-np.exp(-dscMatched)
    p = dscMatched
    return p


def calcProbabilties(preIds, postIds, dscPerPre):
    probs = np.zeros(shape=(preIds.size, postIds.size))
    for i in range(0, preIds.size):        
        preId = preIds[i]
        dscPerPreNeuron = dscPerPre[preId]
        probs[i,:] = calcProbabilitiesSinglePreNeuron(postIds, dscPerPreNeuron)
    return probs


def getMotifMasks(numNodes):
    nEdges = numNodes * (numNodes-1)
    if(nEdges > 8):
        raise ValueError(nEdges)
    edgeConfigurations = np.arange(2**nEdges, dtype=np.uint8).reshape((-1,1))
    masks = np.unpackbits(edgeConfigurations, axis=1)
    masks_inv = np.ones_like(masks) - masks
    masks = masks.astype(bool)
    masks_inv = masks_inv.astype(bool)
    return masks[:,(8-nEdges):], masks_inv[:,(8-nEdges):]


def calcMotifProbability(p, p_inv, mask, mask_inv, average=True):
    p_edges = np.concatenate((p[:,mask], p_inv[:,mask_inv]), axis=1)
    p_motif = np.prod(p_edges, axis=1)
    if(average):
        return np.mean(p_motif)
    else:
        return p_motif


def calcModelProbabilitiesIndependentSamples(stats, nids_A_sample, nids_B_sample, nids_C_sample, dsc_A, dsc_B, dsc_C):
    num_A = nids_A_sample.size
    num_B = nids_B_sample.size
    num_C = nids_C_sample.size

    numSamples = num_A * num_B * num_C
    p_model = np.zeros(shape=(numSamples, 6))

    p_A_B = calcProbabilties(nids_A_sample, nids_B_sample, dsc_A)
    p_B_A = calcProbabilties(nids_B_sample, nids_A_sample, dsc_B)
    p_A_C = calcProbabilties(nids_A_sample, nids_C_sample, dsc_A)
    p_C_A = calcProbabilties(nids_C_sample, nids_A_sample, dsc_C)
    p_B_C = calcProbabilties(nids_B_sample, nids_C_sample, dsc_B)
    p_C_B = calcProbabilties(nids_C_sample, nids_B_sample, dsc_C)

    p_all = np.concatenate((p_A_B, p_B_A, p_A_C, p_C_A, p_B_C, p_C_B), axis=None)

    stats["avg_A-B"] = np.mean(p_A_B)
    stats["sd_A-B"] = np.std(p_A_B)
    stats["avg_B-A"] = np.mean(p_B_A)
    stats["sd_B-A"] = np.std(p_B_A)
    stats["avg_A-C"] = np.mean(p_A_C)
    stats["sd_A-C"] = np.std(p_A_C)            
    stats["avg_C-A"] = np.mean(p_C_A)
    stats["sd_C-A"] = np.std(p_C_A)            
    stats["avg_B-C"] = np.mean(p_B_C)
    stats["sd_B-C"] = np.std(p_B_C)
    stats["avg_C-B"] = np.mean(p_C_B)
    stats["sd_C-B"] = np.std(p_C_B)  
    stats["avg_all"] = np.mean(p_all)
    stats["sd_all"] = np.std(p_all)

    invalidSamples = 0

    idx = 0
    for i in range(0, num_A):
        for j in range(0, num_B):
            for k in range(0, num_C):
                if(i == j or i == k or j == k):
                    p_model[idx, :] = 0
                    invalidSamples += 1
                else:
                    p_model[idx, 0] = p_A_B[i,j]
                    p_model[idx, 1] = p_B_A[j,i]
                    p_model[idx, 2] = p_A_C[i,k]
                    p_model[idx, 3] = p_C_A[k,i]
                    p_model[idx, 4] = p_B_C[j,k]
                    p_model[idx, 5] = p_C_B[k,j]
                idx += 1

    #print("invalid samples", invalidSamples, "of", numSamples)

    return p_model


def calcProbilityAll(nids_A, nids_B, nids_C, p_model):    
    nids_ABC = np.concatenate((nids_A.reshape(-1,1), nids_B.reshape(-1,1), nids_C.reshape(-1,1)), axis=1)

    _, indices_AB = np.unique(nids_ABC[:, (0,1)], return_index = True, axis=1)
    _, indices_AC = np.unique(nids_ABC[:, (0,2)], return_index = True, axis=1)
    _, indices_BC = np.unique(nids_ABC[:, (1,2)], return_index = True, axis=1)
    
    p_A_B_unique = p_model[indices_AB, 0]
    p_B_A_unique = p_model[indices_AB, 1]
    p_A_C_unique = p_model[indices_AC, 2]
    p_C_A_unique = p_model[indices_AC, 3]
    p_B_C_unique = p_model[indices_BC, 4]
    p_C_B_unique = p_model[indices_BC, 5]

    p_ABC_unique = np.concatenate((p_A_B_unique, p_B_A_unique, p_A_C_unique, p_C_A_unique, p_B_C_unique, p_C_B_unique), axis=None)
    
    return p_ABC_unique


def calcModelProbabilitiesDependentSamples(stats, nids_A, nids_B, nids_C, dsc_ABC):    
    num_A = nids_A.size
    num_B = nids_B.size
    num_C = nids_C.size

    if(num_A != num_B or num_B != num_C):
        raise RuntimeError("dependent sample size mismatch")
    numSamples = num_A

    p_model = np.zeros(shape=(numSamples, 6))

    nids_merged_unique, rev_indices = getMergedUnique(nids_A, nids_B, nids_C)
    p_ABC = calcProbabilties(nids_merged_unique, nids_merged_unique, dsc_ABC)

    for i in range(0, numSamples):
        ia = rev_indices[i]
        ib = rev_indices[i + numSamples]
        ic = rev_indices[i + 2 * numSamples]

        p_model[i, 0] = p_ABC[ia, ib]
        p_model[i, 1] = p_ABC[ib, ia]
        p_model[i, 2] = p_ABC[ia, ic]
        p_model[i, 3] = p_ABC[ic, ia]
        p_model[i, 4] = p_ABC[ib, ic]
        p_model[i, 5] = p_ABC[ic, ib]
    
    p_all = calcProbilityAll(nids_A, nids_B, nids_C, p_model)

    stats["avg_A-B"] = np.mean(p_model[:, 0])
    stats["sd_A-B"] = np.std(p_model[:, 0])
    stats["avg_B-A"] = np.mean(p_model[:, 1])
    stats["sd_B-A"] = np.std(p_model[:, 1])
    stats["avg_A-C"] = np.mean(p_model[:, 2])
    stats["sd_A-C"] = np.std(p_model[:, 2])            
    stats["avg_C-A"] = np.mean(p_model[:, 3])
    stats["sd_C-A"] = np.std(p_model[:, 3])            
    stats["avg_B-C"] = np.mean(p_model[:, 4])
    stats["sd_B-C"] = np.std(p_model[:, 4])
    stats["avg_C-B"] = np.mean(p_model[:, 5])
    stats["sd_C-B"] = np.std(p_model[:, 5])  
    stats["avg_all"] = np.mean(p_all)
    stats["sd_all"] = np.std(p_all)    
        
    return p_model

def getProbabilitiesAsArray(probabilities_16):
    maxK = 16
    probabilities_array = np.zeros(maxK)
    for k in range(0, maxK):
        probabilities_array[k] = probabilities_16[k+1]
    return probabilities_array


def calcMotifProbabilities(stats, nids_A_sample, nids_B_sample, nids_C_sample, dsc_A, dsc_B, dsc_C, independentSamples):
    
    if(independentSamples):
        p_model = calcModelProbabilitiesIndependentSamples(stats, nids_A_sample, nids_B_sample, nids_C_sample, dsc_A, dsc_B, dsc_C)
    else:
        dsc_ABC = dsc_A
        p_model = calcModelProbabilitiesDependentSamples(stats, nids_A_sample, nids_B_sample, nids_C_sample, dsc_ABC)

    p_model_inv = 1 - p_model

    p_avg = np.zeros(6)
    p_avg[0] = stats["avg_A-B"]
    p_avg[1] = stats["avg_B-A"]
    p_avg[2] = stats["avg_A-C"]
    p_avg[3] = stats["avg_C-A"]
    p_avg[4] = stats["avg_B-C"]
    p_avg[5] = stats["avg_C-B"]

    p_avg = p_avg.reshape((-1,6))    
    p_avg_inv = 1 - p_avg

    masks, masks_inv = getMotifMasks(3)    

    stats["motif_probabilities_64_random"] = {}
    stats["motif_probabilities_64_model"] = {}

    for i in range(0, masks.shape[0]):
        mask = masks[i,:]
        mask_inv = masks_inv[i,:]
        
        p_motif_random = calcMotifProbability(p_avg, p_avg_inv, mask, mask_inv) 
        p_motif_model = calcMotifProbability(p_model, p_model_inv, mask, mask_inv) 
        
        motifKey = tuple(mask.astype(int))
        stats["motif_probabilities_64_random"][motifKey] = p_motif_random
        stats["motif_probabilities_64_model"][motifKey] = p_motif_model


def getMergedUnique(nidsA, nidsB, nidsC):
    merged = np.concatenate((nidsA, nidsB, nidsC))
    mergedUnique, reverseIndices = np.unique(merged, return_inverse=True)
    return mergedUnique, reverseIndices  


def calcMotifDistribution(dscPerPreNeuron):
    idsPre = list(dscPerPreNeuron.keys())
    idsPre.sort()
    ids = np.array(idsPre).astype(int)
    stats = {}
    calcMotifProbabilities(stats, ids, ids, ids, dscPerPreNeuron, dscPerPreNeuron, dscPerPreNeuron, True)            
    stats["motif_probabilities_16_random"] = aggregateProbabilties_16(stats["motif_probabilities_64_random"])
    stats["motif_probabilities_16_model"] = aggregateProbabilties_16(stats["motif_probabilities_64_model"])
    return stats


  
def calcDeviation(p_random, p_model, maxDeviation = 1000000000000):    
    if(p_random > 0):
        deviation = p_model / p_random
        return min(deviation, maxDeviation)
    elif(p_model == 0):
        return 1
    else:
        return maxDeviation


def getDeviations(p_random_values, p_model_values, idx_offset=0):
    kMax = 16
    deviations = np.zeros(kMax)
    for k in range(0, kMax):
        deviations[k] = calcDeviation(p_random_values[k+idx_offset], p_model_values[k+idx_offset])
    return deviations
    