import os
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

def get_data_dir():
    return Path(__file__).parent.parent.parent / "data"


def get_celltype_map(soma_file):
    somas = pd.read_csv(soma_file)
    celltype_map = {} # neuron id -> celltype
    for _, row in somas.iterrows():
        celltype_map[row["neuron_id_mapped"].astype(int)] = row["celltype"].astype(int)
    return celltype_map

def parsePosition(posAsString):
    pruned = posAsString.replace("[","").replace("]","")
    return np.fromstring(pruned, dtype=int, sep=" ")


def makeCleanDir(dirname):
    if(os.path.exists(dirname)):
        shutil.rmtree(dirname, ignore_errors=False, onerror=None)
    os.mkdir(dirname)


def loadNeuronIdMapping(somaFile):
    mapping = {}
    with open(somaFile) as f:
        lines = f.readlines()
        for line in lines[1:]:
            neuronId = int(line.rstrip().split(",")[3])
            mappedNeuronId = int(line.rstrip().split(",")[4])
            mapping[neuronId] = mappedNeuronId
    return mapping


def loadIds(filename, sortedList = False):
    ids = set()
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            ids.add(int(line.rstrip()))
    if(sortedList):
        return sorted(ids)
    else:
        return ids

def loadConnectionsPycPyc(filename):
    connections = {}    
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            parts = lines[i].rstrip().split(",")
            preId = int(parts[2])
            postId = int(parts[3])
            key = (preId, postId)
            if(key not in connections):
                connections[key] = 0
            connections[key] += 1
    return connections