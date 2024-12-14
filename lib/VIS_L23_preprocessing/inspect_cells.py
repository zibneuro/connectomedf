import os
import sys


def getProofreadSynapseIds(filename):
    ids = set()    
    with open(filename) as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            parts = lines[i].rstrip().split()
            preId = int(parts[4])
            postId = int(parts[5])
            ids.add(preId)
            ids.add(postId)
    return ids


if __name__ == "__main__":
    filename = "/srv/public/datasets/VIS-L23/211019_pyc-pyc_subgraph_v185.csv"

    pyrIds = getProofreadSynapseIds(filename)
    print(pyrIds)
    print(len(pyrIds))