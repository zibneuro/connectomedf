import os
import sys
import numpy as np
import pandas as pd


if __name__ == "__main__":
    dataFolder = sys.argv[1]
    metaFolder = os.path.join(dataFolder, "meta")

    somas = pd.read_csv(os.path.join(metaFolder, "soma.csv"))
    
    pyr_100axon = set(somas[somas["axon_length"] >= 100]["neuron_id"].to_numpy(int))
    print(pyr_100axon)