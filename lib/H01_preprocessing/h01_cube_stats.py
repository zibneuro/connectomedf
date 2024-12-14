import os
import numpy as np

import h01_util


if __name__ == "__main__":
    griddedSynapsesFolder = os.path.join(h01_util.getBaseFolder(), "synapses", "gridded_complete-volume_100000-100000-87500")
    filenames = h01_util.getFiles(griddedSynapsesFolder)
    total = 0
    for filename in filenames:
        synapses = np.atleast_2d(np.loadtxt(filename, skiprows=1, delimiter=","))
        numSynapses = synapses.shape[0]        
        total += numSynapses
        print(numSynapses, total)            