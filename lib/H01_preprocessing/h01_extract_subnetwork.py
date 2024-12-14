import os
import sys
import numpy as np
import math
from pyvis.network import Network
import networkx as nx

import h01_util
import h01_constants

if __name__ == "__main__":

    baseFolder = h01_util.getBaseFolder()
    outfolder = os.path.join(baseFolder, "eval", "subnetworks")
    h01_util.makeDir(outfolder)

    if(os.path.exists(sys.argv[1])):
        preNeuronIds = list(h01_util.loadIds(sys.argv[1]))
    else:
        preNeuronIds = [int(sys.argv[1])]

    for preNeuronId in preNeuronIds:
        outfile = os.path.join(outfolder, "{}.html".format(preNeuronId))
        
        filenameSynapses = os.path.join(baseFolder, "synapses", "filtered", "synapses-classified-neurons.csv")
        connections = h01_util.loadConnections(filenameSynapses)
            
        pre_key, post_connections = h01_util.getConnectionsFromPre(connections, preNeuronId)
        
        neuronId_idx = {}
        neuronId_idx[pre_key[0]] = 0
        for post_key in post_connections:
            neuronId_idx[post_key[0]] = len(neuronId_idx)
        print(sorted(neuronId_idx.keys()))

        def getShape(celltypeId):
            celltype = h01_constants.getCelltypeFromId(celltypeId)
            if(celltype == "PYR"):
                return "triangle"
            else:
                return "circle"

        def getEdgeColor(compartmentId):
            if(compartmentId == h01_constants.getLabelId("DENDRITE")):
                return "blue"
            elif(compartmentId == h01_constants.getLabelId("SOMA")):
                return "red"
            elif(compartmentId == h01_constants.getLabelId("AIS")):
                return "orange"
            else:
                raise ValueError(compartmentId)
        
        def getNodePosition(nodeIdx, nNodes, radius=150):
            phi = nodeIdx * 2 * math.pi / nNodes
            return [radius * math.cos(phi), radius * math.sin(phi)]

        nt = Network(directed=True)    
       
        #nt.set_options('{"layout": {"randomSeed":5}}')

        nt.add_node(neuronId_idx[pre_key[0]], size=12, color="dimgrey", shape=getShape(pre_key[1]), title=str(pre_key[0]), label=" ", x=0, y=0, physics=True)
        nodeIdx = 0 
        for post_key, compartments in post_connections.items():
            nodePosition = getNodePosition(nodeIdx, len(post_connections))
            nodeIdx += 1
            nt.add_node(neuronId_idx[post_key[0]], size=12, color="dimgrey", shape=getShape(post_key[1]), title=str(post_key[0]), label=" ", x=nodePosition[0], y=nodePosition[1], physics=True)
            for compartmentId, count in compartments.items():                
                nt.add_edge(neuronId_idx[pre_key[0]], neuronId_idx[post_key[0]], color=getEdgeColor(compartmentId), label=str(count))    
        
        #nt.set_edge_smooth('dynamic')
        nt.barnes_hut()
        with open(outfile, "w") as f:
            f.write(nt.generate_html())
        

        

