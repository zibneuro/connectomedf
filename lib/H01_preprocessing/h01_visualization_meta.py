import pandas as pd
from pathlib import Path
from glob import glob
from h01_constants import *


def getRandomSubset(itemSet, n, randomSeed = None):
    if(n >= len(itemSet)):
        return itemSet

    array = list(itemSet)
    if(randomSeed):
        np.random.seed(randomSeed)
        array.sort()
    np.random.shuffle(array)
    return set(array[0:n]) 


def write_sampled_lists(ids, folder, layerCelltype):
    for k in range(10,511,20):
        if(k<=len(ids)):
            sampled = getRandomSubset(ids, k)
            sampled = sorted(sampled)
            filename = folder/f"ids_{layerCelltype}_sampled_{k}"
            with open(filename, "w") as f:
                for neuron_id in sampled:
                    f.write(f"{neuron_id}\n")


data_folder = Path("/srv/public/python-ide/generative-modeling-dev/data/H01")

morphology_folder = data_folder/"morphologies"/"neuron"

meta_vis_folder = data_folder/"meta"/"visualization"
meta_vis_folder.mkdir(parents=True, exist_ok=True)

morphology_files = glob(f"{morphology_folder}/*.swc")
ids_with_morphology = set()

for filename in morphology_files:
    ids_with_morphology.add(int(Path(filename).stem))


neurons_all = {}

for layer in getLayers():
    for celltype in getCelltypes():
        celltypeId = getCelltypeId(layer, celltype)
        filename = data_folder/"meta"/f"ids_{layer}_{celltype}"
        labelled_ids = set(np.loadtxt(filename).astype(int))

        available_ids = ids_with_morphology & labelled_ids
        if(len(available_ids)):
            available_ids = sorted(available_ids)
            for neuron_id in available_ids:
                neurons_all[neuron_id] = celltypeId

            write_sampled_lists(available_ids, meta_vis_folder, f"{layer}_{celltype}")


ids = sorted(neurons_all.keys())
with open(meta_vis_folder/"neurons", "w") as f:
    for neuron_id in ids:
        f.write(f"{neuron_id} {neurons_all[neuron_id]}\n")
