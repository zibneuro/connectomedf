import os
import sys
import pandas as pd
import glob
from pathlib import Path

if __name__ == "__main__":
    metaFolder = Path("/srv/public/gencon/data/VIS/meta")

    morphologiesFolder = Path("/scratch/visual/bzfharth/datasets/VIS-L23/morphologies_stl")
    df_soma = pd.read_csv(metaFolder/"soma.csv", dtype={"neuron_id": str})
    df_soma["mesh_available"] = 0

    for idx, row in df_soma.iterrows():
        neuron_id = row["neuron_id"]
        mapped_neuron_id = row["neuron_id_mapped"]
        stl_file = morphologiesFolder/"{}.stl".format(neuron_id)
        renamed_file = morphologiesFolder/"{}.stl".format(mapped_neuron_id)
        if(stl_file.exists()):
            print(stl_file)
            print(renamed_file)
            os.rename(stl_file, renamed_file)
            df_soma.at[idx, "mesh_available"] = 1

    df_soma.to_csv(metaFolder/"soma.csv", index=False)
