import pandas as pd


def filter_synapses(synapse_file, filtered_file, neuron_ids):        
    synapses = pd.read_csv(synapse_file)
    neuron_ids = set(neuron_ids)

    if (synapses.columns.to_list() != ["x","y","z","pre_id","post_id","pre_celltype","post_celltype","post_compartment"]):
        raise NotImplementedError(synapses.columns.to_list())

    with open(filtered_file, "w") as f:
        f.write("x,y,z,pre_id,post_id,pre_celltype,post_celltype,post_compartment\n")

        for _, row in synapses.iterrows():
            if(row["pre_id"] in neuron_ids and row["post_id"] in neuron_ids):
                f.write("{},{},{},{},{},{},{},{}\n".format(
                    row["x"], row["y"], row["z"],
                    row["pre_id"], row["post_id"], 
                    row["pre_celltype"], row["post_celltype"], 
                    row["post_compartment"]
                ))