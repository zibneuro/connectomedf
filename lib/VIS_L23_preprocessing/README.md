## Required raw data files:

### soma_valence_v185.csv
https://github.com/AllenInstitute/ChandelierL23/tree/main/anatomical_analysis/data/soma_valence_v185.csv

copy into folder: VIS-L23/data_gitrepo_ChandelierL23

### soma.csv
https://doi.org/10.5281/zenodo.3710458 

211019_vignette_motif_analysis_data.tgz: vignette_motif_base_data/data/soma.csv

copy into folder: data_Microns_L23

###  pni_synapses_v185.csv
https://doi.org/10.5281/zenodo.3710458 

copy into folder: data_Microns_L23

### ais_synapse_data_all_v185.h5
https://github.com/AllenInstitute/ChandelierL23/anatomical_analysis/data/ais_synapse_data_all_v185.h5

copy into folder: VIS-L23/data_gitrepo_ChandelierL23

### 211019_pyc-pyc_subgraph_v185.csv
https://doi.org/10.5281/zenodo.3710458 

copy into folder: data_Microns_L23

### layer23_v185.tar.gz
https://doi.org/10.5281/zenodo.3710458 

extract in folder: VIS-L23

## Preprocessing steps
```
vis_L23_process_soma.py <data-folder>
vis_L23_process_synapses.py <data-folder>
vis_L23_process_meshes.py <data-folder>
vis_L23_rename_morphology_files.py
```