# Dissecting origins of wiring specificity

## Standalone demo

A standalone demo of the code is available in [Google Colab](https://colab.research.google.com/drive/1D2xk0fsGmt_-xT6k4mZFA9iKafoLSULg?usp=sharing)
 (expected runtime ca. 5 minutes).

## Local setup

### System requirements
- Linux OS with a Nvidia GPU.
- Tested on Debian 12 with **NVIDIA GeForce RTX 4080 (16GB)** 
- Synapse data preprocessing (gridding into overlap volumes) requires [cuDF](https://github.com/rapidsai/cudf), which is only available under Linux and requires a GPU with CUDA 12.0+ and Compute Capability >=7.0.  

### Installation instructions
Expected installation time ca. 45 minutes.

The recommended way of setting up your local Python environments is using [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html).
```
cd <path-to-repository>
conda create -n connectomedf python=3.11
conda activate connectomedf
pip install -r requirements.txt
```
Manually install a [CuPy version](https://docs.cupy.dev/en/stable/install.html) that matches your local GPU setup, e.g.:
```
pip install cupy-cuda12x==13.2.0
```
Register the conda environment with **ipykernel** to use it from within Jupyter notebooks:
```
conda activate connectomedf:
python -m ipykernel install --user --name i --display-name "connectomedf"
```
Create a separate Python Environment for Synapse Data Preprocessing. This step requires [cuDF](https://github.com/rapidsai/cudf), which has extended GPU hardware requirements; please adjust for your local GPU setup.
```
cd <path-to-repository>
conda create -n preproc python=3.11
conda activate preproc 

pip install cudf-cu11==24.6.0
pip install ipykernel==6.29.5

python -m ipykernel install --user --name i --display-name "preproc"
```

### Instructions for use
To apply the framework to custom dense connectome data, a csv-file of all synapses in the reconstructed volume is required. 

The fields in the csv-file depend on the available meta information (e.g., cell type, subcompartment label). Mandatory fields are the spatial coordinates of the synapses x,y,z in nanometers (if the reconstruction contains separate locations for pre- and postsynaptic sites, we compute the center point), presynaptic neuron ID, and postsynaptic neuron ID.

For example, the raw synapses file used in our analyses looks as follows:
```
x,y,z,pre_id,post_id,pre_celltype,post_celltype,post_compartment
442184,191008,68080,648518346349368558,648518346349532006,-1,1,2
277176,178544,8320,648518346341406765,648518346349537692,-1,1,2
349024,279152,25440,648518346342407933,648518346349517783,-1,20,4
...
```
Note that unknown values (e.g., unknown cell type labels) are encoded as -1.

Please refer to the notebooks `VIS_synapse_preparation.ipynb` and `H01_synapse_preparation.ipynb` as examples how to convert the raw synapses file into the preprocessed file in which each synapse has been assigned to subvolumes.


### Regenerating figures

Download complete data used in the analyses from:

https://doi.org/10.5281/zenodo.14507539 

(access will be granted upon request, please contact <harth@zib.de>).

To regenerate the figures, run notebooks in the following order  using the connectomedf Python environment (expected runtime ca. 6h).
```
VIS_run_models.ipynb
VIS_population_level.ipynb
VIS_cellular_level.ipynb
VIS_realizations.ipynb
VIS_celltype_clustering.ipynb
VIS_overlap_comparison.ipynb

H01_run_models.ipynb
H01_population_model.ipynb

SBI_example.ipynb
SBI_parameter_distributions.ipynb
```

To rerun the synapse preprocessing step (i.e., spatial partitioning into subvolumes), run the following notebooks using the preproc-Python environment.
```
VIS_synapse_preparation.ipynb
H01_synapse_preparation.ipynb 
```

## Publication
Dissecting origins of wiring specificity in dense reconstructions of neural tissue.
Philipp Harth, Daniel Udvary, Jan Boelts, Jakob H Macke, Daniel Baum, Hans-Chrisitian Hege, Marcel Oberlaender.
*bioRxiv* 2024.12.14.628490; doi: https://doi.org/10.1101/2024.12.14.628490
