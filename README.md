# Dissecting origins of wiring specificity

## Local setup

### System requirements
- Tested on a Linux system with **NVIDIA GeForce RTX 4080 (16GB)** 
- The recommended way of setting up your local Python environments is using [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html).

### Create Main Python Environment 
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
### Create Python Environment for Synapse Data Preprocessing (optional) 
Requires [cuDF](https://github.com/rapidsai/cudf), which has extended GPU hardware requirements; please adjust for your local GPU setup.
```
cd <path-to-repository>
conda create -n preproc python=3.11
conda activate preproc 

pip install cudf-cu11==24.6.0
pip install ipykernel==6.29.5

python -m ipykernel install --user --name i --display-name "preproc"
```

## Regenerating figures

Download data used in the analyses from:

https://doi.org/10.5281/zenodo.14507539 

(access will be granted upon request, please contact <harth@zib.de>).

To regenerate the figures, run notebooks in the following order  using the connectomedf Python environment.
```
VIS_run_models.ipynb
VIS_population_level.ipynb
VIS_cellular_level.ipynb
VIS_realizations.ipynb
VIS_celltype_clustering.ipynb
VIS_overlap_comparison.ipynb

H01_synapse_preparation.ipynb
H01_run_models.ipynb
H01_population_model.ipynb

SBI_example.ipynb
SBI_parameter_distributions.ipynb
```

To rerun the synapse preprocessing step (i.e., spatial partitioning into overlap volumes), run the following notebooks using the preproc-Python environment.
```
VIS_synapse_preparation.ipynb
H01_synapse_preparation.ipynb 
```

## Publication
Dissecting origins of wiring specificity in dense reconstructions of neural tissue.
Philipp Harth, Daniel Udvary, Jan Boelts, Jakob H Macke, Daniel Baum, Hans-Chrisitian Hege, Marcel Oberlaender.
*bioRxiv* 2024.12.14.628490; doi: https://doi.org/10.1101/2024.12.14.628490
