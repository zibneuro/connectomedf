# Dissecting origins of wiring specificity

## Setting up Python environment 
The recommended way of setting up your local Python environment is using [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html).

```
cd <path-to-repository>
conda create --name generative-modeling python=3.9
conda activate generative-modeling
pip install -r requirements.txt
conda install pytorch
```
Next, install the Jupyter kernel for running notebooks:
```
conda activate generative-modeling:
pip install ipykernel
python -m ipykernel install --user --name i --display-name "wiring-specificity"
```

## Regenerating figures
Run notebooks in the following order to regenerate figures:
### Mouse viusal cortex dataset (VIS)
```
VIS_synapse_preparation.ipynb 
VIS_run_models.ipynb
VIS_population_level.ipynb
VIS_cellular_level.ipynb
VIS_realizations.ipynb
VIS_celltype_clustering.ipynb
VIS_overlap_comparison.ipynb
```
### Human temporal cortex dataset (H01)
```
H01_synapse_preparation.ipynb
H01_run_models.ipynb
H01_population_model.ipynb
```
### Simulation-based inference (SBI)
```
SBI_example.ipynb
SBI_parameter_distributions.ipynb
```


## Notes

### Cuda setup

Show GPU utilization
```
nvidia-smi
ps -o user= -p <PID>
```
Select GPU from code
```
import cupy as cp
cp.cuda.Device(3).use()

LD_LIBRARY_PATH=/software/CUDA/cuda-12.2/lib64
```