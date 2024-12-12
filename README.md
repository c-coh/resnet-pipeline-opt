# resnet-pipeline-opt
Pipelining optimization for ResNet50 as part of a CSDS 451 project.


## Setup and Initialization
After gaining access to the HPC, load the following modules:
```
module load Python/3.11.3-GCCcore-12.3.0
module load mpi4py
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
```

Then request a GPU for resources to run the model. Make sure to set number of nodes with -N.
```
srun -A sxk1942_csds451 -p markov_gpu -N 4 --gres=gpu:1 --mem=4gb --pty bash
```

### Running the Pytorch Lightning Code

This actually requires the newest version of pytorch (so can't be run with the same module list as the MPI code).

This needs the lightning module, so activate a virtual environment to run.
```
module load Python/3.12.3-GCCcore-13.3.0
python -m venv venv
source venv/bin/activate
pip install torch torchvision lightning
./baseline_lightning.py
```

To run GPipeleline code using torchrun:
```
torchrun --nnodes 3 --nproc_per_node 2 <file.py>
```

### Running MPI pipeline code

Run the application (set to 4 nodes) by running:
```
mpiexec -n 4 pipeline.py
```

