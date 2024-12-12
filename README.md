# resnet-pipeline-opt
Pipelining optimization for ResNet50 as part of a CSDS 451 project.

## Project Overview
As neural networks have grown in size and complexity over the last decade, computer hardware has posed a significant limitation to the training and performance of large models. Pipelining offers a solution for long training times by implementing parallelization of model training. The objective of this project is to implement a framework for efficient model pipelining at the layer level for the Resnet50 model and evaluate its performance against a baseline implementation.

## Running Baseline Examples

Some of the baselines (gpipeline) actually require the newest version of pytorch (so can't be run with the same module list as the MPI code).
```
module load Python/3.12.3-GCCcore-13.3.0
python -m venv venv
source venv/bin/activate
pip install torch torchvision lightning
```
Then we can proceed to run the baseline examples with reources
```
srun -A sxk1942_csds451 -p markov_gpu --gres=gpu:2 --mem=4gb --pty bash
```

### PyTorch
For the PyTorch code:
```
./baseline_nopipe.py -b <batch size>
```

### PyTorch Lightning
For the lightning code, we can run:
```
./baseline_lightning.py -b <batch size>
```

### GPipeline
To run GPipeleline code using torchrun:
```
torchrun --nnodes 3 --nproc_per_node 2 <file.py>
```

## Setup and Initialization

Some of the MPI code has compatibility issues. The easiest way to resolve this is by using the HPC cluster. After gaining access to the HPC, load the following modules:
```
module load Python/3.11.3-GCCcore-12.3.0
module load mpi4py
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
```

Then request a GPU for resources to run the model. Make sure to set number of nodes (with -N) so that MPI recognizes the nodes for parallelization.
```
srun -A sxk1942_csds451 -p markov_gpu -N 3 --gres=gpu:1 --mem=4gb --pty bash
```

### Running the Pytorch Lightning Code

This needs the lightning module, so activate a virtual environment to run.

Run the application (set to 4 nodes) by running:
```
mpiexec -n 4 pipeline.py
```

