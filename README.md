# resnet-pipeline-opt
Pipelining optimization for ResNet50 as part of a CSDS 451 project.


## Setup and Initialization
After gaining access to the HPC, load the following modules:
```
module load Python/3.11.3-GCCcore-12.3.0
module load mpi4py
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
```

Then request a GPU for resources to run the model.
```
srun -A sxk1942_csds451 -p markov_gpu --gres=gpu:1 --mem=4gb --pty bash
```
