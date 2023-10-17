# Evaluate SuSIE on CALVIN benchmark

This fork of the CALVIN repo contains code for running a trained SuSIE model on the CALVIN benchmark. 

## Installation
1. ```git clone --recurse-submodules https://github.com/pranavatreya/calvin-sim.git```
2. ```conda create -n susie-calvin python=3.8```
3. Install [tensorflow](https://www.tensorflow.org/install/pip) and [JAX](https://jax.readthedocs.io/en/latest/installation.html)
4. ```sh install.sh``` (for troubleshooting see https://github.com/mees/calvin)

You will also need to install in the same conda environment two other repos that the SuSIE code interfaces with, [BridgeData V2](https://github.com/rail-berkeley/bridge_data_v2) and [denoising-diffusion-flax](https://github.com/kvablack/denoising-diffusion-flax). This can be done by simply cloning the repos and pip installing them.

## Download checkpoints

Download the diffusion model and goal conditioned policy checkpoints from https://huggingface.co/patreya/susie-calvin-checkpoints

## Evaluation

To evaluate SuSIE on CALVIN,

1. Set the values of the environment variables in ```eval_susie.sh``` to the paths to your downloaded checkpoints.
2. Run ```./eval_susie.sh```

Videos of SuSIE rollouts on CALVIN will be logged to the ```experiments``` folder.

