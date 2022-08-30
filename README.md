# Masterthesis: Meta-Reinforcement Learning with Augmented Data in Contunious Control Tasks with Sparse Rewards

by Atef Kort (Technical University Munich)

This is the reference implementation of the `Distributional Meta-RL with Augmented Data (DMRLAD)` algorithm.
The implementation is based on [rlkit](https://github.com/vitchyr/rlkit), [PEARL](https://github.com/katerakelly/oyster) and [CEMRL]. 

--------------------------------------

## Installation

### Mujoco
For our experiments we use MuJoCo200, however due to old PEARL dependencies, older versions have to be installed, too:
- Get a [MuJoCo](https://www.roboti.us/index.html) license key. Follow the instructions.
- Put your key file in `~/.mujoco`.
- Download the versions mujoco131, mujoco150, mujoco200 and put them in `~/.mujoco`.
- Set `LD_LIBRARY_PATH` to point to all the MuJoCo binaries (`~/.mujoco/mujoco200/bin`, mujoco150, mujoco200 respectively).
- Set `LD_LIBRARY_PATH` to point to your gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).

### Conda environment
For the remaining dependencies, we recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Use the `environment.yml` file to [set up](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) a conda virtual machine.
Make sure the correct GPU driver is installed and you use a matching version of CUDA toolkit for your GPU.

### Dependencies

#### Old PEARL dependency
- Clone the [rand_param_envs](https://github.com/dennisl88/rand_param_envs.git) repository to `/path/to/rand_param_envs`.

#### Modified Gym environments:
We created our own versions of the standard [MuJoCo](https://www.roboti.us/index.html) / [Gym](https://gym.openai.com/) environments.

#### Meta-World
- Clone the [metaworld](https://github.com/rlworkgroup/metaworld) repository to `/path/to/metaworld`.

### Installation of dependencies
Install all previous dependencies to the conda environment in dev-mode.
```
cd /path/to/dependency
conda activate dmrlad
pip install -e .
```
This installation has been tested only on 64-bit Ubuntu 18.04.

## Run experiments
To reproduce an experiment, run:
```
conda activate dmrlad
python runner.py configs/[EXP].json
# Options:
--use_mp            # parallelize data collection across num_workers
--num_workers=8     # configure number of workers, default: 4
--gpu=2             # configure GPU number, default: no GPU
```
A working starting example is `python runner.py configs/cheetah-stationary-vel.json`.
Experiments in `configs/others` are deprecated and might not work.
- Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name is uniquely generated based on the date.
The file `progress.csv` contains statistics logged over the course of training, `variant.json` documents the used parameters,
  further files contain pickled data for specific epochs like weights, encodings.
  

## Analyze experiments
With the script `analysis_runner.py` you can do the following:
log to database, plot rewards, plot encodings, showcase the learned policy etc.
- copy the path of a specific experiment to `path_to_weights` in `configs/analysis_config.py`,
  select which parts of the analysis should be done (by setting them `True` in the `analysis_params`).
- run `python analysis_runner.py`


## Things to notice
- Most relevant code for the `dmrlad` algorithm is in the folder `./dmrlad`.
- We use [ray](https://docs.ray.io/en/master/) for data collection parallelization.
Make sure to configure a suitable amount of `num_workers` to not crash the program.
- Experiments are configured via `json` configuration files located in `./configs`, include a basic default configuration `./configs/default.py`.
- Environment wrappers are located in `rlkit/envs`.
- Adjust the `max_replay_buffer_size` according to the amount of collected data and supported memory.