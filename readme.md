
Flexibility Design with Neural Reinforcement Learning  
==================================

## Setup 
This code is based on [the spinningup project from OpenAi](https://github.com/openai/spinningup). 
The steps are summarized below, 
- [install spinningup prerequisites](https://spinningup.openai.com/en/latest/user/installation.html).  
We modified the installation steps to include requirements for our project below
- install RL_flex_design
- install Gurobi. Gurobi is needed by the Flexibility Environment 
to evaluate rewards by solving linear programming problems. 

### Install Spinningup Prerequisites
Spinning Up requires Python3, OpenAI Gym, and OpenMPI. Although Gym is not needed for our project, 
We recommend to install it so that we can verify the installation of Spinning UP. 

#### Install Python 
```commandline
conda create -n spinningup python=3.6
```

To use Python from the environment spinningup, activate the environment with: 
```commandline
conda activate spinningup 
```

#### Install OpenMPI 
##### Ubuntu
```commandline
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
##### Mac OS X
```commandline
brew install openmpi
```

### Install RL_flex_design
```commandline
git clone https://github.com/zhlzhl/RL_flex_design.git
cd RL_flex_design
pip install -e .
```
#### Check your install 
To see if you’ve successfully installed Spinning Up, try running PPO in the LunarLander-v2 environment with
```commandline
python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
```
This might run for around 10 minutes, and you can leave it going in the background while you continue reading through documentation. This won’t train the agent to completion, but will run it for long enough that you can see some learning progress when the results come in.

After it finishes training, watch a video of the trained policy with
```commandline
python -m spinup.run test_policy data/installtest/installtest_s0
```

And plot the results with
```commandline
python -m spinup.run plot data/installtest/installtest_s0
```

### Install Gurobi 
From a terminal window issue the following command to add the Gurobi channel to your default search list
```commandline
conda config --add channels http://conda.anaconda.org/gurobi
```

Now issue the following command to install the Gurobi package to the RL_flex_design environment
```commandline
conda activate RL_flex_design
conda install gurobi
```

#### Install a Gurobi License 
Install a Gurobi license (if you haven't already done so), by following the instruction from the Gurobi website. 

## Run the codes
The entry point of running the RL training using FlexibilityEnv is run_flexibility.py under /spinningup/spinup. 
It allows multiple experiments to be run using ExperimentGrid, a tool from OpenAI. 
For example, below snipet allows four experiments to be run sequentially, each with a different target_arcs, 27, 29, 31, 33. 

```commandline
python -m spinup.run_flexibility 
    --algo ppo  
    --env_name Flexibility8x16_SP50-v3 
    --exp_name Flexibility8x16_SP50_PPO_CH1024-128_ENV3_JG  
    --cpu 8 
    --epochs 400  
    --save_freq 10  
    --custom_h 1024-128 
    --env_version 3 
    --env_input inputJG_m8n16_cv0.4.pkl 
    --target_arcs 27 29 31 33
```

Bash scripts can also be created to invoke experiments by calling run_flexibility.py. 
Check out one of the examples "run_experiments...." under /spinningup. 

Below is an example of running a bash script from command line. 
```
(spinningup) user@ubuntu:~/git/RL_flex_design$ ./run_experiments_CH1024-128_ENV3_input_ran10x10a_cv0.8.sh
```

## Algorithm 
PPO algorithm is used in our RL training. We modified the ppo.py file under /spinningup/spinup/algos/ppo/ from spinningup to run our Flexibility Env. 
ppo.py runs one experiment at a time. If multiple experiments are specified in run_flexibility.py, 
they are passed into ppo.py sequentially. 

Log files are saved in the /spinningup/data directory, unless otherwise specified in the user_config.py file. 

## Neural Net Architecture 
A three layer multi-layer perceptron (MLP) is used for both actor and critic networks. 
core.py file in the ppo directory specify the actor and critic networks. 
We modified core.py to add temperature into the policy action. 
The size of the two hidden layers of the 3-layer MLP can be specified using `--custom_h`, e.g., `--custom_h 1024-128`. 

## Monitor the training progress 
Tensorboard is used to monitor the training progress. 
If tensorboard is installed, launch it from command line under the same conda environment by running 
```commandline
tensorboard --logdir=data
```

## Collect genereated structures 
Evaluation can be carried during training which can be specified through a combination of 
e.g., `--save_frequency 10` and `--do_checkpoint_eval`. Trained models at each checkpoint is 
saved into a "simple_saveN" (N is the epoch number) directory under the log directory of this experiment under data. 
"simple_save999999" directory saves the best model so far, and also the best structure generated by the best model. 
After the experiments are finished, we can run "collect_best_structure.py" file under /spinningup/best_structures
to collect the best structures generated during training. To identify best structures from which
experiments to be collected, use "exp_settings.py" file. 

## Directory structure 
is mostly unchanged from spinningup. A few changes are made as listed below: 
- ppo.py has been modified and is the main algorithm we use. vpg.py is also modified, but is not completed yet. 
- Subdirectory /spinningup/spinup/FlexibilityEnv is created to keep FlexibilityEnv.py, which is a custom Flexibility Environment 
that is created to follow gym environment format. 
- Subdirectory /spinningup/spinup/FlexibilityEnv_input is created to keep input files which specify settings for FlexibilityEnv objects for different experiments. 
- run_flexibility.py is created to run rl games with the FlexibilityEnv. It is modified based on run.py
- A few utils files are created under /spinningup/spinup/utils 
- Bash scripts are created under /spinningup to load multiple experiments 
- /spnningup/best_structure directory is created to collect best structures generated during training. 

