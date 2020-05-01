
Flexibility Design with Neural Reinforcement Learning  
==================================

## Setup 
This code is based on [the spinningup project from OpenAi](https://github.com/openai/spinningup). 
The steps are summarized below, 
1. [install spinningup prerequisites](https://spinningup.openai.com/en/latest/user/installation.html).  
We modified the installation steps to include requirements for our project below
2. install RL_flex_design
3. install Gurobi. Gurobi is needed by the Flexibility Environment 
to evaluate rewards by solving linear programming problems. 

### Step 1. Install Spinningup Prerequisites
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

_Ubuntu_
```commandline
sudo apt-get update && sudo apt-get install libopenmpi-dev
```

_Mac OS X_
If you have already installed `brew , run command below
```commandline
brew install openmpi
```
Otherwise, install `brew` first using
```commandline
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### Step 2. Install RL_flex_design
```commandline
git clone https://github.com/zhlzhl/NRL_flex_design.git
cd NRL_flex_design
pip install -e .
```

### Step 3. Install Gurobi 
From a terminal window issue the following command to add the Gurobi channel to your default search list
```commandline
conda config --add channels http://conda.anaconda.org/gurobi
```

Now issue the following command to install the Gurobi package to the RL_flex_design environment
```commandline
conda activate spinningup
conda install gurobi
```

#### Install a Gurobi License 
Install a Gurobi license (if you haven't already done so), by following the instruction from the
 [Gurobi website](https://www.gurobi.com/academia/academic-program-and-licenses/)
 to get a free license for academic use. If you are not on campus, 
 make sure you are on VPN of your affiliated school when 
 obtaining the license. 



#### Check your install
 
To see if you’ve successfully installed Spinning Up, try running PPO in the LunarLander-v2 environment with
```commandline
python -m spinup.run vpg --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
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




## FlexibilityEnv 

description of different versions of FlexibilityEnv are to be added later. 

- env_version 1 and 2 are for a special case where there is no fixed cost for each arc, and the profit matrix is one, 
i.e., each arc all have the same profit coefficient. In particular, 

..* version 1 uses the gym format where a pre-defined FlexibilityEnv is built and registered with gym. To construct
FlexibilityEnv with env_version=1, use `gym.make(Flexibility-v0)`. 

..* version 2 uses a FlexibilityEnv class which support dynamic creation of FlexibilityEnv, where the different 
settings of FlexibilityEnv can be passed into the class during creation. 

- env_version 3, 4, and 5 are for the general case of FlexibilityEnv as defined in the paper, with customizable settings 
including fixed costs, profit-matrix, and starting_structure. 

The differences among 3, 4 and 5 are that env_version 4 and 5 support dummy actions while env_version 3 does not. 
In particular, 
- env_version 3 supports both adding and removing an arc - if an action selects an arc that doesn't exist, 
add the arc to the structure, otherwise, remove the arc from the structure. 
- env_version 4 supports both adding and removing just like env_version 3. However, version 4 also has 
an additional dummy action, and if the dummay action is selected, nothing is done to the stucture, 
and reward is zero unless this is the final step then strcture performance is evaluated as reward. 
-env_version 5 only add arcs. However if a selected arc already exists, then nothing is done, which 
is equivalent to dummy actions. 

Env version 3 does well when adding arcs always increases structure performance. However, when 
adding arcs doesn't always increase structure performance due to fixed costs, Env version 4 and 5 perform 
better. We focuse on env 5 as the mechanism is simpler. 


## Algorithm 
PPO algorithm is used in our RL training. We modified the ppo.py file under /spinningup/spinup/algos/ppo/ from spinningup to run our Flexibility Env. 
ppo.py runs one experiment at a time. If multiple experiments are specified in run_flexibility.py, 
they are passed into ppo.py sequentially. 

Log files are saved in the /spinningup/data directory, unless otherwise specified in the user_config.py file. 

## Neural Network Architecture 
A three layer multi-layer perceptron (MLP) is used for both actor and critic networks. 
core.py file in the ppo directory specify the actor and critic networks. 
We modified core.py to add temperature into the policy action. 
The size of the two hidden layers of the 3-layer MLP can be specified using `--custom_h`, e.g., `--custom_h 1024-128`. 


## Run the codes

### Run the experiments 
The entry point of running the RL training using FlexibilityEnv is run_flexibility.py under /spinningup/spinup. 
It allows multiple experiments to be run using ExperimentGrid, a tool from OpenAI. 
For example, the commandline below allows 12 variations of experiments to be run sequentially - 
 FlexibilityEnv of each arget_arcs, 27, 29, 31, 33 is run three times, each with a different random seed. 

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
    --num_runs 3
```

Note that each value of `--target_arcs`, e.g., 27, leads to a unique FlexibilityEnv environment, i.e., a unique game to play. 
If you want to run each FlexibilityEnv with different seeds, you can achieve this by specifying `--num_runs` or `--seed`. 
For example, if `--num_runs 3` is given, then the program will automatically create three seeds with values 0, 10, and 20. 
If you want to set seeds explicitely (e.g., 300, 310, 320), use `--seed 300, 310, 320`. 

The above variations is automated through ExperimentGrid from spinup. If more than one `--target_arcs` are specified, 
 for each target_arcs a separate data directory is 
created for logging, with '*tar{n}' as part of the auto-generated directory name. Otherwise, if there is only 
one `--target_arcs` value, the log directory name will not be identified by '*tar{n}'. Therefore, if you want to run 
multiple experiments with different `--target_arcs`, it is a good practice to specify at least two values for `--target_arcs`.  

If multiple runs/seeds are specified, 
under this directory, multiple sub-directories are created with one for each run (or random seed); the sub-directory 
name is identified by appending '*s{n}' to the directory name. 


Shell scripts can also be created to invoke experiments by calling run_flexibility.py. 
Check out one of the examples "run_experiments...." under /spinningup. 

Below is an example of running a bash script from command line. 
```
(spinningup) user@ubuntu:~/git/RL_flex_design$ ./run_10x10a-lspe_ENV5_batch0_0.sh
```

To facilitate running scripts in parallel, a script `run_script_generator.py` can be used to generate Shell scripts 
automatically. For example, you can specify below parameters. 
```python
    # specify parameters
    experiment = '10x10a-lspe'
    env_input = get_input(experiment)
    env_version_list = [5]
    epoch_episodes = 800
    num_tars_per_script = 4
    # the number of entrypoints to be created with different seeds and everything else the same, 
    # the purpose is to do more parallelization
    num_batches = 2  
    # the number of runs with different seed for each target arc
    num_runs = 3  
```
This will create two batches of scripts. You can consider the two batches are running the same set of experiments but 
with different seeds. 
Each batch includes one entrypoint script, and several sub_scripts to be called by the entrypoint script to run in parallel. 
You can consider the sub_scripts are running experiments with different groups of target_arcs. 
Each sub_script will run experiments with different target_arcs in sequence, and the number of target_arcs is decided by 
`num_tars_per_script`. 

The above example allows us to run two entrypoint scripts side by side, and each calls four sub_scripts to run in parallel, 
therefore, allowing eight experiments to run in parallel, subjecting to constraint of computer hardware. 

### Visualization 
We can visualize how arcs are been added/removed during training and evaluation. However, visualization during training 
is turned off to speed up training. Visualization during evaluation is also turned off by default. If you want to turn it on, 
there is a quick way to set it through `spinup/user_config.py` by setting `FORCE_RENDER` to `True`. 

### Monitor the training progress 
Tensorboard is used to monitor the training progress. 
If tensorboard is installed, launch it from command line under the same conda environment by running 
```commandline
tensorboard --logdir=data
```

Note that the default behavior is during training, rewards are calculated using 50 samples of demand to evaluate the structgure, 
while during the evaluation, rewards are calculated using 5000 samples. 
Therefore, one important metric to look for is `Eval_EpRet-Max`, which shows the structure performance of the best out of 
50 structures that are generated during each evaluation every `--save_freq` epoch. 


### Collect the best structures 
Evaluation of trained models are carried during training which is specified through a combination of 
e.g., `--save_frequency 10` and `--do_checkpoint_eval`. Trained models at each checkpoint is 
saved into `simple_save` directory y, or a `simple_save{N}` (N is the epoch number if specified so)
 under the log director. 
"simple_save999999" directory saves the best model that has been trained so far, and also the best structure of the 50 structures 
generated by the best model. 
After the experiments are finished, we can run `/spinningup/best_structures/collect_best_structures.py`  
to collect the best structures generated during training. 

For example, we can specify below to collect best structures for experiment `10x10b` with FlexibilityEnv of env_version 4 and 4. 
```python
    experiment = '10x10b'
    envs = ['ENV4', 'ENV5']
```

The script will look into the log directory, `data` by default, and look for directories which include `10x10b`
in directory name, under those directories, look for `simple_save999999` sub-directories where best model and the best structure 
is stored. Best structure is saved in file with prefix `best_eval_performance_n_structure`. 
After collecting the best structures of different runs of the same experiment, in the `collect_best_structures.py` script, 
best structures are evaluated again using a fixed evaluation data to find the **best** structure. 
Best structures are then stored in directory `10x10b` under `best_structures`, with subdirectories `ENV4` and `ENV5`, etc. 



## Directory structure 
is mostly unchanged from spinningup. A few changes are made as listed below: 
- ppo.py has been modified and is the main algorithm we use. vpg.py is also modified, but is not completed yet. 
- Subdirectory `/spinningup/spinup/FlexibilityEnv` is created to keep FlexibilityEnv.py, which is a custom Flexibility Environment 
that is created to follow gym environment format. 
- Subdirectory `/spinningup/spinup/FlexibilityEnv_input` is created to keep input files which specify settings for FlexibilityEnv objects for different experiments. 
- run_flexibility.py is created to run rl games with the FlexibilityEnv. It is modified based on run.py
- A few utils files are created under `/spinningup/spinup/utils` 
- Bash scripts are created under `/spinningup` to load multiple experiments 
- `/spnningup/best_structure directory` is created to collect best structures generated during training. 

