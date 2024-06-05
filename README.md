# Reinforcement Learning in Doom

The project aims for exploring [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) possibilities in a dynamic computer game environment using the [ViZDoom](https://vizdoom.farama.org/) library and [OpenAI Gym](https://gymnasium.farama.org/).
The main objective was to train an agent capable of completing a given game level after prior learning.


## Doom scenarios
Doom, being a first person shooter, allows for a variety of game levels called [scenarios](https://vizdoom.farama.org/environments/default/) to be played. The scenarios used in this project are:

- [`Basic`](https://vizdoom.farama.org/environments/default/#basic) - a simple scenario with a single room and one enemy to kill
- [`Defend The Center`](https://vizdoom.farama.org/environments/default/#defend-the-center) - our character is placed in the center of a room and has to defend it from enemies that approach from every side possible
- [`Deadly Corridor`](https://vizdoom.farama.org/environments/default/#deadly-corridor) - the objective is to reach the vest waiting at the end of the corridor, whereas along the way there are enemies shooting at the player
- [`My Way Home`](https://vizdoom.farama.org/environments/default/#my-way-home) - peculiar scenario in which the player has to find a way out of a maze-like environment, without enemies

## Repo structure explained

- `scenarios/` - contains the files necessary to run the scenarios. Each scenario has its own pair of files
  - `.cfg` - configuration file for the scenario
  - `.wad` - file that contains level data such as textures etc.

- `utils/` - contains two files setting the foundation for training RL agents
  - `environments.py` - describes class VizDoomGym that serves as a wrapper for setting up game instances for training
  - `callbacks.py` - contains callback class for logging training progress and saving models at various stages of training process

- `Basic/`, `DefendTheCenter/`, `DeadlyCorridor/`, `MyWayHome/` - each scenario has its own directory containing the implementation of the training in the according `.py` file.
In order to start working on Your own scenario, it is recommended to copy one of our existing directories and modify the code accordingly. 
Adjusting the code includes setting desired paths for logs and models, changing the scenario name, and altering the training parameters such as learning rate or number of training episodes.

- `training/logs` - log files from the training process may be found here. To obtain accces to the log dashboards follow these steps:
    - navigate in command line to the directory in which the log file is
    - type in cmd `tensorboard --logdir .`
    - open the browser and go to ` http://localhost:6006/`

- `training/train` - contains folders with the trained models obtained throughout the training process. The models may be trained further from the point they were saved. 


## RL algorithms used

### Proximal Policy Optimization (PPO)
PPO is an on-policy method as it learns through direct interaction with the environment by executing actions. These methods rely on the use of two neural networks: an agent containing the policy for action selection and a network predicting the reward value at a given time for a given state. Learning involves the agent's interaction with the environment, comparing the received reward to the expected reward, and updating the network weights in a way that favors actions that have yielded the highest relative reward.

### Advantage Actor Critic (A2C)

A2C is a learning algorithm that combines the concepts of value-based and policy-based learning. In actor-critic methods, we train two networks: the actor, which represents the policy function responsible for indicating actions, and the critic, which is the value function that evaluates the action taken by the actor.
