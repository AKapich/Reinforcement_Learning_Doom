from Environments_deadly_corridor import VizDoomGym as Env

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.callbacks import TrainAndLoggingCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN

CHECKPOINT_DIR = './training/train/train_deadly_corridor'
LOG_DIR = './training/logs/log_deadly_corridor'


callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

# train
# env = Env(render=False, scenario='deadly_corridor_medium_multiple_variables')
# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
# model.learn(total_timesteps=1800000, callback=callback)

#train
# env = Env(render=False, scenario='deadly_corridor_easy')
# model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=2048)
# model.learn(total_timesteps=400000, callback=callback)


# test
env = Env(render=False, scenario='deadly_corridor_medium_multiple_variables')
model = PPO.load(CHECKPOINT_DIR + '/best_model_1600000', env=env)
mean_reward, std_reward = evaluate_policy(model, Env(render=True, scenario='deadly_corridor_medium_multiple_variables'), n_eval_episodes=5)

print(f"mean_reward:{mean_reward:.2f}")
print(f"std_reward:{std_reward:.2f}")