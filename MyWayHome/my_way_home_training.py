from utils.environments import VizDoomGym as Env
from utils.callbacks import TrainAndLoggingCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from my_way_home_env import MyWayHomeGym


CHECKPOINT_DIR = "../training/train/my_way_home"
LOG_DIR = "../training/logs/log_my_way_home"


callback = TrainAndLoggingCallback(check_freq=5000, save_path=CHECKPOINT_DIR)

# train
env = MyWayHomeGym(render=True, scenario="my_way_home")
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.0001,
    n_steps=2048,
)
model.learn(
    total_timesteps=100000,
    callback=callback,
)

# test
model = PPO.load(CHECKPOINT_DIR + "/best_model_100000", env=env)

mean_reward, std_reward = evaluate_policy(
    model, Env(render=True, scenario="my_way_home"), n_eval_episodes=10
)

print(f"mean_reward:{mean_reward:.2f}")
print(f"std_reward:{std_reward:.2f}")