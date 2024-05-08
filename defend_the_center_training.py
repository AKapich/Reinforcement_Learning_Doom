from Environments import VizDoomGym as Env
from callbacks import TrainAndLoggingCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO


CHECKPOINT_DIR = './training/train/train_defend_the_center'
LOG_DIR = './training/logs/log_defend_the_center'


callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# train
env = Env(render=False, scenario='defend_the_center')
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=4096)
model.learn(total_timesteps=150000, callback=callback)

# test
model = PPO.load(CHECKPOINT_DIR + '/best_model_150000', env=env)
mean_reward, std_reward = evaluate_policy(model, Env(render=True, scenario='defend_the_center'), n_eval_episodes=10)

print(f"mean_reward:{mean_reward:.2f}")
print(f"std_reward:{std_reward:.2f}")