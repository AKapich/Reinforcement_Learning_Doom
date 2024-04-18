from environments import BasicVizDoomGym as BaseEnv
from callbacks import TrainAndLoggingCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO


CHECKPOINT_DIR = './training/train/train_basic'
LOG_DIR = './training/logs/log_basic'


callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

# train
env = BaseEnv(render=False)
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=2048)
model.learn(total_timesteps=20000, callback=callback)

# test
model = PPO.load(CHECKPOINT_DIR + '/best_model_20000', env=env)
mean_reward, std_reward = evaluate_policy(model, BaseEnv(render=True), n_eval_episodes=10)

print(f"mean_reward:{mean_reward:.2f}")
print(f"std_reward:{std_reward:.2f}")