import gymnasium as gym
import stable_baselines3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import torch


# Create directories to hold models and logs
model_dir = 'models'
log_dir = 'logs'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Set the training device
if torch.cuda.is_available(): 
    device = "cuda" 
else: 
    device = "cpu" 
device = torch.device(device)

def train(env, sb3_algo):
    model = sb3_class('MlpPolicy', env, verbose=1, device=device, tensorboard_log=log_dir)
        
    # Stop training when mean reward reaches reward_threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

    # Stop training when model shows no improvement after max_no_improvement_evals,
    # but do not start counting towards max_no_improvement_evals until after min_evals.
    # Number of timesteps before possibly stopping training = min_evals * eval_freq (below)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10000, verbose=1)

    eval_callback = EvalCallback(
        env, 
        eval_freq=10000, # how often to perform evaluation i.e. every 10000 timesteps.
        callback_on_new_best=callback_on_best,
        callback_after_eval=stop_train_callback,
        verbose=1,
        best_model_save_path=os.path.join(model_dir, f'{args.gymenv}_{args.sb3_algo}')
    )

    """
    Total_timesteps: pass in a very large number to train (almost) indefinitely.
    tb_log_name: create log files with the name [gym env name]_[sb3 algorithm] i.e. Pendulum_v1_SAC
    callback: pass in reference to a callback function above
    """


    TIMESTEPS = int(1e10)

    model.learn(total_timesteps=TIMESTEPS, tb_log_name=f'{args.gymenv}_{args.sb3_algo}', callback=eval_callback)


def test(env):
    model = sb3_class.load(os.path.join(model_dir, f'{args.gymenv}_{args.sb3_algo}', 'best_model') env=env)
    obs = env.reset()[0]

    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break




if __name__ == '__main__':
    # Command to train the model
    # python "Mujoko environment.py" Pendulum-v1 A2C -t

    # Command to show the performance in tensorboard
    # tensorboard --logdir logs

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Pendulum-v1')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, DQN, PPO, SAC, TD3')
    parser.add_argument('--test', help='Test mode', metavar='path_to_model')
    args = parser.parse_args()

    # Dynamic way to import algorithm. For example, passing in DQN is equivalent to hardcoding:
    # from stable_baseline3 import DQN
    sb3_class = getattr(stable_baselines3, args.sb3_algo)


    # Command to test the model for training
    # python "Automatically stop training.py" BipedalWalker-v3 SAC
    if args.test:
        env = gym.make(args.gymenv, render_mode='human')
        test()
    else:
        env = gym.make(args.gymenv)
        env = Monitor(env)
        train()
