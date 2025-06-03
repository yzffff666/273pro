import gym
import torch
import torch.optim as optim
import argparse

from model import DQN, Dueling_DQN
from learn import dqn_learning, OptimizerSpec
from utils.atari_wrappers import *
from utils.gym_setup import *
from utils.schedules import *
from deeprm_env_wrapper import DeepRMWrapper  # Your DeepRM adapter

# === Hyperparameters ===
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 1000000
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1)
LEARNING_STARTS = 50000

# === Learning Function ===
def atari_learn(env, env_id, num_timesteps, double_dqn, dueling_dqn, device):
    def stopping_criterion(env, t):
        return t >= num_timesteps

    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )

    dqn_learning(
        env=env,
        env_id=env_id,
        q_func=Dueling_DQN if dueling_dqn else DQN,
        optimizer_spec=optimizer,
        exploration=EXPLORATION_SCHEDULE,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ,
        double_dqn=double_dqn,
        dueling_dqn=dueling_dqn,
        device=device  # ✅ Pass device correctly
    )

    env.close()

# === Main Entrypoint ===
def main():
    parser = argparse.ArgumentParser(description='Train an RL agent on DeepRM')
    parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to use (e.g., 0)")
    parser.add_argument("--double-dqn", type=int, default=0, help="Use Double DQN (1 = Yes)")
    parser.add_argument("--dueling-dqn", type=int, default=0, help="Use Dueling DQN (1 = Yes)")
    args = parser.parse_args()

    # === GPU Setup ===
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for training:", device)
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # === Agent settings ===
    double_dqn = (args.double_dqn == 1)
    dueling_dqn = (args.dueling_dqn == 1)

    # === Environment ===
    env = DeepRMWrapper()
    env_id = "DeepRM"
    max_timesteps = 5000  # You can increase this later

    print("Training on DeepRM, double_dqn %d, dueling_dqn %d" % (double_dqn, dueling_dqn))
    atari_learn(env, env_id, num_timesteps=max_timesteps,
                double_dqn=double_dqn, dueling_dqn=dueling_dqn, device=device)

if __name__ == '__main__':
    main()