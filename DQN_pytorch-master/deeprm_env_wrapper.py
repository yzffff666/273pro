import numpy as np
import sys
import os
import gym
from gym.spaces import Box, Discrete

# Ensure deeprm-master is in Python path
sys.path.append(os.path.abspath('../deeprm-master'))

from environment import Env  # DeepRM environment
from parameters import Parameters  # Simulation parameter definitions


class DeepRMWrapper(gym.Env):
    def __init__(self):
        # Load parameters
        self.pa = Parameters()

        # Initialize the DeepRM environment
        self.env = Env(self.pa, render=False, repre='image', end='no_new_job')
        self.env.reset()

        # Observation state: typically image (convert to 1D)
        self.state_shape = self.env.observe().shape
        self.state_dim = int(np.prod(self.state_shape))

        # Define Gym-style spaces
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Action space: job slots + 1 void action
        self.action_dim = self.env.get_action_space()
        self.action_space = Discrete(self.action_dim)

        self.done = False
        self.total_steps = 0

    def reset(self):
        self.done = False
        self.total_steps = 0
        self.env.reset()
        return self._get_state()

    def step(self, action):
        full_result = self.env.step(action)
        #print("DEBUG - Raw env.step() result:", full_result, "Type:", type(full_result))

        if isinstance(full_result, tuple) and len(full_result) == 4:
            reward = full_result[1]
        else:
            reward = full_result  # fallback

        self.done = self.env.is_done()
        obs = self._get_state()
        info = {}
        return obs, reward, self.done, info

    def _get_state(self):
        state = self.env.observe()
        return state.flatten().astype(np.float32)

    def close(self):
        pass