from vizdoom import *
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2
from scipy.stats import norm


class VizDoomGym(Env):
    def __init__(self, scenario, render=True):

        self.game = DoomGame()
        self.game.load_config(f'{scenario}.cfg')

        self.game.set_window_visible(render)
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(7)

        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 26

    def step(self, action):
        actions = np.identity(7)
        movement_reward = self.game.make_action(actions[action], 4)

        reward = 0
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)

            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo, CAMERA_ANGLE = game_variables
            #change to damage count

            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            if 90 <= CAMERA_ANGLE <= 270:
                camera_reward = -norm.pdf(CAMERA_ANGLE, 180, 28)*10000
            else:
                camera_reward = 0

            reward = movement_reward + damage_taken_delta*10 + hitcount_delta*210 + ammo_delta*5 + camera_reward
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        terminated = self.game.is_episode_finished()

        truncated = self.game.is_player_dead() or self.game.is_player_dead() or self.game.is_player_dead()

        return state, reward, terminated, truncated, info

    def reset(self, seed=0):
        self.game.new_episode()

        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 26

        state = self.game.get_state().screen_buffer

        if self.game.get_state():
            info = self.game.get_state().game_variables[0]  # ammo
        else:
            info = 0

        return (self.grayscale(state), {'ammo': info})

    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    def close(self):
        self.game.close()