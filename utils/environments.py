from vizdoom import *
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2


class VizDoomGym(Env): 
    def __init__(self, scenario, render=True, number_of_actions=3):

        self.game = DoomGame()
        self.game.load_config(f'{scenario}.cfg')
        
        self.game.set_window_visible(render)
        self.game.init()
        
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        self.number_of_actions = number_of_actions
        self.action_space = Discrete(number_of_actions)
        
    def step(self, action):
        actions = np.identity(self.number_of_actions)
        reward = self.game.make_action(actions[action], 4) 
        
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            info = self.game.get_state().game_variables[0] # ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        terminated = self.game.is_episode_finished()

        truncated = self.game.is_player_dead() or self.game.is_player_dead() or self.game.is_player_dead()
        
        return state, reward, terminated, truncated, info 
        
    def reset(self, seed=0): 
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        
        if self.game.get_state():
            info = self.game.get_state().game_variables[0] # ammo
        else:
            info = 0

        return (self.grayscale(state), {'ammo': info})
    

    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
    
    def close(self): 
        self.game.close()