from vizdoom import *
from vizdoom import GameVariable
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2


class MyWayHomeGym(Env):
    def __init__(self, scenario, render=True, number_of_actions=3):
        self.game = DoomGame()
        self.game.load_config(f"{scenario}.cfg")

        # self.game.set_mode(Mode.SPECTATOR)  # spectator

        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.add_available_game_variable(GameVariable.POSITION_Z)

        self.pos = None

        self.game.set_window_visible(render)
        self.game.init()

        self.pos_history_length = 200
        self.position_history = [None] * self.pos_history_length
        self.i = 0

        # self.observation_space = Box(
        #     low=0, high=255, shape=(100, 160, 320), dtype=np.uint8
        # )
        self.observation_space = Box(
            low=0, high=255, shape=(100, 160, 1), dtype=np.uint8
        )
        self.number_of_actions = number_of_actions
        self.action_space = Discrete(number_of_actions)

    def step(self, action):
        actions = np.identity(self.number_of_actions)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            _, pos_x, pos_y, pos_z = self.game.get_state().game_variables
            pos = np.array([pos_x, pos_y, pos_z])

            cur_index = self.i % self.pos_history_length
            self.position_history[cur_index] = pos

            prev_pos = self.position_history[self.pos_history_length - cur_index - 1]

            same_place_penalty = None
            if np.array_equal(self.position_history[cur_index], prev_pos):
                same_place_penalty = -1
            else:
                same_place_penalty = (
                    -0.5
                    / np.sqrt(
                        np.sum((self.position_history[cur_index] - prev_pos) ** 2)
                    )
                    if prev_pos is not None
                    else 0
                )

            same_place_penalty = max(-1, same_place_penalty)

            reward += same_place_penalty

            self.i += 1

            movement_reward = 0
            if self.pos is not None:
                dist = np.sqrt(np.sum((pos - self.pos) ** 2))
                movement_reward = dist * 0.005
                reward += movement_reward

            self.pos = pos

            state = self.game.get_state().screen_buffer

            green_reward = self.get_green_reward(np.moveaxis(state, 0, -1))

            reward += green_reward
            print(movement_reward, green_reward, reward, same_place_penalty)

            state = self.grayscale(state)
            # state = self.grayscale(state)
            info = self.game.get_state().game_variables[0]  # ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        terminated = self.game.is_episode_finished()

        truncated = (
            self.game.is_player_dead()
            or self.game.is_player_dead()
            or self.game.is_player_dead()
        )

        return state, reward, terminated, truncated, info

    def reset(self, seed=0):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer

        if self.game.get_state():
            info = self.game.get_state().game_variables[0]  # ammo
        else:
            info = 0

        return (self.grayscale(state), {"ammo": info})

    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    def get_green_reward(self, observation):
        hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
        imask_green = mask_green > 0
        green = np.zeros_like(observation, np.uint8)
        green[imask_green] = observation[imask_green]
        cv2.imwrite("green.jpg", green)

        green_px_count = np.count_nonzero(green)

        # print("PX COUNT", green_px_count)

        if green_px_count > 800 and green_px_count < 3000:
            print("Vest visible!")
            return 0.3

        pw = 10**6
        return green_px_count / pw
        # if green_px_count < 115000:
        #     return green_px_count / pw
        # else:
        #     b = 115000 / pw
        #     a = -1 / pw * 0.4
        #     return a * (green_px_count - 115000) + b

    def close(self):
        self.game.close()
