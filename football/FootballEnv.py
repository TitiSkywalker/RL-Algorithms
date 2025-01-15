"""
This file implements the wrapper for football environment.
"""

import gfootball.env as football_env
import random
import math
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import gfootball.env as football_env
import cv2

from scipy.ndimage import gaussian_filter

'''
Available environments:
- 11_vs_11_stochastic
- 11_vs_11_easy_stochastic
- 11_vs_11_hard_stochastic
- ...
https://github.com/google-research/football/blob/master/gfootball/doc/scenarios.md

Observation space & action space:
https://github.com/google-research/football/blob/master/gfootball/doc/observation.md
'''

# Football environment used for PPO
class FootballSingleEnv:
    def __init__(self, num_frames = 4, skip_frames = 4, evaluate = False):
        self.num_frames = num_frames
        self.skip_frames = skip_frames      # skip 1 frames
        self.evaluate = evaluate

        self.parameter_stack = []
        self.minimap_stack = []

        self.prev_ball = (0, 0, 0)

        self.score_board = (0, 0)

        self.env = football_env.create_environment(env_name = "11_vs_11_easy_stochastic", representation = "raw", render = evaluate)

        self.record_frames = []

    def reset(self):
        self.score_board = (0, 0)
        self.parameter_stack = []
        self.minimap_stack = []
        self.record_frames = []

        init_state = self.env.reset()[0]

        parameters, minimap = self.process_single_state(init_state)

        if self.evaluate:
            self.record_frames.append(self.env.render("rgb_array"))

        for _ in range(self.num_frames):
            self.parameter_stack.append(parameters)
            self.minimap_stack.append(minimap)
        
        self.prev_ball = init_state["ball"]

        return torch.stack(self.parameter_stack), torch.stack(self.minimap_stack)

    def step(self, action) -> tuple[torch.Tensor, torch.Tensor, float, bool, dict]:
        next_state, reward, terminated, info = self.env.step(action)
        next_state = next_state[0]

        reward = self.calculate_reward(self.prev_ball, next_state["ball"], info, next_state["ball_owned_team"])

        if self.evaluate:
            self.record_frames.append(self.env.render("rgb_array"))

        # update score board
        if info["score_reward"] == 1:
            print(next_state["score"])
            self.score_board = (self.score_board[0]+1, self.score_board[1])
            self.prev_ball = (0, 0, 0)
        elif info["score_reward"] == -1:
            print(next_state["score"])
            self.score_board = (self.score_board[0], self.score_board[1]+1)
            self.prev_ball = (0, 0, 0)
        else:
            self.prev_ball = next_state["ball"]            

        # skip frames and cumulate rewards
        for _ in range(self.skip_frames):
            if terminated:
                break
            extra_state, extra_reward, extra_terminated, extra_info = self.env.step(action)
            extra_state = extra_state[0]

            if self.evaluate:
                self.record_frames.append(self.env.render("rgb_array"))

            terminated = (terminated or extra_terminated)
            info = extra_info

            reward += self.calculate_reward(self.prev_ball, extra_state["ball"], extra_info, extra_state["ball_owned_team"])

            # update score board
            if extra_info["score_reward"] == 1:
                print(extra_state["score"])
                self.score_board = (self.score_board[0]+1, self.score_board[1])
                self.prev_ball = (0, 0, 0)
            elif extra_info["score_reward"] == -1:
                print(extra_state["score"])
                self.score_board = (self.score_board[0], self.score_board[1]+1)
                self.prev_ball = (0, 0, 0)
            else:
                self.prev_ball = extra_state["ball"]
            
            next_state = extra_state
        
        parameters, minimap = self.process_single_state(next_state)

        self.parameter_stack.pop(0)
        self.parameter_stack.append(parameters)
        self.minimap_stack.pop(0)
        self.minimap_stack.append(minimap)

        if terminated:
            # default behavior after gameover: reset
            self.reset()
        
        return torch.stack(self.parameter_stack), torch.stack(self.minimap_stack), reward, terminated, info

    def get_score_board(self) -> list[tuple]:
        return self.score_board

    # the function should be based on state transition
    def calculate_reward(self, ball1, ball2, info, ball_owned_team) -> float:
        x1, y1, z1 = ball1
        # potential1 = 4/((x1-1)**2+y1**2+0.2)-4/((x1+1)**2+y1**2+0.2)
        # potential1 = 0.5/(((x1-1)**2+y1**2)+0.05)-1/((x1+1)**2+y1**2+0.1)
        # potential1 = 10*math.exp(-100*((x1-1)**2+y1**2))-1/((x1+1)**2+y1**2+0.1)

        potential1 = 50-50*math.sqrt((x1-1)**2+y1**2)

        x2, y2, z2 = ball2
        # potential2 = 4/((x2-1)**2+y2**2+0.2)-4/((x2+1)**2+y2**2+0.2)
        # potential2 = 0.5/(((x1-1)**2+y1**2)+0.05)-1/((x2+1)**2+y2**2+0.1)
        # potential2 = 10*math.exp(-100*((x2-1)**2+y2**2))-1/((x2+1)**2+y2**2+0.1)

        potential2 = 50-50*math.sqrt((x2-1)**2+y2**2)

        reward = potential2 - potential1

        if info["score_reward"] == 1:       # left team goal
            print("left goal")
            reward += 100
        elif info["score_reward"] == -1:    # right team goal
            print("right goal")
            reward -= 200

        if ball_owned_team == 0:
            # controlling ball is good
            reward += 0.2
        
        # if control == 0:            # left team in control
        #     reward += 0.1
        # elif control == 1:          # right team in control
        #     reward -= 0.1

        return reward
    
    # generate a binary super minimap for players and football
    def generate_minimap(self, points) -> torch.Tensor:
        width, height = 96, 72

        scaled_points = np.clip(((points + np.array([1, 0.42])) / np.array([2, 0.84])) * np.array([width, height]), 0, [width - 1, height - 1]).astype(int)

        grid = np.zeros((height, width), dtype=int)

        for point in scaled_points:
            x, y = point
            grid[y, x] = 1

        smooth_grid = gaussian_filter(grid.astype(float), sigma=1)
        # Increase brightness by scaling the grid values
        brightness_factor = 7.0  # Increase for more brightness
        smooth_grid = smooth_grid * brightness_factor

        # Clip values to ensure they remain in the range [0, 1]
        smooth_grid = np.clip(smooth_grid, 0, 1)

        # Display the minimap
        # plt.imshow(smooth_grid, cmap='gray', origin='upper')
        # plt.title("Super Minimap with Larger Light Spots")
        # plt.axis('off')  # Hide axis for cleaner visualization
        # plt.show()
        # exit(0)

        return torch.Tensor(grid)

    def process_single_state(self, raw_state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # concat parameters
        parameters = []

        # ball information
        parameters.append(torch.Tensor(raw_state["ball"]))
        parameters.append(torch.Tensor(raw_state["ball_direction"]))
        parameters.append(torch.Tensor(raw_state["ball_rotation"]))
        one_hot = torch.zeros(3)
        one_hot[int(raw_state["ball_owned_team"])+1] = 1            # one-hot encoding
        parameters.append(one_hot)

        # ball owned player information
        one_hot = torch.zeros(22)
        if raw_state["ball_owned_team"] == 0:
            one_hot[int(raw_state["ball_owned_player"])] = 1        # one-hot encoding
        elif raw_state["ball_owned_team"] == 1:
            one_hot[int(raw_state["ball_owned_player"])+11] = 1     # one-hot encoding
        parameters.append(one_hot)
        if raw_state["ball_owned_team"] == -1:              
            parameters.append(torch.zeros(2))
            parameters.append(torch.zeros(2))
        elif raw_state["ball_owned_team"] == 0:             
            parameters.append(torch.Tensor(raw_state["left_team"][raw_state["ball_owned_player"]]))
            parameters.append(torch.Tensor(raw_state["left_team_direction"][raw_state["ball_owned_player"]]))
        else:                                               
            parameters.append(torch.Tensor(raw_state["right_team"][raw_state["ball_owned_player"]]))
            parameters.append(torch.Tensor(raw_state["right_team_direction"][raw_state["ball_owned_player"]]))
        
        # left team information
        parameters.append(torch.Tensor(raw_state["left_team"]).flatten())
        parameters.append(torch.Tensor(raw_state["left_team_direction"]).flatten())
        parameters.append(torch.Tensor(raw_state["left_team_tired_factor"]))
        parameters.append(torch.Tensor(raw_state["left_team_active"]).to(int))
        # right team information
        parameters.append(torch.Tensor(raw_state["right_team"]).flatten())
        parameters.append(torch.Tensor(raw_state["right_team_direction"]).flatten())
        parameters.append(torch.Tensor(raw_state["right_team_tired_factor"]))
        parameters.append(torch.Tensor(raw_state["right_team_active"]).to(int))

        # active player information
        active_player = raw_state["active"]
        one_hot = torch.zeros(11)
        one_hot[raw_state["active"]] = 1
        parameters.append(one_hot)
        parameters.append(torch.Tensor(raw_state["left_team"][active_player]))
        parameters.append(torch.Tensor(raw_state["left_team_direction"][active_player]))

        parameters = torch.cat(parameters, dim=0)

        # generate minimap
        minimap = []
        minimap.append(self.generate_minimap(raw_state["left_team"]))
        minimap.append(self.generate_minimap(raw_state["right_team"]))
        minimap.append(self.generate_minimap([raw_state["ball"][:2]]))
        minimap.append(self.generate_minimap([raw_state["left_team"][raw_state["active"]]]))
        minimap = torch.stack(minimap)

        return parameters, minimap
    
    def close(self):
        self.env.close()
    
    def save_video(self):
        print(f"Save video, {len(self.record_frames)} frames")
        output_file = "../results/football.mp4"
        frame_rate = 20

        video_writer = cv2.VideoWriter(output_file, -1, frame_rate, (1280, 720))

        for img in self.record_frames:
            video_writer.write(img)

        # Release the VideoWriter
        video_writer.release()
