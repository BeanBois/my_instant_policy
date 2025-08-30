from .game import Game, GameMode, Player
import numpy as np
import os 
from .game_configs import NUM_OF_EDIBLE_OBJECT, NUM_OF_OBSTACLES, BLACK, YELLOW, RED, GREEN
from collections import defaultdict

class GameInterface:

    agent_keypoints = Player(100,100).get_keypoints(frame='self')


    def __init__(self,num_edibles = NUM_OF_EDIBLE_OBJECT, num_obstacles = NUM_OF_OBSTACLES, mode = GameMode.DEMO_MODE, objective = None):
        # dont run headless
        if 'SDL_VIDEODRIVER' in os.environ:
            del os.environ['SDL_VIDEODRIVER']

        self.game = Game(num_edibles=num_edibles, num_obstacles=num_obstacles, objective=objective)
        self.running = True
        self.t = 0
        self.mode = mode
        self.observations = []

    def start_game(self):
        self.game.draw()
        obs = self.get_obs()
        self.observations.append(obs)
        self.game.clock.tick(60)

        return obs 
    
    def reset(self):
        self.game.restart_game()
        self.running = True
        self.t = 0
        self.observations = []

    def change_mode(self,mode):
        self.mode = mode

    def get_obs(self, track = True):
        agent_pos = self._get_agent_pos()
        agent_state = self.game.player.state # works now but maybe refactor
        # Since our 'point clouds' are represented as pixels in a 2d grid, our dense point cloud will be a 2d matrix of Screen-width x Screen-height
        raw_dense_point_clouds = self.game.get_screen_pixels()
        raw_coords = np.array([[(x,y) for y in range(self.game.screen_height) ]  for x in range(self.game.screen_width)])
        # To ensure that only objects point clouds are picked up, we remove all white pixels and agent pixels
        
        
        
        WHITE  = np.array([255,255,255])
        PURPLE = np.array([128,  0,128])  # player eating
        BLUE   = np.array([  0,  0,255])  # player not eating

        is_white  = np.all(raw_dense_point_clouds == WHITE,  axis=2)
        is_purple = np.all(raw_dense_point_clouds == PURPLE, axis=2)
        is_blue   = np.all(raw_dense_point_clouds == BLUE,   axis=2)

        mask = ~(is_white | is_purple | is_blue)
        valid_points = np.where(mask)
        
        coords = raw_coords[valid_points]
        obs =  {
            #segmented point clouds 
            'coords' : coords,
            # agent info
            'agent-pos' : agent_pos,
            'agent-state' : agent_state,
            'agent-orientation' : self.game.player.get_orientation('deg'),
            #game info
            'done': not self.running,
            'time' : self.t,
            'won' : self.game.game_won 
        }
        if track:
            self.observations.append(obs)
        return obs 

    def step(self,action = None):
        self.t += 1
        if self.mode == GameMode.DEMO_MODE:
            self.running = self.game.handle_events()
        else:
            assert action is not None
            self.running = self.game.handle_action(action) 
        self.game.update()
        self.game.draw()
        obs = self.get_obs()
        self.game.clock.tick(60)
        return obs 
    
    def set_initial_config(self,filename):
        self.game.load_config(filename)

    def save_config(self,filename):
        self.game.save_config(filename)

    def _get_agent_pos(self):
        _, front, back_left, back_right = [v for _, v in self.game.player.get_keypoints(frame = 'self').items()]
        center = self.game.player.get_pos()
        center = np.array(center)
        # player is a triangle so i want to capture the 3 edges of the triangle
        # at player_ori == 0 degree, edges == (tl, bl, (tr+br)//2)
        tri_points = np.array([front, back_left, back_right])
        ori_in_deg = self.game.player.get_orientation(mode='deg')
        R = self._rotation_matrix_2d(ori_in_deg) 

        # rotate around center

        rotated = (R @ tri_points.T).T
        final = rotated + center
        player_pos = np.vstack([center, final])
        return player_pos   
    
    def _rotation_matrix_2d(self, theta_deg):
        theta = theta_deg/180 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s],
                        [s,  c]])