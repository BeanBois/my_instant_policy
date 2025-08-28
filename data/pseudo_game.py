from .pseudo_game_aux import *
import os 
import math
import random 
from typing import List
import matplotlib.pyplot as plt

def is_obstacle_blocking(player_pos, goal_center, obst_center, obst_width, obst_height):
    # Vector from player to goal
    to_goal = goal_center - player_pos
    to_goal_normalized = to_goal / np.linalg.norm(to_goal)
    
    # Vector from player to obstacle center
    to_obstacle = obst_center - player_pos
    
    # Project obstacle center onto the line from player to goal
    projection_length = np.dot(to_obstacle, to_goal_normalized)
    
    # If projection is negative or beyond goal, obstacle is not in the way
    if projection_length < 0 or projection_length > np.linalg.norm(to_goal):
        return False, None
    
    # Find the closest point on the line to the obstacle center
    closest_point = player_pos + projection_length * to_goal_normalized
    
    # Calculate distance from obstacle center to the line
    distance_to_line = np.linalg.norm(obst_center - closest_point)
    
    # Check if obstacle intersects with the path (considering obstacle dimensions)
    obstacle_radius = max(obst_width, obst_height) / 2
    if distance_to_line < obstacle_radius:
        return True, closest_point
    
    return False, None

def _interpolate_polyline(wps_xy: np.ndarray, step_px: float) -> np.ndarray:
    """Return dense (M,2) path by linear interpolation with ~step_px spacing."""
    out = []
    for a, b in zip(wps_xy[:-1], wps_xy[1:]):
        v = b - a
        d = float(np.hypot(v[0], v[1]))
        if d < 1e-9:    # duplicate point
            continue
        n = int(np.ceil(d / max(1e-6, step_px)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)  # exclude b; next segment will include its start
        seg = a[None, :] + t[:, None] * v[None, :]
        out.append(seg)
    out.append(wps_xy[-1][None, :])  # include last point exactly
    path = np.vstack(out)
    # round to integer grid if you keep an integer world
    path = np.rint(path).astype(np.int32)
    # dedupe consecutive duplicates caused by rounding
    keep = np.ones(len(path), dtype=bool)
    keep[1:] = np.any(path[1:] != path[:-1], axis=1)
    return path[keep]

def build_interpolated_plan(waypoints: np.ndarray, step_px: float):
    """
    waypoints: (N,3) [x,y,state_value]. Returns (M,3) interpolated [x,y,state_value].
    State changes ‘take effect’ once the agent REACHES each original waypoint.
    """
    wps_xy = waypoints[:, :2].astype(float)
    wps_state = waypoints[:, 2].astype(int)

    path_xy = _interpolate_polyline(wps_xy, step_px)

    # Map each original waypoint index into the flattened path index
    # by re-running interpolation but recording segment lengths.
    idx_map = [0]
    for a, b in zip(wps_xy[:-1], wps_xy[1:]):
        d = float(np.hypot(*(b - a)))
        n = int(np.ceil(d / max(1e-6, step_px)))
        idx_map.append(idx_map[-1] + max(1, n))  # segment contributed n points (without endpoint), next starts there
    idx_map[-1] = len(path_xy) - 1  # ensure last maps to the last index exactly

    # Build state timeline: hold previous, apply change from waypoint i at its mapped index
    state_line = np.empty(len(path_xy), dtype=int)
    cur_state = wps_state[0] if len(wps_state) else 0
    state_line[:] = cur_state
    for i in range(1, len(wps_state)):
        cur_state = wps_state[i]
        state_line[idx_map[i]:] = cur_state

    plan = np.column_stack([path_xy, state_line])
    return plan.astype(np.int32)


class PseudoGame:
    agent_keypoints = PseudoPlayer(100,100).get_keypoints(frame='self')
    
    def __init__(self, 
                 num_objects = NUM_OF_OBJECTS,
                 max_num_sampled_waypoints = MAX_NUM_SAMPLED_WAYPOINTS, 
                 min_num_sampled_waypoints = MIN_NUM_SAMPLED_WAYPOINTS, 
                 biased = DEFAULT_NOT_BIASED, 
                 augmented = DEFAULT_NOT_AUGMENTED, 
                 screen_width = SCREEN_WIDTH, 
                 screen_height = SCREEN_HEIGHT, 
                 player_starting_pos = (PLAYER_START_X, PLAYER_START_Y),
                 MAX_LENGTH = 100):
        
        # setup configs for pseudo game
        # init game
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = np.zeros([self.screen_width, self.screen_height, 3])
        self.max_length = MAX_LENGTH
        self.min_num_sampled_waypoints = min_num_sampled_waypoints
        self.max_num_sampled_waypoints = max_num_sampled_waypoints
        self.t = 0
        self.game_config = None 

        # setup game by creating relevant object
        self.objects = []
        self.num_objects = num_objects
        self._populate_pseudo_game()

        self.num_waypoints_used = np.random.randint(self.min_num_sampled_waypoints, self.max_num_sampled_waypoints)
        self.done = False
        self.biased = biased
        self.augmented = augmented
        self._wp_offset = 0
        self.waypoints = self._sample_waypoints()
        self._plan_idx = 0 
        self.plan = build_interpolated_plan(self.waypoints, MAX_FORWARD_DIST)
        
        self.observations = []
        self.actions : List[Action] = [] 
    
    # this function does a run of the pseudogame and returns the observations in pointclouds 
    # it will be a list of 'frames' at each timepoints
    def reset_game(self,shuffle = True):
        self.t = 0
        self.done = False

        self.objects = []
        self._populate_pseudo_game()

        self.waypoints = []
        self.num_waypoints_used = np.random.randint(self.min_num_sampled_waypoints, self.max_num_sampled_waypoints)
        self.observations = []
        self.actions = []
        self._wp_offset = 0
        self._plan_idx = 0
        self.waypoints = self._sample_waypoints()
        self.draw() # to update screen to reflect object location

    def set_augmented(self, _augmented):
        self.augmented = _augmented

    def run(self):
        # self.draw()
        # obs = self.get_obs()
        # self.observations.append(obs)
        _temp_for_act = 0
        while not self.done and self.t < self.max_length: 
            self.draw()
            obs = self.get_obs()
            self.observations.append(obs)
            self.go_to_next_waypoint()
            while _temp_for_act == len(self.actions) and not self.done: # ensures that an action is taken 
                self.go_to_next_waypoint()
            _temp_for_act = len(self.actions)
            self.t += 1
            self.update()

        # add last observation
        # len(self.observations) == len(actions) + 1 here only when self.waypoint loops till self.done = True
        if len(self.observations) == len(self.actions):
            self.draw()
            obs = self.get_obs()
            self.observations.append(obs)
        self._end_game()
    
    def get_player_keypoints(self, frame = 'self'):
        return self.player.get_keypoints(frame=frame)

    def get_actions(self, mode = 'object', angle_unit = 'deg'):
        if mode == 'vector':
            return [action.as_vector(mode=angle_unit) for action in self.actions]
        elif mode =='se2':
            return [action.to_SE2() for action in self.actions]

        return self.actions

    def get_obs(self):
        agent_pos = self._get_agent_pos()
        agent_state = self.player.state

        # Since our 'point clouds' are represented as pixels in a 2d grid, our dense point cloud will be a 2d matrix of Screen-width x Screen-height
        raw_dense_point_clouds = self._get_screen_pixels()
        raw_coords = np.array([[(x,y) for y in range(self.screen_height) ]  for x in range(self.screen_width)])
        
        WHITE  = np.array([255,255,255])
        PURPLE = np.array([128,  0,128])  # player eating
        BLUE   = np.array([  0,  0,255])  # player not eating

        is_white  = np.all(raw_dense_point_clouds == WHITE,  axis=2)
        is_purple = np.all(raw_dense_point_clouds == PURPLE, axis=2)
        is_blue   = np.all(raw_dense_point_clouds == BLUE,   axis=2)

        mask = ~(is_white | is_purple | is_blue)
        valid_points = np.where(mask)
        coords = raw_coords[valid_points]

        return {
            'coords' : coords,
            'agent-pos' : agent_pos,
            'agent-state' : agent_state,
            'agent-orientation' : self.player.get_orientation('deg'),
            'done' : self.t >= self.max_length or self.done , # wrong
            'time' : self.t
        }

    def go_to_next_waypoint(self):
        idx = self._plan_idx
        if idx >= len(self.plan):
            self.done = True
            return

        target_row = self.plan[idx]
        target = target_row[:2].astype(float)
        state_val = int(target_row[2])
        state_change = next((ps for ps in PlayerState if ps.value == state_val), PlayerState.NOT_EATING)

        px, py = map(float, self.player.get_pos())
        theta_deg = float(self.player.get_orientation(mode='deg'))  # 0=East, +CCW

        dx, dy = target[0] - px, target[1] - py
        dist = int(np.hypot(dx, dy))

        WAYPOINT_THRESHOLD = max(1.0, float(PLAYER_SIZE) * 1)
        # DEG_THRESHOLD = 5 # deg
        HEADING_DEADBAND = 90.0  # deg
        if dist <= WAYPOINT_THRESHOLD:
            self._plan_idx = idx + 1
            if self._plan_idx >= len(self.plan):
                self.done = True
            return

        goal_bearing = float(np.degrees(np.arctan2(dy, dx)))
        err = goal_bearing - theta_deg
        while err > 180.0:  err -= 360.0
        while err < -180.0: err += 360.0
        if abs(err) > HEADING_DEADBAND:
            rotation_cmd = float(np.clip(err, -MAX_ROTATION, MAX_ROTATION))
            action = Action(forward_movement=0.0, rotation_deg=rotation_cmd, state_change=state_change)
            self.actions.append(action)
            self.player.move_with_action(action)
            return

        align = max(0.0, np.cos(np.radians(err)))
        forward_cmd = float(min(dist, MAX_FORWARD_DIST) * align)
        rotation_cmd = float(np.clip(err, -MAX_ROTATION, MAX_ROTATION))
        action = Action(forward_movement=forward_cmd, rotation_deg=rotation_cmd, state_change=state_change)
        self.actions.append(action)
        self.player.move_with_action(action)

        px2, py2 = map(float, self.player.get_pos())
        if int(np.hypot(target[0] - px2, target[1] - py2)) <= WAYPOINT_THRESHOLD:
            self._plan_idx = idx + 1
            if self._plan_idx >= len(self.plan):
                self.done = True

    def update(self):
        # dont need to update since syntatical logic not important
        # player_rect = self.player.get_rect()
        # object_rect = self.object.get_rect()
        # if player_rect.colliderect(object_rect) and type(self.object) == EdibleObject and self.player.state is PlayerState.EATING:
        #     self.object.eaten = True 
        return 
    
    def draw(self, plot = False):
        self.screen[:,:] = WHITE

        self.screen = self.player.draw(self.screen)
        for obj in self.objects:
            self.screen = obj.draw(self.screen)

        if plot:
            for waypoint in self.waypoints:
                center = waypoint[:-1]
                for i in range(-2,2):
                    for j in range(-2,2):
                        self.screen[int(center[1]) + i, int(center[0]) + j] = [0,0,0]
            self.plot_screen()
    
    def plot_screen(self):
        # funciton is wrong oof
        W, H = self.screen.shape[:2]
        plt.imshow(
            self.screen,
            origin="lower",          # y increases upward
            extent=[0, W, 0, H],     # x: 0→W, y: 0→H in math coords
        )
        # plt.gca().set_aspect("equal")
        plt.show()

    def _get_agent_pos(self):
        _, front, back_left, back_right = [v for _, v in self.player.get_keypoints(frame = 'self').items()]
        center = self.player.get_pos()
        center = np.array(center)
        # player is a triangle so i want to capture the 3 edges of the triangle
        # at player_ori == 0 degree, edges == (tl, bl, (tr+br)//2)
        tri_points = np.array([front, back_left, back_right])
        ori_in_deg = self.player.get_orientation(mode='deg')
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
    
    def _populate_pseudo_game(self):
        player_starting_pos_x =  np.random.randint(0 + PseudoPlayer.width ,self.screen_width - PseudoPlayer.width) 
        player_starting_pos_y =  np.random.randint(0 + PseudoPlayer.height,self.screen_height - PseudoPlayer.height) 
        self.player = PseudoPlayer(player_starting_pos_x, player_starting_pos_y)
        OBJECT_SPAWN_OFFSET = 50
        if self.game_config is None:
            self.game_config = {}
            self.game_config['objects'] = []
            for _ in range(self.num_objects):
                # generate random starting position and choose a random object
                start_position_x = np.random.randint(OBJECT_SPAWN_OFFSET,self.screen_width - OBJECT_SPAWN_OFFSET) 
                start_position_y = np.random.randint(OBJECT_SPAWN_OFFSET, self.screen_height - OBJECT_SPAWN_OFFSET) 
                object_class = random.choice(AVAILABLE_OBJECTS)
                object = object_class(start_position_x, start_position_y)
                self.objects.append(object)
                self.game_config['objects'].append(object_class)
        else:
            for object_class in self.game_config['objects']:
                start_position_x = np.random.randint(OBJECT_SPAWN_OFFSET,self.screen_width - OBJECT_SPAWN_OFFSET) 
                start_position_y = np.random.randint(OBJECT_SPAWN_OFFSET, self.screen_height - OBJECT_SPAWN_OFFSET) 
                object = object_class(start_position_x, start_position_y)
                self.objects.append(object)

    def _sample_waypoints(self):

        waypoints_with_player_state = None 
        waypoints = []
        if not self.biased:
            waypoints = []

            # random 
            # error here i believe for no  biased not augmented keeps adding 0,0,0 into waypoints
            for _ in range(self.num_waypoints_used):
                obj = random.choice(self.objects)
                # see if sample on object or around object
                mode = 'on' if random.random() < 0.5 else 'arnd'
                tl, tr, br, bl, center = [v for _, v in obj.get_keypoints(frame='world').items()]
                width = abs(br[0]-bl[0])
                height = abs(br[1]-tr[1])
                
                if mode == 'on':
                    # sample on object
                    sample_x = random.randint(0,width)
                    sample_y = random.randint(0,height)
                    waypoint = np.array((sample_x + center[0], sample_y + center[1]))
                    waypoints.append(waypoint)
                else:
                    # sample around object 
                    hyp = math.ceil(math.sqrt(width**2 + height**2))
                    radius = random.randint(1 + hyp, MAX_SAMPLING_RADIUS + hyp )
                    theta = (random.random() * 2 * math.pi) - math.pi # 
                    addition = np.array([(np.cos(theta), np.sin(theta))]) * radius
                    waypoint =  addition + np.array(center) 
                    waypoints.append(waypoint) 

            # then sort waypoints wrt to distance to agent
            player_pos = np.array(self.player.get_pos())
            waypoints = sorted(waypoints, key=lambda wp: np.linalg.norm(np.array(wp) - player_pos))
            waypoint_dim = waypoints[0].shape[0] if len(waypoints[0].shape) == 1 else waypoints[0].shape[1]
            waypoints_with_player_state = np.zeros((len(waypoints), waypoint_dim + 1))
            num_waypoints_to_alter_state = min(1, np.sum(np.random.random(self.num_waypoints_used) > 0.5))
            chosen_waypoints = np.random.choice(self.num_waypoints_used, size=num_waypoints_to_alter_state, replace=False) # randomnly chose {num_waypoints_to_alter_state} indexes from range(self.num_waypoints_used)
            player_state = PlayerState.NOT_EATING
            for idx, waypoint in enumerate(waypoints):
                if idx in chosen_waypoints:
                    if player_state is PlayerState.NOT_EATING:
                        player_state = PlayerState.EATING
                    else:
                        player_state = PlayerState.NOT_EATING
                waypoints_with_player_state[idx, :-1] = np.rint(np.array(waypoint, dtype=float)).astype(int)
                waypoints_with_player_state[idx, -1] = player_state.value

        else:
            object_types = [type(obj) for obj in self.objects]
            # if 2 food, waypoints to eat 2 food
            if object_types.count(PseudoEdibleObject) == 2:
                # plan waypoint (self.num_waypoints) such that waypoints lead player to the 2 edibles
                # since min waypoint == 2, can start by just assigining waypoints
                
                player_pos = np.array(self.player.get_pos()) 
                waypoints.append(player_pos)
                obj_centers = []

                for obj in self.objects:
                    center = np.array(obj.get_pos())
                    obj_centers.append(center)
                
                if np.linalg.norm(player_pos - obj_centers[0]) > np.linalg.norm(player_pos - obj_centers[1]):
                    waypoints.append(obj_centers[1])
                    waypoints.append(obj_centers[0])
                
                else:
                    waypoints += obj_centers 
                while len(waypoints) < self.num_waypoints_used + 1:
                    chosen_index = random.randint(0,len(waypoints) - 2)
                    # choose to add a waypoint between waypoints[chosen_index] waypoints[chosen_index + 1] 
                       # choose to add a waypoint between waypoints[chosen_index] and waypoints[chosen_index + 1]
                    point_a = np.array(waypoints[chosen_index])
                    point_b = np.array(waypoints[chosen_index + 1])
                    
                    # Create a waypoint halfway between the two points (or with some randomness)
                    midpoint = (point_a + point_b) / 2
                    
                    # add some randomness to avoid perfectly straight lines
                    noise = np.random.normal(BIAS_SAMPLING_NOISE_MEAN, BIAS_SAMPLING_NOISE_STD, 2)  # small random offset
                    midpoint += noise
                    
                    # Insert the new waypoint between the chosen points
                    waypoints.insert(chosen_index + 1, midpoint)
                waypoints = waypoints[1:] # remove initial waypoint
                waypoint_dim = waypoints[0].shape[0] if len(waypoints[0].shape) == 1 else waypoints[0].shape[1]
                waypoints_with_player_state = np.zeros((len(waypoints), waypoint_dim + 1))
                for idx, waypoint in enumerate(waypoints):
                    waypoints_with_player_state[idx, :-1] = waypoint
                    if np.any(waypoint == obj_centers):
                        waypoints_with_player_state[idx, -1] = PlayerState.EATING.value
                    else:
                        waypoints_with_player_state[idx, -1] = PlayerState.NOT_EATING.value
            # if 1 food, 1 goal, waypoint to eat food then go goal
            elif object_types.count(PseudoEdibleObject) == 1 and object_types.count(PseudoGoal) == 1:
                # plan waypoint (self.num_waypoints) such that waypoints lead player to the edible, then to the goal 
                # since min waypoint == 2, can start by just assigining waypoints 
                player_pos = np.array(self.player.get_pos())
                waypoints.append(player_pos)
                edible_center = None
                goal_center = None

                for obj in self.objects:
                    center = np.array(obj.get_pos())
                    if type(obj) is PseudoEdibleObject:
                        edible_center = center
                    else:
                        goal_center = center
                    
                waypoints.append(edible_center)
                waypoints.append(goal_center)
                while len(waypoints) < self.num_waypoints_used + 1:
                    chosen_index = random.randint(0,len(waypoints) - 2)
                    # choose to add a waypoint between waypoints[chosen_index] waypoints[chosen_index + 1] 
                       # choose to add a waypoint between waypoints[chosen_index] and waypoints[chosen_index + 1]
                    point_a = np.array(waypoints[chosen_index])
                    point_b = np.array(waypoints[chosen_index + 1])
                    
                    # Create a waypoint halfway between the two points (or with some randomness)
                    midpoint = (point_a + point_b) / 2
                    
                    # add some randomness to avoid perfectly straight lines
                    noise = np.random.normal(BIAS_SAMPLING_NOISE_MEAN, BIAS_SAMPLING_NOISE_STD, 2)  # small random offset
                    midpoint += noise
                    
                    # Insert the new waypoint between the chosen points
                    waypoints.insert(chosen_index + 1, midpoint)
                waypoints = waypoints[1:] # remove initial waypoint
                waypoint_dim = waypoints[0].shape[0] if len(waypoints[0].shape) == 1 else waypoints[0].shape[1]
                waypoints_with_player_state = np.zeros((len(waypoints), waypoint_dim + 1))
                for idx, waypoint in enumerate(waypoints):
                    waypoints_with_player_state[idx, :-1] = waypoint
                    if np.all(waypoint == edible_center):
                        waypoints_with_player_state[idx, -1] = PlayerState.EATING.value
                    else:
                        waypoints_with_player_state[idx, -1] = PlayerState.NOT_EATING.value
            # if 1 goal, 1 obstacle plan waypooints to avoid obstacle and reach goal 
            elif object_types.count(PseudoObstacle) == 1 and object_types.count(PseudoGoal) == 1:
                # plan waypoint (self.num_waypoints) such that waypoints lead player to the goal while avoiding
                # center of goal automatically waypoint,
                # only need to avoid obstacle if its 'in the way' such that path towards goal is blocked by obstacle
                # calculate blocking with play orientation, position of player, obstacle center and obstacle width and height
                # if blocking, find the normal at the point of intersection along the plane, and sample 1 waypoint along this normal by using a random.randint(MIN_OBSTACLE_AVOIDANCE_DIST,MAX_OBSTACLE_AVOIDANCE_DIST)
                # since the deviation can be either 'left' or 'right', make sure to choose the right side

                player_pos = np.array(self.player.get_pos())
                waypoints.append(player_pos)

                obst_center = None
                obst_width = None
                obst_height = None
                goal_center = None

                for obj in self.objects:
                    center = np.array(obj.get_pos())
                    if type(obj) is PseudoGoal:
                        goal_center = center
                    else:
                        obst_center = center
                        tl, tr, br, bl, _ = obj.get_keypoints().values()
                        obst_width = abs(br[0]-bl[0])
                        obst_height = abs(br[1]-tr[1])


                # Check if obstacle blocks the direct path to goal
                if obst_center is not None and goal_center is not None:
                    is_blocking, intersection_point = is_obstacle_blocking(player_pos, goal_center, obst_center, obst_width, obst_height)
                    
                    if is_blocking:
                        # Calculate avoidance waypoint
                        to_goal = goal_center - player_pos
                        to_goal_normalized = to_goal / np.linalg.norm(to_goal)
                        
                        # Create perpendicular vector (normal to the path)
                        perpendicular = np.array([-to_goal_normalized[1], to_goal_normalized[0]])
                        
                        # Determine which side to go around based on player orientation and obstacle position
                        to_obstacle = obst_center - player_pos
                        cross_product = np.cross(to_goal_normalized, to_obstacle)
                        
                        # Choose the side that's more aligned with avoiding the obstacle
                        if cross_product > 0:
                            # Obstacle is to the left, go right
                            avoidance_direction = -perpendicular
                        else:
                            # Obstacle is to the right, go left
                            avoidance_direction = perpendicular
                        
                        # Sample avoidance distance
                        avoidance_distance = random.randint(MIN_OBSTACLE_AVOIDANCE_DIST, MAX_OBSTACLE_AVOIDANCE_DIST)
                        
                        # Create avoidance waypoint
                        avoidance_waypoint = intersection_point + avoidance_direction * avoidance_distance
                        waypoints.append(avoidance_waypoint)

                # Always add the goal as the final waypoint
                waypoints.append(goal_center)

                # Fill remaining waypoints if needed (interpolate between existing waypoints)
                while len(waypoints) < self.num_waypoints_used + 1:
                    chosen_index = random.randint(0, len(waypoints) - 2)
                    point_a = np.array(waypoints[chosen_index])
                    point_b = np.array(waypoints[chosen_index + 1])
                    midpoint = (point_a + point_b) / 2
                    waypoints.insert(chosen_index + 1, midpoint)

                waypoints = waypoints[1:] # remove initial waypoint
                waypoint_dim = waypoints[0].shape[0] if len(waypoints[0].shape) == 1 else waypoints[0].shape[1]
                waypoints_with_player_state = np.zeros((len(waypoints), waypoint_dim + 1))
                for idx, waypoint in enumerate(waypoints):
                    waypoints_with_player_state[idx, :-1] = waypoint
                    waypoints_with_player_state[idx, -1] = PlayerState.NOT_EATING.value
            # if 1 edible and 1 obstacle, avoid obst and go to edible
            elif object_types.count(PseudoObstacle) == 1 and object_types.count(PseudoEdibleObject) == 1:

                player_pos = np.array(self.player.get_pos())
                waypoints.append(player_pos)

                obst_center = None
                obst_width = None
                obst_height = None
                goal_center = None

                for obj in self.objects:
                    center = np.array(obj.get_pos())
                    if type(obj) is PseudoEdibleObject:
                        goal_center = center
                    else:
                        obst_center = center
                        tl, tr, br, bl, _ = obj.get_keypoints().values()
                        obst_width = abs(br[0]-bl[0])
                        obst_height = abs(br[1]-tr[1])

                # Check if obstacle blocks the direct path to goal
                if obst_center is not None and goal_center is not None:
                    is_blocking, intersection_point = is_obstacle_blocking(player_pos, goal_center, obst_center, obst_width, obst_height)
                    
                    if is_blocking:
                        # Calculate avoidance waypoint
                        to_goal = goal_center - player_pos
                        to_goal_normalized = to_goal / np.linalg.norm(to_goal)
                        
                        # Create perpendicular vector (normal to the path)
                        perpendicular = np.array([-to_goal_normalized[1], to_goal_normalized[0]])
                        
                        # Determine which side to go around based on player orientation and obstacle position
                        to_obstacle = obst_center - player_pos
                        cross_product = np.cross(to_goal_normalized, to_obstacle)
                        
                        # Choose the side that's more aligned with avoiding the obstacle
                        if cross_product > 0:
                            # Obstacle is to the left, go right
                            avoidance_direction = -perpendicular
                        else:
                            # Obstacle is to the right, go left
                            avoidance_direction = perpendicular
                        
                        # Sample avoidance distance
                        avoidance_distance = random.randint(MIN_OBSTACLE_AVOIDANCE_DIST, MAX_OBSTACLE_AVOIDANCE_DIST)
                        
                        # Create avoidance waypoint
                        avoidance_waypoint = intersection_point + avoidance_direction * avoidance_distance
                        waypoints.append(avoidance_waypoint)

                # Always add the goal as the final waypoint
                waypoints.append(goal_center)

                # Fill remaining waypoints if needed (interpolate between existing waypoints)
                while len(waypoints) < self.num_waypoints_used + 1:
                    chosen_index = random.randint(0, len(waypoints) - 2)
                    point_a = np.array(waypoints[chosen_index])
                    point_b = np.array(waypoints[chosen_index + 1])

                    midpoint = (point_a + point_b) / 2
                    waypoints.insert(chosen_index + 1, midpoint)
                    
                waypoints = waypoints[1:] # remove initial waypoint
                waypoint_dim = waypoints[0].shape[0] if len(waypoints[0].shape) == 1 else waypoints[0].shape[1]
                waypoints_with_player_state = np.zeros((len(waypoints), waypoint_dim + 1))
                for idx, waypoint in enumerate(waypoints):
                    waypoints_with_player_state[idx, :-1] = waypoint
                    if np.all(waypoint == goal_center):
                        waypoints_with_player_state[idx, -1] = PlayerState.EATING.value
                    else:
                        waypoints_with_player_state[idx, -1] = PlayerState.NOT_EATING.value
            # other scenarious: (2 goal, 2 obstacle, ), focus on 1 particular object
            else:
                obj = random.choice(self.objects)
                if type(obj) is PseudoEdibleObject or type(obj) is PseudoGoal:
                    # waypoint to go to goal
                    # same with the 2 objects, but this time just plan to go towards one object
                    player_pos = np.array(self.player.get_pos())
                    waypoints.append(player_pos)
                    target_center = np.array(obj.get_pos())
                    waypoints.append(target_center)
                    while len(waypoints) < self.num_waypoints_used + 1:
                                        chosen_index = random.randint(0, len(waypoints) - 2)
                                        point_a = np.array(waypoints[chosen_index])
                                        point_b = np.array(waypoints[chosen_index + 1])
                                        midpoint = (point_a + point_b) / 2
                                        waypoints.insert(chosen_index + 1, midpoint)
                    waypoints = waypoints[1:]
                    waypoint_dim = waypoints[0].shape[0] if len(waypoints[0].shape) == 1 else waypoints[0].shape[1]
                    waypoints_with_player_state = np.zeros((len(waypoints), waypoint_dim + 1))
                    if type(obj) is PseudoEdibleObject:
                        for idx, waypoint in enumerate(waypoints):
                            waypoints_with_player_state[idx, :-1] = waypoint
                            if np.all(waypoint == target_center):
                                waypoints_with_player_state[idx, -1] = PlayerState.EATING.value
                            else:
                                waypoints_with_player_state[idx, -1] = PlayerState.NOT_EATING.value
                    else:
                        for idx, waypoint in enumerate(waypoints):
                            waypoints_with_player_state[idx, :-1] = waypoint
                            waypoints_with_player_state[idx, -1] = PlayerState.NOT_EATING.value

                    
                else: 
                    # waypoints to avoid obstacles, 
                    # since no direct goal, we use current orientation 
                    # if current direction path blocked by obj then we move away
                    player_pos = np.array(self.player.get_pos())
                    player_orientation = self.player.get_orientation(mode='rad')
                    waypoints.append(player_pos)
                    
                    # Create a forward direction vector based on player orientation
                    forward_direction = np.array([np.cos(player_orientation), np.sin(player_orientation)])
                    
                    # Project forward to see if obstacle blocks the path
                    obst_center = np.array(obj.get_pos())
                    tl, tr, br, bl, _ = obj.get_keypoints().values()
                    obst_width = abs(br[0]-bl[0])
                    obst_height = abs(br[1]-tr[1])
                    
                    # Create a forward projection point (some distance ahead)
                    forward_distance = max(obst_width, obst_height) * 3  # Look ahead 3x obstacle size
                    forward_point = player_pos + forward_direction * forward_distance
                    
                    # Check if obstacle blocks the forward path
                    is_blocking, intersection_point = is_obstacle_blocking(
                        player_pos, forward_point, obst_center, obst_width, obst_height
                    )
                    
                    if is_blocking:
                        # Move away from obstacle
                        perpendicular = np.array([-forward_direction[1], forward_direction[0]])
                        
                        # Determine which side to go (choose the side that moves away from obstacle center)
                        to_obstacle = obst_center - player_pos
                        cross_product = np.cross(forward_direction, to_obstacle)
                        
                        if cross_product > 0:
                            # Obstacle is to the left, go right
                            avoidance_direction = -perpendicular
                        else:
                            # Obstacle is to the right, go left
                            avoidance_direction = perpendicular
                        
                        # Create multiple waypoints to navigate around the obstacle
                        avoidance_distance = random.randint(MIN_OBSTACLE_AVOIDANCE_DIST, MAX_OBSTACLE_AVOIDANCE_DIST)
                        
                        # First waypoint: move to the side
                        side_waypoint = player_pos + avoidance_direction * avoidance_distance
                        waypoints.append(side_waypoint)
                        
                        # Second waypoint: move forward while maintaining distance from obstacle
                        forward_waypoint = side_waypoint + forward_direction * (obst_width + obst_height) / 2
                        waypoints.append(forward_waypoint)
                        
                        # Third waypoint: return to original heading
                        final_waypoint = forward_waypoint - avoidance_direction * (avoidance_distance / 2)
                        waypoints.append(final_waypoint)
                    else:
                        # No obstacle blocking, just move forward
                        forward_waypoint = player_pos + forward_direction * forward_distance
                        waypoints.append(forward_waypoint)

                    # process waypoints 
                    waypoint_dim = waypoints[0].shape[0] if len(waypoints[0].shape) == 1 else waypoints[0].shape[1]
                    waypoints_with_player_state = np.zeros((len(waypoints), waypoint_dim + 1))
                    for idx, waypoint in enumerate(waypoints):
                        waypoints_with_player_state[idx, :-1] = waypoint
                        waypoints_with_player_state[idx, -1] = PlayerState.NOT_EATING.value  

        if self.augmented:
            waypoints_with_player_state = self._augment_waypoints(waypoints_with_player_state)
        waypoints_with_player_state = np.rint(waypoints_with_player_state).astype(int)
        return waypoints_with_player_state
        
    def _augment_waypoints(self, waypoints_with_player_state):
        """
        Implement data augmentation as described in the paper:
        1. For 30% of trajectories: add local disturbances with corrective actions
        2. For 10% of data points: change gripper open-close state (eating state)
        """
        augmented_waypoints = waypoints_with_player_state.copy()
        
        # 1. Local disturbances for 30% of trajectories
        if np.random.random() < 0.3:
            augmented_waypoints = self._add_local_disturbances(augmented_waypoints)
        
        # 2. State changes for 10% of data points
        augmented_waypoints = self._add_state_changes(augmented_waypoints, change_probability=0.1)
        
        return augmented_waypoints

    def _add_local_disturbances(self, waypoints_with_player_state):
        """
        Add local disturbances to waypoints and insert corrective waypoints
        to bring the trajectory back to the reference path.
        """
        original_waypoints = waypoints_with_player_state.copy()
        augmented_waypoints = []
        

        
        for i in range(len(original_waypoints)):
            current_waypoint = original_waypoints[i].copy()
            
            # Decide whether to add disturbance to this waypoint (not every waypoint)
            if np.random.random() < 0.4:  # 40% chance per waypoint
                # Add the original waypoint first
                augmented_waypoints.append(current_waypoint)
                
                # Create disturbed waypoint
                disturbance = np.random.normal(0, DISTURBANCE_MAGNITUDE, 2)
                disturbed_position = current_waypoint[:-1] + disturbance
                
                # Clamp to screen boundaries
                disturbed_position[0] = np.clip(disturbed_position[0], 0, self.screen_width - 1)
                disturbed_position[1] = np.clip(disturbed_position[1], 0, self.screen_height - 1)
                
                # Create disturbed waypoint with same state
                disturbed_waypoint = np.zeros_like(current_waypoint)
                disturbed_waypoint[:-1] = disturbed_position
                disturbed_waypoint[-1] = current_waypoint[-1]  # Keep same player state
                
                augmented_waypoints.append(disturbed_waypoint)
                
                # Add correction waypoints to bring back to reference trajectory
                num_correction_steps = np.random.randint(MIN_CORRECTION_STEPS, MAX_CORRECTION_STEPS + 1)
                
                # Get the next reference waypoint for correction target
                next_reference_idx = min(i + 1, len(original_waypoints) - 1)
                target_position = original_waypoints[next_reference_idx][:-1]
                
                # Create intermediate correction waypoints
                for step in range(1, num_correction_steps + 1):
                    # Interpolate back towards the reference trajectory
                    alpha = step / num_correction_steps
                    correction_position = (1 - alpha) * disturbed_position + alpha * target_position
                    
                    # Create correction waypoint
                    correction_waypoint = np.zeros_like(current_waypoint)
                    correction_waypoint[:-1] = correction_position
                    correction_waypoint[-1] = current_waypoint[-1]  # Keep same player state
                    
                    augmented_waypoints.append(correction_waypoint)
            else:
                # No disturbance, just add the original waypoint
                augmented_waypoints.append(current_waypoint)
        
        return np.array(augmented_waypoints)

    def _add_state_changes(self, waypoints_with_player_state, change_probability=0.1):
        """
        Randomly change the eating state for a percentage of waypoints.
        This helps the policy learn to re-grasp/re-engage after state changes.
        """
        augmented_waypoints = waypoints_with_player_state.copy()
        
        for i in range(len(augmented_waypoints)):
            if np.random.random() < change_probability:
                current_state = augmented_waypoints[i, -1]
                
                # Flip the eating state
                if current_state == PlayerState.EATING.value:
                    augmented_waypoints[i, -1] = PlayerState.NOT_EATING.value
                else:
                    augmented_waypoints[i, -1] = PlayerState.EATING.value
        
        return augmented_waypoints
   
    def _get_screen_pixels(self):
        return self.screen
    
    def _end_game(self):
        return
    

