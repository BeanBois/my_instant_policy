
# store pseudo auxilaries like Player, Actions ect
from .game_aux import PlayerState, Action, RED, GREEN, YELLOW, BLACK, BLUE, PURPLE, WHITE
from .pseudo_configs import *

import math
import numpy as np




    

class PseudoPlayer:
    width = PLAYER_SIZE
    height = PLAYER_SIZE 

    def __init__(self, x, y, state : PlayerState = PlayerState.NOT_EATING, screen_width = SCREEN_WIDTH, screen_height = SCREEN_HEIGHT):
        self.x = x
        self.y = y
        self.angle = 0  # Facing direction in degrees
        self.size = PLAYER_SIZE
        self.speed = 5
        self.rotation_speed = 5
        self.state = state
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.color = BLUE
        
    def move_forward(self):
        # Convert angle to radians and move in facing direction
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        self.x = int(self.x)
        self.y = int(self.y)

        
        # Keep player within screen bounds
        # self.x = max(self.size, min(self.screen_width - self.size, self.x))
        # self.y = max(self.size, min(self.screen_height - self.size, self.y))

    def move_backward(self):
        rad = math.radians(self.angle)
        self.x -= self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)
        self.x = int(self.x)
        self.y = int(self.y)
        # Keep player within screen bounds
        # self.x = max(self.size, min(self.screen_width - self.size, int(self.x)))
        # self.y = max(self.size, min(self.screen_height - self.size, int(self.y)))

    def rotate_left(self):
        self.angle -= self.rotation_speed
        self.angle %=  360

    def rotate_right(self):
        self.angle += self.rotation_speed
        self.angle %= 360

    
    def alternate_state(self):
        if self.state is PlayerState.EATING:
            self.state = PlayerState.NOT_EATING 
            self.color = BLUE
        elif self.state is PlayerState.NOT_EATING:
            self.state = PlayerState.EATING
            self.color = PURPLE
        else:
            pass
    
    def move_with_action(self,action : Action):
        # action is ROTATION @ TRANSLATION, SO A 2X2 matrix 
        # we need to update self.x, self.y and self.angle respectively
        
        movement_vector = action.as_vector(mode='deg')
        state_change_action = movement_vector[2]
        angle = movement_vector[1]
        # Update the object's angle (add the rotation to current angle)
        self.angle += angle
        
        # Optional: Keep angle in [0, 360) range
        self.angle = self.angle % 360


        # Update position
        theta_angle = np.deg2rad(self.angle)
        self.x = int(self.x  + movement_vector[0] * np.cos(theta_angle))
        self.y =int(self.y +  movement_vector[0] * np.sin(theta_angle))

        if state_change_action is not None and state_change_action != self.state: #here state change action will be represented by PlayerState
            self.alternate_state()

    def get_state(self):
        return self.state
       
    # TL, TR, BR, BL, center dictonary
    def get_pos(self):
        return (self.x, self.y)
    
    def get_orientation(self, mode = 'deg'):
        if mode == 'deg':
            return self.angle
        else:
            return np.deg2rad(self.angle)
    # illustration of keypoints:
    #  * (back-left)
    #  |  \ 
    #  |   \
    #  |    \
    #  |  *  * (front)
    #  |    /
    #  |   /
    #  |  /
    #  * (back-right)
    def get_keypoints(self, frame = 'self'):
        if frame == 'world':
            rad = math.radians(self.angle)

            # Calculate triangle points
            front_x = self.x + self.size * math.cos(rad)
            front_y = self.y + self.size * math.sin(rad)
            
            back_left_x = self.x + self.size * math.cos(rad + 2.4)
            back_left_y = self.y + self.size * math.sin(rad + 2.4)
            
            back_right_x = self.x + self.size * math.cos(rad - 2.4)
            back_right_y = self.y + self.size * math.sin(rad - 2.4)
            
            return {
                'center' : (self.x, self.y),
                'front' : (front_x, front_y),
                'back-left' : (back_left_x, back_left_y),
                'back-right' : (back_right_x, back_right_y),
            }
        else:           
            r = self.size
            return {
                'center' : (0,0),
                'front' : (self.size, 0),
                'back-left' : (r * math.cos(2.4), r * math.sin(2.4)),
                'back-right' : (r * math.cos(-2.4), r * math.sin(-2.4)),
            }   

    def draw(self, screen):
        # Draw player as a triangle pointing in the direction it's facing
        rad = math.radians(self.angle)
        
        # Calculate triangle points
        front_x = self.x + self.size * math.cos(rad)
        front_y = self.y + self.size * math.sin(rad)
        
        back_left_x = self.x + self.size * math.cos(rad + 2.4)
        back_left_y = self.y + self.size * math.sin(rad + 2.4)
        
        back_right_x = self.x + self.size * math.cos(rad - 2.4)
        back_right_y = self.y + self.size * math.sin(rad - 2.4)
        
        points = np.array([(front_x, front_y), (back_left_x, back_left_y), (back_right_x, back_right_y)])
        
        # Create mask for triangle
        mask = self.create_triangle_mask(screen.shape, points)
        
        # Apply color to masked pixels
        screen[mask] = self.color
        return screen
    
    def create_triangle_mask(self, screen_shape, points):
        """Create triangle mask using barycentric coordinates -- by claude"""
        
        h, w = screen_shape[:2]

        # Get triangle vertices
        p1, p2, p3 = points
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        # Calculate barycentric coordinates for each pixel
        denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
        
        if abs(denom) < 1e-10:  # Degenerate triangle
            return np.zeros((h, w), dtype=bool)
        
        a = ((p2[1] - p3[1]) * (x - p3[0]) + (p3[0] - p2[0]) * (y - p3[1])) / denom
        b = ((p3[1] - p1[1]) * (x - p3[0]) + (p1[0] - p3[0]) * (y - p3[1])) / denom
        c = 1 - a - b
        
        # Point is inside triangle if all barycentric coordinates are positive
        return (a >= 0) & (b >= 0) & (c >= 0)

    def _snap_xy(self):
        """snap current (x,y) to nearest int and clamp to screen."""
        p = [int(self.x), int(self.y)]
        # clamp to screen bounds considering size
        px = int(max(self.size, min(self.screen_width - self.size, p[0])))
        py = int(max(self.size, min(self.screen_height - self.size, p[1])))
        self.x, self.y = px, py
# we will refactor this when everything is done

# change each to return get_pos-> (x,y) and get_kp

class PseudoEdibleObject:
    def __init__(self, x, y, width=20, height=20):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.eaten = False
        self.color = GREEN
    
    
    def draw(self, screen):
        h, w = screen.shape[:2]
        
        # Calculate bounds
        radius = min(self.width, self.height) // 2
        left = max(0, int(self.x - radius))
        right = min(w, int(self.x + radius + 1))
        top = max(0, int(self.y - radius))
        bottom = min(h, int(self.y + radius + 1))
        
        if left < right and top < bottom:
            # Create coordinate grids only for the bounding region
            y_local, x_local = np.ogrid[top:bottom, left:right]
            
            # Calculate distance from center
            distance_squared = (x_local - self.x)**2 + (y_local - self.y)**2
            
            # Create local circle mask
            local_circle_mask = distance_squared <= radius**2
            
            # Apply to the screen region
            screen[top:bottom, left:right][local_circle_mask] = GREEN
            # screen[left:right, top:bottom][local_circle_mask] = GREEN
        
        return screen

    # TL, TR, BR, BL, CENTROID dictonary
    def get_pos(self):
        return (self.x, self.y)
    
    def get_keypoints(self, frame = 'self'):
        if frame == 'self':
            return {
                'top-left': (-self.width//2, -self.height//2),
                'top-right': (self.width//2, -self.height//2),
                'bot-right': (self.width//2, self.height//2),
                'bot-left': (-self.width//2, self.height//2),
                'center': (0, 0),
            }
        else:   
            return {
                'top-left': (self.x - self.width//2, self.y - self.height//2),
                'top-right': (self.x + self.width//2, self.y - self.height//2),
                'bot-right': (self.x + self.width//2, self.y + self.height//2),
                'bot-left': (self.x - self.width//2, self.y + self.height//2),
                'center': (self.x, self.y),
            }

class PseudoObstacle:
    def __init__(self, x, y, width=40, height=40):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = RED
    
    
    def draw(self, screen):
        h, w = screen.shape[:2]
        
        # Calculate bounds
        left = max(0, int(self.x - self.width//2))
        right = min(w, int(self.x + self.width//2))
        top = max(0, int(self.y - self.height//2))
        bottom = min(h, int(self.y + self.height//2))
        
        # Draw rectangle using slicing
        if left < right and top < bottom:
            screen[top:bottom, left:right] = self.color
            # screen[left:right, top:bottom] = self.color
        
        return screen
    
    # TL, TR, BR, BL, CENTROID dictonary
    def get_pos(self):
        return (self.x, self.y)
    
    def get_keypoints(self, frame = 'self'):
        if frame == 'self':
            return {
                'top-left': (-self.width//2, -self.height//2),
                'top-right': (self.width//2, -self.height//2),
                'bot-right': (self.width//2, self.height//2),
                'bot-left': (-self.width//2, self.height//2),
                'center': (0, 0),
            }
        else:   
            return {
                'top-left': (self.x - self.width//2, self.y - self.height//2),
                'top-right': (self.x + self.width//2, self.y - self.height//2),
                'bot-right': (self.x + self.width//2, self.y + self.height//2),
                'bot-left': (self.x - self.width//2, self.y + self.height//2),
                'center': (self.x, self.y),
            }
    
class PseudoGoal:
    def __init__(self, x, y, width=50, height=50):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = YELLOW
        self.border_color = BLACK
        self.border_offset = 3
    
    
    def draw(self, screen):
        border_offset = self.border_offset
        h, w = screen.shape[:2]
        
        # Calculate bounds
        left = max(0, int(self.x - self.width//2))
        right = min(w, int(self.x + self.width//2))
        top = max(0, int(self.y - self.height//2))
        bottom = min(h, int(self.y + self.height//2))
        
        # Inner rectangle bounds
        inner_left = max(0, int(self.x - self.width//2 + border_offset))
        inner_right = min(w, int(self.x + self.width//2 - border_offset))
        inner_top = max(0, int(self.y - self.height//2 + border_offset))
        inner_bottom = min(h, int(self.y + self.height//2 - border_offset))
        
        # Draw outer rectangle (border)
        if left < right and top < bottom:
            screen[top:bottom, left:right] = self.border_color
            # screen[left:right, top:bottom] = self.border_color

        
        # Draw inner rectangle (main color)
        if inner_left < inner_right and inner_top < inner_bottom:
            screen[inner_top:inner_bottom, inner_left:inner_right] = self.color
            # screen[inner_left:inner_right,inner_top:inner_bottom] = self.color

        
        return screen
    
    def get_pos(self):
        return (self.x, self.y)
    
    def get_keypoints(self, frame = 'self'):
        if frame == 'self':
            return {
                'top-left': (-self.width//2, -self.height//2),
                'top-right': (self.width//2, -self.height//2),
                'bot-right': (self.width//2, self.height//2),
                'bot-left': (-self.width//2, self.height//2),
                'center': (0, 0),
            }
        else:   
            return {
                'top-left': (self.x - self.width//2, self.y - self.height//2),
                'top-right': (self.x + self.width//2, self.y - self.height//2),
                'bot-right': (self.x + self.width//2, self.y + self.height//2),
                'bot-left': (self.x - self.width//2, self.y + self.height//2),
                'center': (self.x, self.y),
            }
AVAILABLE_OBJECTS = [PseudoEdibleObject, PseudoObstacle, PseudoGoal]

# class GameObjective(Enum):
#     EAT_ALL = 1
#     REACH_GOAL = 2


# # NEED TO INCORPORATE GAMEINTERFACE SUCH THAT IT CAN TAKE IN DEMO AND AGENT MOVES 
# class GameMode(Enum):
#     DEMO_MODE = 1
#     AGENT_MODE = 2 

