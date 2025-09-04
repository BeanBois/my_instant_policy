
# store pseudo auxilaries like Player, Actions ect
from .game_aux import PlayerState, Action, RED, GREEN, YELLOW, BLACK, BLUE, PURPLE, WHITE
from .pseudo_configs import *

import math
import numpy as np


def world_to_img(screen_height, x, y):
    row = int(screen_height - 1 - y)   # y up  -> row down
    col = int(x)                       # x right -> col right
    return row, col

def world_to_img_xy(screen_height, x, y):
    """Return (x_img, y_img) == (col, row) for rasterising formulas."""
    r, c = world_to_img(screen_height, x, y)
    return float(c), float(r)

def triangle_mask(hw, pts_xy):
    """Filled triangle mask from 3 vertices given in image coords (x=col, y=row)."""
    H, W = hw
    (x1,y1), (x2,y2), (x3,y3) = pts_xy

    # Bounding box (clipped)
    xmin = max(int(np.floor(min(x1,x2,x3))), 0)
    xmax = min(int(np.ceil (max(x1,x2,x3)))+1, W)
    ymin = max(int(np.floor(min(y1,y2,y3))), 0)
    ymax = min(int(np.ceil (max(y1,y2,y3)))+1, H)
    if xmin >= xmax or ymin >= ymax:
        return np.zeros((H,W), dtype=bool)

    xs = np.arange(xmin, xmax)
    ys = np.arange(ymin, ymax)
    X, Y = np.meshgrid(xs, ys)  # X=cols, Y=rows

    # Barycentric test
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-10:
        out = np.zeros((H,W), dtype=bool)
        return out
    a = ((y2 - y3)*(X - x3) + (x3 - x2)*(Y - y3)) / denom
    b = ((y3 - y1)*(X - x3) + (x1 - x3)*(Y - y3)) / denom
    c = 1.0 - a - b
    inside = (a >= 0) & (b >= 0) & (c >= 0)

    out = np.zeros((H,W), dtype=bool)
    out[ymin:ymax, xmin:xmax] = inside
    return out

    


    

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

    def draw(self, screen, screen_height):
        rad = math.radians(self.angle)

        # Triangle vertices in *world* coords
        front = (self.x + self.size*math.cos(rad),          self.y + self.size*math.sin(rad))
        bl    = (self.x + self.size*math.cos(rad + 2.4),    self.y + self.size*math.sin(rad + 2.4))
        br    = (self.x + self.size*math.cos(rad - 2.4),    self.y + self.size*math.sin(rad - 2.4))

        # Convert each to *image* coords (x_img, y_img) = (col, row)
        pts_img = np.array([
            world_to_img_xy(screen_height, *front),
            world_to_img_xy(screen_height, *bl),
            world_to_img_xy(screen_height, *br),
        ], dtype=float)   # shape (3,2): [[x1,y1],[x2,y2],[x3,y3]]

        mask = triangle_mask(screen.shape[:2], pts_img)  # boolean (H,W)
        screen[mask] = self.color
        return screen

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
    
    
    def draw(self, screen, screen_height):
        H, W = screen.shape[:2]
        row_c = float(screen_height - 1 - self.y)
        col_c = float(self.x)
        r = float(min(self.width, self.height) // 2)

        # Bounds in image coords
        top    = max(0, int(np.floor(row_c - r)))
        bottom = min(H, int(np.ceil (row_c + r)) + 1)
        left   = max(0, int(np.floor(col_c - r)))
        right  = min(W, int(np.ceil (col_c + r)) + 1)
        if left >= right or top >= bottom:
            return screen

        yy, xx = np.ogrid[top:bottom, left:right]
        mask = (xx - col_c)**2 + (yy - row_c)**2 <= r**2
        screen[top:bottom, left:right][mask] = GREEN
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
    
    
    def draw(self, screen, screen_height):
        H, W = screen.shape[:2]
        row_c = int(screen_height - 1 - self.y)  # world y -> image row
        col_c = int(self.x)

        half_h = self.height // 2
        half_w = self.width  // 2

        top    = max(0, row_c - half_h)
        bottom = min(H, row_c + half_h)
        left   = max(0, col_c - half_w)
        right  = min(W, col_c + half_w)

        if left < right and top < bottom:
            screen[top:bottom, left:right] = self.color
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
    
    
    def draw(self, screen, screen_height):
        H, W = screen.shape[:2]
        row_c = int(screen_height - 1 - self.y)
        col_c = int(self.x)

        half_h = self.height // 2
        half_w = self.width  // 2
        bo     = int(self.border_offset)

        top,    bottom = max(0, row_c - half_h), min(H, row_c + half_h)
        left,   right  = max(0, col_c - half_w), min(W, col_c + half_w)

        if left < right and top < bottom:
            screen[top:bottom, left:right] = self.border_color

        itop, ibottom = max(0, top+bo),    min(H, bottom-bo)
        ileft, iright = max(0, left+bo),   min(W, right-bo)
        if ileft < iright and itop < ibottom:
            screen[itop:ibottom, ileft:iright] = self.color
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

