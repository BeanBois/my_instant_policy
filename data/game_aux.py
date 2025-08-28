
# store game auxilaries like Player, Actions ect
from .game_configs import *

import math
import numpy as np
from enum import Enum 
import pygame 


class PlayerState(Enum):
    NOT_EATING = 0
    EATING = 1
    

class Action:
    def __init__(self, forward_movement : float, rotation_deg : float, state_change : PlayerState):
        self.forward_movement = forward_movement
        self.rotation = rotation_deg
        self.state_change = state_change
    
    def to_SE2(self):
        SE2 = np.zeros((3,3))
        theta_rad = np.deg2rad(self.rotation)
        _cos = np.cos(theta_rad)
        _sin = np.sin(theta_rad)
        SE2[0,0] = _cos 
        SE2[0,1] = -_sin
        SE2[1,0] = _sin
        SE2[1,1] = _cos

        SE2[0,2] = _cos * self.forward_movement
        SE2[1,2] = _sin * self.forward_movement
        SE2[2,2] = 1 
        
        return (SE2, self.state_change.value)
    
    def as_vector(self, mode = 'deg'):
        vector = np.zeros(3)
        theta = np.deg2rad(self.rotation)
        vector[0] = self.forward_movement
        vector[1] = self.rotation if mode == 'deg' else theta
        vector[2] = self.state_change.value
        return vector 

    

class Player:
    width = PLAYER_SIZE
    height = PLAYER_SIZE
    def __init__(self, x, y, state : PlayerState = PlayerState.NOT_EATING, screen_width = SCREEN_WIDTH, screen_height = SCREEN_HEIGHT):
        self.x = x
        self.y = y
        self.angle = 0  # Facing direction in degrees
        self.size = PLAYER_SIZE
        self.speed = PLAYER_SPEED
        self.rotation_speed = PLAYER_ROTATION_SPEED
        self.state = state
        self.screen_width = screen_width
        self.screen_height = screen_height
        
    def move_forward(self):
        # Convert angle to radians and move in facing direction
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        
        # Keep player within screen bounds
        self.x = max(self.size, min(self.screen_width - self.size, self.x))
        self.y = max(self.size, min(self.screen_height - self.size, self.y))
    
    def move_backward(self):
        rad = math.radians(self.angle)
        self.x -= self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)
        
        # Keep player within screen bounds
        self.x = max(self.size, min(self.screen_width - self.size, self.x))
        self.y = max(self.size, min(self.screen_height - self.size, self.y))
    
    def rotate_left(self):
        self.angle -= self.rotation_speed
        self.angle %= 360
    
    def rotate_right(self):
        self.angle += self.rotation_speed
        self.angle %= 360
    
    def alternate_state(self):
        if self.state is PlayerState.EATING:
            self.state = PlayerState.NOT_EATING 
        elif self.state is PlayerState.NOT_EATING:
            self.state = PlayerState.EATING
        else:
            pass
 
    def get_rect(self):
        return pygame.Rect(self.x - self.size, self.y - self.size, 
                          self.size * 2, self.size * 2)
    

    def move_with_action(self,action : Action):
        # action is ROTATION @ TRANSLATION, SO A 2X2 matrix 
        # we need to update self.x, self.y and self.angle respectively
        
        movement_vector = action.as_vector(mode='deg')
        # movement_vector = action.vector_for_pygame(mode='deg')

        state_change_action = movement_vector[2]
        angle = movement_vector[1]
        # Update the object's angle (add the rotation to current angle)
        self.angle += angle
        
        # Optional: Keep angle in [0, 360) range
        self.angle = self.angle % 360

        # Update position
        theta_angle = np.deg2rad(self.angle)

        self.x = int(self.x  + movement_vector[0] * np.cos(theta_angle))
        self.y = int(self.y +  movement_vector[0] * np.sin(theta_angle))
        

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
        
        points = [(front_x, front_y), (back_left_x, back_left_y), (back_right_x, back_right_y)]

        if self.state is PlayerState.NOT_EATING:
            pygame.draw.polygon(screen, BLUE, points)
        else:
            pygame.draw.polygon(screen, PURPLE, points)


# we will refactor this when everything is done

# change each to return get_pos-> (x,y) and get_kp
class EdibleObject:
    def __init__(self, x, y, width=20, height=20):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.eaten = False
    
    def get_rect(self):
        return pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                          self.width, self.height)
    
    def draw(self, screen):
        if not self.eaten:
            pygame.draw.ellipse(screen, GREEN, self.get_rect())

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

    def move_forward(self, angle):
        speed = PLAYER_SPEED
        # Convert angle to radians and move in facing direction
        rad = math.radians(angle)
        self.x += speed * math.cos(rad)
        self.y += speed * math.sin(rad)
        
        # self.x = max(self.width, min(self.screen_width - self.width, self.x))
        # self.y = max(self.height, min(self.screen_height - self.height, self.y))
    
    def move_backward(self, angle):
        speed = PLAYER_SPEED
        rad = math.radians(angle)
        self.x -= speed * math.cos(rad)
        self.y -= speed * math.sin(rad)
        
        # self.x = max(self.size, min(self.screen_width - self.width, self.x))
        # self.y = max(self.size, min(self.screen_height - self.height, self.y))
    
    def rotate_left(self, radius):

        angle = -PLAYER_ROTATION_SPEED
        theta = math.radians(theta)
        self.x += radius * math.cos(theta)
        self.y += radius * math.sin(theta)
 
    def rotate_right(self, radius):
        angle = PLAYER_ROTATION_SPEED
        theta = math.radians(theta)
        self.x += radius * math.cos(theta)
        self.y += radius * math.sin(theta)

    def move_with_action(self,action : Action):
        movement_vector = action.as_vector(mode='deg')
        angle = movement_vector[1]
        # Update position
        theta_angle = np.deg2rad(angle)

        self.x = int(self.x  + movement_vector[0] * np.cos(theta_angle))
        self.y = int(self.y +  movement_vector[0] * np.sin(theta_angle))
        

class Obstacle:
    def __init__(self, x, y, width=40, height=40):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def get_rect(self):
        return pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                          self.width, self.height)
    
    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.get_rect())
    
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
    
class Goal:
    def __init__(self, x, y, width=50, height=50):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def get_rect(self):
        return pygame.Rect(self.x - self.width//2, self.y - self.height//2, 
                          self.width, self.height)
    
    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW, 
                        (self.x - 25, self.y - 25, 50, 50))
        pygame.draw.rect(screen, BLACK, 
                        (self.x - 25, self.y - 25, 50, 50), 3)
    
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



# NEED TO INCORPORATE GAMEINTERFACE SUCH THAT IT CAN TAKE IN DEMO AND AGENT MOVES 
class GameMode(Enum):
    DEMO_MODE = 1
    AGENT_MODE = 2 

