# game_aux.py (append near your enums)
from dataclasses import dataclass
import math
import random 
from .game_aux import * 

class GameObjective(Enum):
    EAT_ALL = 1
    REACH_GOAL = 2
    PUSH_AND_PLACE = 3
    PARKING = 4

class GameDifficulty:
    EASY = 1
    MEDIUM = 2
    HARD = 3 

class ObjectiveStrategy:
    def setup(self, game): ...
    def update(self, game): ...
    def draw(self, game): ...
    def is_success(self, game) -> bool: return False


class EatAllStrategy(ObjectiveStrategy):
    NUM_EDIBLES = 2
    def __init__(self):
        pass 

    def setup(self, game): 
        for _ in range(EatAllStrategy.NUM_EDIBLES):
            x = random.randint(50, game.screen_width - 50)
            y = random.randint(50, game.screen_height - 50)
            # Make sure edibles don't spawn too close to player
            while math.sqrt((x - game.player.x)**2 + (y - game.player.y)**2) < 100:
                x = random.randint(50, game.screen_width - 50)
                y = random.randint(50, game.screen_height - 50)
            
            width = random.randint(15, 25)
            height = random.randint(15, 25)
            game.edibles.append(EdibleObject(x, y, width, height))

    def update(self, game):
        # check if eating
        player_rect = game.player.get_rect()
        for edible in game.edibles:
            if not edible.eaten and player_rect.colliderect(edible.get_rect()) and game.player.state is PlayerState.EATING:
                edible.eaten = True
        
        # check if eaten all
        if all(edible.eaten for edible in game.edibles):
            game.game_won = True

    def draw(self, game): 
     for edible in game.edibles:
            edible.draw(game.screen)

    def is_success(self, game):
        # computed in update
        return all(e.eaten for e in game.edibles)

class ReachGoalStrategy(ObjectiveStrategy):
    NUM_OBSTACLES = 1
    def __init__(self):
        pass 

    def setup(self, game): 
        for _ in range(ReachGoalStrategy.NUM_OBSTACLES):
            x = random.randint(50, game.screen_width - 50)
            y = random.randint(50, game.screen_height - 50)
            # Make sure obstacles don't spawn too close to player
            while math.sqrt((x - game.player.x)**2 + (y - game.player.y)**2) < 150:
                x = random.randint(50,  game.screen_width - 50)
                y = random.randint(50, game.screen_height - 50)
            
            width = random.randint(30, 50)
            height = random.randint(30, 50)
            game.obstacles.append(Obstacle(x, y, width, height))
        goal_position = (game.screen_width - 100 + np.random.randint(-25,25), game.screen_height - 100 + np.random.randint(-25,25))
        game.goal = Goal(goal_position[0], goal_position[1])

    def update(self, game):

        player_rect = game.player.get_rect() 
        for obstacle in game.obstacles:
            if player_rect.colliderect(obstacle.get_rect()):
                game.game_over = True
                return

        goal_rect = game.goal.get_rect()
        if player_rect.colliderect(goal_rect):
            game.game_won = True

    def draw(self, game):
        for obstacle in game.obstacles:
            obstacle.draw(game.screen)
        game.goal.draw(game.screen)

    def is_success(self, game):
        # computed in update
        return game.game_won

# New: Parking
@dataclass
class ParkingTarget:
    x: float; y: float; theta_deg: float
    pos_threshold: float = 5.0
    ang_threshold_deg: float = 10.0

class ParkAtPoseStrategy(ObjectiveStrategy):
    def __init__(self, target: ParkingTarget, difficulty = GameDifficulty.EASY):
        self.target = target
        self.difficulty = difficulty

    def setup(self, game):
        # optional: show a visual marker using your Goal rect for now
        game.goal = Goal(int(self.target.x), int(self.target.y))

        # create 3 obstacles walls around parking space of PLAYER_SIZE * 2.5 
        # walls (short side) are of width 10  
        space = PLAYER_SIZE * 2.5 
        wall_thickness = 10
        wall1 = Obstacle(self.target.x + space//2, self.target.y, width=wall_thickness, height=space)
        wall2 = Obstacle(self.target.x - space//2, self.target.y,width=wall_thickness, height=space)
        wall3 = Obstacle(self.target.x, self.target.y + space//2, width=space, height=wall_thickness )
        wall4 = Obstacle(self.target.x, self.target.y - space//2, width=space, height=wall_thickness )
        walls = [wall1, wall2, wall3, wall4]
        rm_idx = random.randrange(len(walls))
        del walls[rm_idx]
        game.obstacles = walls 

    def update(self, game):

        if self.difficulty == GameDifficulty.HARD: # only disqualify if difficulty is hard
            player_rect = game.player.get_rect() 
            for obstacle in game.obstacles:
                if player_rect.colliderect(obstacle.get_rect()):
                    game.game_over = True
                    return


    def draw(self, game):
        # could draw a small orientation arrow at target.theta_deg if you like
        game.goal.draw(game.screen)
        if self.difficulty == GameDifficulty.HARD:
            for obstacle in game.obstacles:
                obstacle.draw(game.screen)

    def is_success(self, game):
        if game.game_over:
            return False 
        px, py = game.player.get_pos()
        ang = game.player.get_orientation('deg')
        # position
        dx = px - self.target.x; dy = py - self.target.y
        pos_ok = math.hypot(dx, dy) <= self.target.pos_threshold

        if self.difficulty == GameDifficulty.EASY:
            return pos_ok
        # angle (wrap)
        d = (ang - self.target.theta_deg + 180.0) % 360.0 - 180.0
        ang_ok = abs(d) <= self.target.ang_threshold_deg

        # if self.difficulty == GameDifficulty.MEDIUM:
            # return pos_ok and ang_ok
        return pos_ok and ang_ok

        # we assume ON to be engine off, its confusing but using OFF will be to easy 
        # player_state_ON = game.player.state == PlayerState.EATING 
        
        # return pos_ok and ang_ok and player_state_ON
        

# New: Push-and-Place (simple goal check first)
@dataclass
class PushSpec:
    goal_x: float; goal_y: float; goal_radius: float = 12.0

class PushObjectToGoalStrategy(ObjectiveStrategy):
    def __init__(self, spec: PushSpec = None, difficulty = GameDifficulty.EASY):
        self.spec = spec
        self.object_pos = None 
        self.difficulty = difficulty


    def setup(self, game):
        game.goal = Goal(int(self.spec.goal_x), int(self.spec.goal_y))
        game.edibles = [EdibleObject(int(self.object_pos[0]), int(self.object_pos[1]))]

    def draw(self, game):
        for edible in game.edibles:
            edible.draw(game.screen)
        game.goal.draw(game.screen)
        

        # IMPORTANT: for a true push, you’ll later add simple kinematics to move the object
    
    def update(self, game):
        px, py = game.player.get_pos()
        obj = game.edibles[0]
        obj_x, obj_y = obj.get_pos()
        push_radius = min([game.player.size, obj.width, obj.height])
        if math.hypot(px - obj_x, py - obj_y) <= push_radius:
            if self.difficulty != GameDifficulty.EASY and game.player.state == PlayerState.EATING:
                obj.eaten = True
            elif self.difficulty == GameDifficulty.EASY:
                obj.eaten = True 
        else:
            obj.eateN = False

    
    def is_success(self, game):

        obj = game.edibles[0]
        if obj is None:
            return False
        ox, oy = obj.get_pos()
        return math.hypot(ox - self.spec.goal_x, oy - self.spec.goal_y) <= self.spec.goal_radius

# Factory to choose strategy by GameObjective or config
def make_objective_strategy(game, cfg=None):
    # cfg is optional per-level info e.g. parking target or push spec
    if game.objective == GameObjective.EAT_ALL:
        return EatAllStrategy()
    if game.objective == GameObjective.REACH_GOAL:
        return ReachGoalStrategy()
    if getattr(game, 'objective', None) == GameObjective(3):  # or add enum PARK_AT_POSE
        x,y = cfg['target']
        parking_target = ParkingTarget(x,y,0)
        return ParkAtPoseStrategy(parking_target, cfg['difficulty'])
    if getattr(game, 'objective', None) == GameObjective(4):  # or add enum PUSH_OBJECT_TO_GOAL
        x,y = cfg['target']
        push_spec = PushSpec(x,y)
        return PushObjectToGoalStrategy(push_spec, cfg['difficulty'])
    raise ValueError("Unknown objective")
