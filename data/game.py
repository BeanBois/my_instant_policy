# should store Game Code only
from .game_aux import * 
import random
import json
import os
from .game_obj import GameDifficulty, make_objective_strategy, GameObjective

class Game:
    
    def __init__(
        self, 
        objective: GameObjective = None, 
        difficulty: GameDifficulty  = None,
        objective_cfg = None, 
        num_edibles : int= NUM_OF_EDIBLE_OBJECT, 
        num_obstacles : int= NUM_OF_OBSTACLES, 
        screen_width = SCREEN_WIDTH, 
        screen_height = SCREEN_HEIGHT,
        player_start_pos = None
        ):


        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("2D Game - Eat or Avoid")
        self.clock = pygame.time.Clock()
        
        # Initialize game objects
        self.player_start_pos = player_start_pos
        if player_start_pos is None:
            player_starting_pos_x =  np.random.randint(0 + Player.width ,self.screen_width - Player.width) 
            player_starting_pos_y =  np.random.randint(0 + Player.height,self.screen_height - Player.height) 
            self.player_start_pos = (player_starting_pos_x, player_starting_pos_y)

        self.player = Player(self.player_start_pos[0], self.player_start_pos[1])
        self.edibles = []
        self.obstacles = []
        self.goal = None
        
        # Game state
        self.objective_strategy = None  # helper 
        self.objective_cfg = objective_cfg
        if self.objective_cfg is None:

            offset = 100
            random_target_x = random.random() * (self.screen_width - 2* offset) + offset 
            random_target_y = random.random() * (self.screen_height - 2* offset) + offset 

            self.objective_cfg = {
                'difficulty' : difficulty,
                'target' : (random_target_x, random_target_y)
            }  

        self.objective = None 
        self.game_over = False
        self.game_won = False

        self.setup_game()

    def setup_game(self):
        # Randomly choose objective
        if self.objective is None:
            self.objective = random.choice([objective for objective in GameObjective])

        self.objective_strategy = make_objective_strategy(self, cfg=getattr(self, 'objective_cfg', None))
        self.objective_strategy.setup(self)
        
        # Command line print to inform player
        print("-" * 25)
        objective_str = None
        if self.objective is GameObjective.EAT_ALL:
            objective_str = "eat all food"
        else:
            objective_str = "reach goal"
        print(f"Welcome to the game, your objective is to {objective_str}")
        print("-" * 25)

    def handle_events(self):
        if self.game_over or self.game_won:
            return False
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.game_over and not self.game_won:
                    self.player.alternate_state()
            # Handle continuous key presses
        keys = pygame.key.get_pressed()
        if not self.game_over and not self.game_won:
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.player.move_forward()
                if self.objective == GameObjective.PUSH_AND_PLACE:
                    obj = self.edibles[0]
                    in_contact = obj.eaten 
                    if in_contact:
                        obj.move_forward(self.player.angle)
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.player.move_backward()
                if self.objective == GameObjective.PUSH_AND_PLACE:
                    obj = self.edibles[0]
                    in_contact = obj.eaten 
                    if in_contact:
                        obj.move_backward(self.player.angle)
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.player.rotate_left()
                if self.objective == GameObjective.PUSH_AND_PLACE:
                    obj = self.edibles[0]
                    in_contact = obj.eaten 
                    if in_contact:
                        obj.rotate_left(self.player.size)
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.player.rotate_right()
                if self.objective == GameObjective.PUSH_AND_PLACE:
                    obj = self.edibles[0]
                    in_contact = obj.eaten 
                    if in_contact:
                        obj.rotate_right(self.player.sizes)

        return True

    def handle_action(self,action : Action):
        if self.game_over or self.game_won:
            return False
        
        # Handle continuous key presses
        self.player.move_with_action(action)
        return True
  
    def update(self):
        if self.game_over or self.game_won:
            return
        self.objective_strategy.update(self)
        self.game_won = self.objective_strategy.is_success(self)
        
    def draw(self):
        self.screen.fill(WHITE)
        self.objective_strategy.draw(self)
        self.player.draw(self.screen)
        
        pygame.display.flip()
    
    def restart_game(self):
        self.player = Player(self.player_start_pos[0], self.player_start_pos[1])
        # reset variables 
        for edible in self.edibles:
            edible.eaten = False
        
        # self.edibles = []
        # self.obstacles = []
        # self.goal = None
        self.game_over = False
        self.game_won = False
        # self.setup_game()
    
    def end_game(self):
        pygame.quit()

    # RUNS THE WHOLE GAME 
    def run(self, max_time_out = 1000):
        self.obs = None
        self.obs = []
        running = True
        t = 0
        while running and t < max_time_out:
            t+=1
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        
        pygame.quit()

    def get_screen_pixels(self):
        pixels = [
            [ np.array(self.screen.get_at((x,y))[:3]) for y in range(self.screen_height)] for x in range(self.screen_width)
        ]
        return np.array(pixels)

    def save_config(self, filename):
        """
        Save the current game configuration to a JSON file.
        
        Args:
            filename (str): Path to save the configuration file
        """
        config = {
            # Game parameters
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'objective': self.objective.value if self.objective else None,
            'objective_cfg' : self.objective_cfg,
            # 'player_start_pos': self.player_start_pos,
            
            # Game objects positions and properties
            'edibles': [
                {
                    'x': edible.x,
                    'y': edible.y,
                    'width': edible.width,
                    'height': edible.height,
                    'eaten': edible.eaten
                } for edible in self.edibles
            ],
            
            'obstacles': [
                {
                    'x': obstacle.x,
                    'y': obstacle.y,
                    'width': obstacle.width,
                    'height': obstacle.height
                } for obstacle in self.obstacles
            ],
            
            'goal': {
                'x': self.goal.x,
                'y': self.goal.y
            } if self.goal else None,
            
            # Game state
            'game_over': self.game_over,
            'game_won': self.game_won
        }
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Game configuration saved to {filename}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def load_config(self, filename):
        """
        Load game configuration from a JSON file and reinitialize the game.
        
        Args:
            filename (str): Path to the configuration file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filename):
                print(f"Configuration file {filename} not found")
                return False
            
            with open(filename, 'r') as f:
                config = json.load(f)
            
            # Load game parameters
            self.screen_width = config.get('screen_width', SCREEN_WIDTH)
            self.screen_height = config.get('screen_height', SCREEN_HEIGHT)
            # self.player_start_pos = tuple(config.get('player_start_pos', (PLAYER_START_X, PLAYER_START_Y)))
            player_starting_pos_x =  np.random.randint(0 + Player.width ,self.screen_width - Player.width) 
            player_starting_pos_y =  np.random.randint(0 + Player.height,self.screen_height - Player.height) 
            self.player_start_pos = (player_starting_pos_x, player_starting_pos_y)
            # Load objective
            objective_value = config.get('objective')
            if objective_value:
                self.objective = GameObjective(objective_value)
            else:
                self.objective = None
            self.objective_cfg = config.get('objective_cfg')
            self.objective_strategy = make_objective_strategy(self, self.objective_cfg)
            # Reinitialize pygame screen if dimensions changed
            current_screen_size = self.screen.get_size()
            if current_screen_size != (self.screen_width, self.screen_height):
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            
            # Reset player
            self.player = Player(self.player_start_pos[0], self.player_start_pos[1])
            
            # Clear existing objects
            self.edibles = []
            self.obstacles = []
            self.goal = None
            
            # Load edibles
            for edible_data in config.get('edibles', []):
                edible = EdibleObject(
                    edible_data['x'],
                    edible_data['y'],
                    edible_data['width'],
                    edible_data['height']
                )
                edible.eaten = edible_data.get('eaten', False)
                self.edibles.append(edible)
            
            # Load obstacles
            for obstacle_data in config.get('obstacles', []):
                obstacle = Obstacle(
                    obstacle_data['x'],
                    obstacle_data['y'],
                    obstacle_data['width'],
                    obstacle_data['height']
                )
                self.obstacles.append(obstacle)
            
            # Load goal
            goal_data = config.get('goal')
            if goal_data:
                self.goal = Goal(goal_data['x'], goal_data['y'])
            
            # Load game state
            self.game_over = False
            self.game_won = False
            
            print(f"Game configuration loaded from {filename}")
            print("-" * 25)
            objective_str = None
            if self.objective is GameObjective.EAT_ALL:
                objective_str = "eat all food"
            elif self.objective is GameObjective.REACH_GOAL:
                objective_str = "reach goal"
            elif self.objective is GameObjective.PUSH_AND_PLACE:
                objective_str = 'push and place'
            elif self.objective is GameObjective.PARKING:
                objective_str = 'parking'
                self.objective_strategy.setup(self)
            else:
                objective_str = "unknown"
            print(f"Loaded game objective: {objective_str}")
            print("-" * 25)
            
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False


