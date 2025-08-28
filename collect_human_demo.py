# general filestructure
# > human_demo
#   > demoset0
#     > demo_config.json
#     > demo0.json (or .pkl)
#     > demo1.json
#     > ...
#   > demoset1
#     > demo_config.json
#     > demo0.json
#     > ...

from data import  GameInterface
from data import Game 
from data import PlayerState

import json
import pickle
import os
import numpy as np
from enum import Enum 


DEMOSET_SIZE = 1
NUM_KIND_OF_DEMO = 1


def save_demo(demo, filepath):
    """
    Save demo data (list of dictionaries) to file.
    
    Args:
        demo (list): List of dictionaries containing observations
        filepath (str): Path to save the demo file (without extension)
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Try to save as JSON first (more readable and portable)
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_demo = []
            for observation in demo:
                json_obs = {}
                for key, value in observation.items():
                    if isinstance(value, np.ndarray):
                        json_obs[key] = {
                            'data': value.tolist(),
                            'dtype': str(value.dtype),
                            'shape': value.shape,
                            'type': 'numpy_array'
                        }
                    elif isinstance(value, (np.integer, np.floating)):
                        json_obs[key] = float(value)
                    elif isinstance(value, Enum):
                        json_obs[key] = int(value.value)
                    else:
                        json_obs[key] = value
                json_demo.append(json_obs)
            
            with open(filepath + '.json', 'w') as f:
                json.dump({
                    'demo_length': len(demo),
                    'observations': json_demo
                }, f, indent=2)
            
            print(f"Demo saved as JSON: {filepath}.json ({len(demo)} observations)")
            
        except (TypeError, ValueError) as json_error:
            # If JSON fails, fall back to pickle
            print(f"JSON serialization failed ({json_error}), using pickle instead")
            
            with open(filepath + '.pkl', 'wb') as f:
                pickle.dump({
                    'demo_length': len(demo),
                    'observations': demo
                }, f)
            
            print(f"Demo saved as pickle: {filepath}.pkl ({len(demo)} observations)")
            
    except Exception as e:
        print(f"Error saving demo to {filepath}: {e}")
        raise

def load_demo(filepath):
    """
    Load demo data from file.
    
    Args:
        filepath (str): Path to the demo file (without extension)
        
    Returns:
        list: List of dictionaries containing observations
    """
    try:
        # Try JSON first
        json_path = filepath + '.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Convert back numpy arrays
            demo = []
            for observation in data['observations']:
                obs = {}
                for key, value in observation.items():
                    if isinstance(value, dict) and value.get('type') == 'numpy_array':
                        obs[key] = np.array(value['data'], dtype=value['dtype']).reshape(value['shape'])
                    elif key == 'agent-state':
                        obs[key] = PlayerState(value)
                    else:
                        obs[key] = value
                demo.append(obs)
            
            print(f"Demo loaded from JSON: {json_path} ({len(demo)} observations)")
            return demo
        
        # Try pickle
        pkl_path = filepath + '.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Demo loaded from pickle: {pkl_path} ({data['demo_length']} observations)")
            return data['observations']
        
        raise FileNotFoundError(f"No demo file found at {filepath} (.json or .pkl)")
        
    except Exception as e:
        print(f"Error loading demo from {filepath}: {e}")
        raise

def save_demo_metadata(demo, filepath):
    """
    Save metadata about the demo for quick inspection.
    
    Args:
        demo (list): List of dictionaries containing observations
        filepath (str): Path to save the metadata file
    """
    try:
        if not demo:
            metadata = {'demo_length': 0, 'keys': [], 'sample_observation': None}
        else:
            sample_obs = demo[0]
            metadata = {
                'demo_length': len(demo),
                'keys': list(sample_obs.keys()),
                'sample_observation': {
                    key: {
                        'type': str(type(value)),
                        'shape': getattr(value, 'shape', None),
                        'dtype': str(getattr(value, 'dtype', None))
                    } for key, value in sample_obs.items()
                }
            }
        
        with open(filepath + '_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not save metadata: {e}")

# Enhanced save_demo function with metadata
def save_demo_with_metadata(demo, filepath):
    """
    Save demo and its metadata.
    
    Args:
        demo (list): List of dictionaries containing observations
        filepath (str): Path to save the demo file (without extension)
    """
    save_demo(demo, filepath)
    save_demo_metadata(demo, filepath)
# Include the save_demo functions here (from the artifact above)
# ... [save_demo, load_demo, save_demo_metadata functions] ...

def collect_human_demos(num_types_demo = NUM_KIND_OF_DEMO, demoset_size = DEMOSET_SIZE):
    """Collect human demonstrations across different configurations."""
    
    for i in range(num_types_demo):
        print(f"\n{'='*50}")
        print(f"Collecting DEMOSET {i}/{num_types_demo}")
        print(f"{'='*50}")
        
        game_interface = GameInterface()
        
        # Create directory structure
        filepath = f'human_demo/demoset{i}/'
        os.makedirs(filepath, exist_ok=True)
        
        # Save demo set configuration
        config_path = filepath + 'demo_config'
        game_interface.save_config(config_path + '.json')
        print(f"Saved configuration: {config_path}.json")
        
        # Collect demos for this configuration
        for j in range(demoset_size):
            print(f"\nStarting Demo {j+1}/{demoset_size} for demoset {i}")
            
            # Reset game for new demo
            game_interface.reset()
            game_interface.set_initial_config(config_path + '.json')
            game_interface.start_game()

            # Clear previous observations
            game_interface.observations = []

            # Run demo collection
            step_count = 0
            while game_interface.running:
                game_interface.step()

                step_count += 1
                
                # Optional: Add timeout to prevent infinite loops
                if step_count > 10000:  # Adjust as needed
                    print("Demo timeout reached, ending collection")
                    break

            # Get collected observations
            demo = game_interface.observations
            
            # Save demo with metadata
            demo_path = filepath + f'demo{j}'
            save_demo_with_metadata(demo, demo_path)
            
            print(f"Demo {j} completed: {len(demo)} observations saved")
        
        print(f"Completed demoset {i}: {demoset_size} demos collected")

def load_and_inspect_demo(demoset_id, demo_id):
    """
    Load and inspect a specific demo.
    
    Args:
        demoset_id (int): ID of the demo set
        demo_id (int): ID of the demo within the set
    """
    filepath = f'human_demo/demoset{demoset_id}/demo{demo_id}'
    
    try:
        breakpoint()

        demo = load_demo(filepath)
        print(f"Loaded demo: {len(demo)} observations")
        
        if demo:
            print("Sample observation keys:", list(demo[0].keys()))
            print("First observation:", demo[0])
        
        return demo
        
    except Exception as e:
        print(f"Failed to load demo: {e}")
        return None

def load_demo_config(demoset_id):
    """
    Load the configuration for a specific demo set.
    
    Args:
        demoset_id (int): ID of the demo set
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = f'human_demo/demoset{demoset_id}/demo_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config for demoset {demoset_id}")
        return config
        
    except Exception as e:
        print(f"Failed to load config: {e}")
        return None

def replay_demo_with_config(demoset_id, demo_id):
    """
    Replay a demo using its original configuration.
    
    Args:
        demoset_id (int): ID of the demo set
        demo_id (int): ID of the demo within the set
    """
    # Load configuration
    config = load_demo_config(demoset_id)
    if not config:
        return
    
    # Create game interface with loaded config
    game_interface = GameInterface()
    config_path = f'human_demo/demoset{demoset_id}/demo_config'
    
    if game_interface.set_initial_config(config_path + '.json'):
        print(f"Successfully loaded config for demoset {demoset_id}")
        
        # Load and replay demo
        demo = load_and_inspect_demo(demoset_id, demo_id)
        if demo:
            print(f"Replaying demo {demo_id} from demoset {demoset_id}")
            # matplotlib but kinda lazy icl
            
    else:
        print(f"Failed to load config for demoset {demoset_id}")

# Usage examples:
if __name__ == "__main__":
    # Collect all demos
    # collect_human_demos()
    # Inspect a specific demo
    # demo = load_and_inspect_demo(demoset_id=0, demo_id=0)

    collect_human_demos(num_types_demo=5,demoset_size=20)
    
    # Replay a demo with its configuration (for sanity check)
    # replay_demo_with_config(demoset_id=0, demo_id=0)