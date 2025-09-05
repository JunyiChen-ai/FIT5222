"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys


#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!", e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0

#########################
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
# The path should avoid conflicts with existing paths.
#########################

# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param agent_id The id of given agent
# @param existing_paths A list of lists of locations indicate existing paths. The index of each location is the time that
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, agent_id: int, existing_paths: list, max_timestep: int):
    ############
    # Space-Time A* algorithm implementation for single agent pathfinding
    # Considers existing paths as dynamic obstacles to avoid conflicts
    ############
    import heapq
    import time
    start_time = time.time()
    
    # Helper function to calculate Manhattan distance heuristic
    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Helper function to get next position based on direction
    def get_next_position(pos, direction):
        x, y = pos
        if direction == Directions.NORTH:
            return (x - 1, y)
        elif direction == Directions.EAST:
            return (x, y + 1)
        elif direction == Directions.SOUTH:
            return (x + 1, y)
        elif direction == Directions.WEST:
            return (x, y - 1)
        return pos
    
    # Check if position is within bounds
    def is_valid_position(pos):
        x, y = pos
        return 0 <= x < rail.height and 0 <= y < rail.width
    
    # Check if there's a conflict with existing paths at given time and position
    def has_conflict(pos, timestep, prev_pos=None):
        for path in existing_paths:
            # Position conflict: another agent occupies the same position at same time
            if timestep < len(path) and path[timestep] == pos:
                return True
            
            # Swap conflict: two agents swap positions
            if (prev_pos is not None and timestep > 0 and
                timestep < len(path) and timestep-1 < len(path) and
                path[timestep] == prev_pos and path[timestep-1] == pos):
                return True
                
        return False
    
    # Priority queue: (f_score, g_score, timestep, position, direction, path)
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), 0, 0, start, start_direction, [start]))
    
    # Set to track visited states (timestep, position, direction)
    visited = set()
    
    while pq:
        f_score, g_score, timestep, current_pos, current_dir, path = heapq.heappop(pq)
        
        # Check if we reached the goal
        if current_pos == goal:
            end_time = time.time()
            if debug:
                print(f"Agent {agent_id}: Path found in {end_time - start_time:.4f} seconds, visited {len(visited)} states")
            return path
        
        # Skip if we've already visited this state
        state = (timestep, current_pos, current_dir)
        if state in visited:
            continue
        visited.add(state)
        
        # Check if we exceeded max timestep
        if timestep >= max_timestep:
            continue
        
        # Get valid transitions from current position and direction
        transitions = rail.get_transitions(current_pos[0], current_pos[1], current_dir)
        
        # Option 1: Wait at current position
        next_timestep = timestep + 1
        if not has_conflict(current_pos, next_timestep):
            new_path = path + [current_pos]
            new_f_score = next_timestep + heuristic(current_pos, goal)
            heapq.heappush(pq, (new_f_score, next_timestep, next_timestep, current_pos, current_dir, new_path))
        
        # Option 2: Move to next position
        for new_dir in range(4):
            if transitions[new_dir]:
                next_pos = get_next_position(current_pos, new_dir)
                
                # Check if next position is valid
                if not is_valid_position(next_pos):
                    continue
                
                # Check for conflicts
                if has_conflict(next_pos, next_timestep, current_pos):
                    continue
                
                # Check if we've already visited this state
                next_state = (next_timestep, next_pos, new_dir)
                if next_state in visited:
                    continue
                
                # Calculate new scores
                new_g_score = next_timestep
                new_h_score = heuristic(next_pos, goal)
                new_f_score = new_g_score + new_h_score
                
                # Add to priority queue
                new_path = path + [next_pos]
                heapq.heappush(pq, (new_f_score, new_g_score, next_timestep, next_pos, new_dir, new_path))
    
    # If no path found, return empty list as specified
    if debug:
        print(f"Agent {agent_id}: No path found after {time.time() - start_time:.4f} seconds")
    return []


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,2)


















