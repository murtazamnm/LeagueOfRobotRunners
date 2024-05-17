import MAPF
from typing import Dict, List, Tuple, Set
from queue import PriorityQueue
import numpy as np
import random

#map used
######################
# Warehouse_small_10 #
######################

class pyMAPFPlanner:
    def __init__(self, pyenv=None) -> None:
        self.agitation = {}
        self.paths = {}
        if pyenv is not None:
            self.env = pyenv.env
        self.agitation = []
        self.paths = []
        print("pyMAPFPlanner created!  python debug")
    def initialize(self, preprocess_time_limit: int):
        for i in range(self.env.num_of_agents):
                self.agitation.append(0)  # Initialize agitation level for each agent
                self.paths.append([])
        print("planner initialize done... python debug")
        return True

    def plan(self, time_limit):
        """_summary_

        Return:
            actions ([Action]): the next actions

        Args:
            time_limit (_type_): _description_
        """
        return self.local_repair_a_star(time_limit)
    def getManhattanDistance(self, loc1: int, loc2: int, ag_lvl) -> int:
        loc1_x = loc1 // self.env.cols
        loc1_y = loc1 % self.env.cols
        loc2_x = loc2 // self.env.cols
        loc2_y = loc2 % self.env.cols
        if ag_lvl is not None:
            base_heuristic = abs(loc1_x-loc2_x)+abs(loc1_y-loc2_y)
            random_factor = random.uniform(0, 2) * ag_lvl  # random factor and agitation level of agent
            return base_heuristic * (1 + random_factor)
        return abs(loc1_x - loc2_x) + abs(loc1_y - loc2_y)
    
    def reset_agitation(self):
        for i in range(len(self.agitation)):
            self.agitation[i] = 0
            
    def validateMove(self, loc: int, loc2: int) -> bool:
        loc_x = loc // self.env.cols
        loc_y = loc % self.env.cols
        if loc_x >= self.env.rows or loc_y >= self.env.cols or self.env.map[loc] == 1:
            return False
        loc2_x = loc2 // self.env.cols
        loc2_y = loc2 % self.env.cols
        if abs(loc_x - loc2_x) + abs(loc_y - loc2_y) > 1:
            return False
        return True

    def getNeighbors(self, location: int, direction: int):
        neighbors = []
        candidates = [location + 1, location + self.env.cols, location - 1, location - self.env.cols]
        forward = candidates[direction]
        new_direction = direction
        if (forward >= 0 and forward < len(self.env.map) and self.validateMove(forward, location)):
            neighbors.append((forward, new_direction))
        new_direction = direction - 1
        if (new_direction == -1):
            new_direction = 3
        neighbors.append((location, new_direction))
        new_direction = direction + 1
        if (new_direction == 4):
            new_direction = 0
        neighbors.append((location, new_direction))
        return neighbors

    def get_surrounding_agents(self, location):
        surrounding_agents = []

        for agent_idx, agent in enumerate(self.env.curr_states):
            if agent.location == location:
                surrounding_agents.append(agent_idx)

        # Check for agents in the surrounding cells
        adjacent_locations = [location + offset for offset in [1, self.env.cols, -1, -self.env.cols]]
        for adjacent_location in adjacent_locations:
            for agent_idx, agent in enumerate(self.env.curr_states):
                if agent.location == adjacent_location:
                    surrounding_agents.append(agent_idx)

        return surrounding_agents
    
    def detect_conflicts(self):
        conflicts = set()
        for i in range(self.env.num_of_agents):
            if self.paths[i] and self.paths[i][0][0] != self.env.curr_states[i].location: # If the agent is moving
                location = self.env.curr_states[i].location
                candidates = [location + 1, location + self.env.cols, location - 1, location - self.env.cols]
                forward = candidates[self.env.curr_states[i].orientation]
                surrounding_agents = self.get_surrounding_agents(forward)
                for agent_j in surrounding_agents:
                    if agent_j == i:
                        continue
                    if ((self.env.curr_states[agent_j].location == forward and
                        ((not self.paths[agent_j] or self.paths[agent_j][0][0] == forward) or
                        (self.paths[agent_j] and self.paths[agent_j][0][0] == location))) or
                        (self.paths[agent_j] and self.paths[agent_j][0][0] == forward)):
                        conflicts.add((i, agent_j))
        return conflicts

    def single_agent_plan(self, start: int, start_direct: int, end: int, agitation_level: int):
        open_list = PriorityQueue()
        s = (start, start_direct, 0, self.getManhattanDistance(start, end,agitation_level))
        open_list.put([0, s])
        all_nodes = dict()
        close_list = set()
        parent = {(start, start_direct): None}
        all_nodes[start*4+start_direct] = s
        while not open_list.empty():
            curr = (open_list.get())[1]
            close_list.add(curr[0]*4+curr[1])
            if curr[0] == end:
                curr = (curr[0], curr[1])
                path = []
                while curr != None:
                    path.append(curr)
                    curr = parent[curr]
                path.pop()
                path.reverse()
                return path

            neighbors = self.getNeighbors(curr[0], curr[1])
            for neighbor in neighbors:
                if (neighbor[0]*4+neighbor[1]) in close_list:
                    continue
                next_node = (neighbor[0], neighbor[1], curr[2]+1,
                             self.getManhattanDistance(neighbor[0], end, agitation_level))
                parent[(next_node[0], next_node[1])] = (curr[0], curr[1])
                open_list.put([next_node[3]+next_node[2], next_node])
        return [(end, start_direct)]  
    
    def local_repair_a_star(self, time_limit):
        print("Planning with local repair A*")
        actions = [MAPF.Action.W for i in range(len(self.env.curr_states))]
        for i in range(self.env.num_of_agents):
            start = self.env.curr_states[i].location
            start_direct = self.env.curr_states[i].orientation
            goal = self.env.goal_locations[i][0][0] if self.env.goal_locations[i] else None

            if goal is not None:
                path = self.single_agent_plan(start, start_direct, goal, self.agitation[i])
                self.paths[i] = path
            else:
                path = [(start, start_direct)]
                self.paths[i] = path

        conflicts = self.detect_conflicts()
        while conflicts:
            for (i, j) in conflicts:
                # Increase agitation level for conflicted agents
                self.agitation[i] += 1
                self.agitation[j] += 1
                # Replan paths for conflicted agents
                for agent_id in (i, j):
                    start = self.env.curr_states[agent_id].location
                    start_direct = self.env.curr_states[agent_id].orientation
                    goal = self.env.goal_locations[agent_id][0][0]
                    self.paths[agent_id] = self.single_agent_plan(start, start_direct, goal, self.agitation[agent_id])
            conflicts = self.detect_conflicts()
        # Set actions based on the resolved paths
        for i in range(len(self.env.curr_states)):
            path = self.paths[i]
            start = self.env.curr_states[i].location
            start_direct = self.env.curr_states[i].orientation
            if path:
                if path[0][0] != start:
                    actions[i] = MAPF.Action.FW
                elif path[0][1] != start_direct:
                    incr = path[0][1] - start_direct
                    if incr == 1 or incr == -3:
                        actions[i] = MAPF.Action.CR
                    elif incr == -1 or incr == 3:
                        actions[i] = MAPF.Action.CCR

        return np.array(actions, dtype=int)
    

if __name__ == "__main__":
    test_planner = pyMAPFPlanner()
    test_planner.initialize(100)
