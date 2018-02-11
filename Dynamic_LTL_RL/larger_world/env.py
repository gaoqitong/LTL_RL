from __future__ import division
import numpy as np
from collections import defaultdict
import gym
from gym.envs.toy_text import discrete
from utils import *


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
# NONE = 4

class CurrentWorld(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
#         delta_list = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]
        delta_list = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        new_position_candidates = [np.array(current[:2]) + np.array(delta)]
        new_position_candidates += [np.array(current[:2]) + np.array(i) for i in delta_list if i != delta]
        new_positions = [self._limit_coordinates(i).astype(int) for i in new_position_candidates]
        current_rabin_state = current[-1]
        next_rabin_state = [self.rabin.next_state(current, tuple(i)) for i in new_positions]
        deadlock = [False for i in range(len(new_positions))]
        
        for i in range(len(new_positions)):
            if next_rabin_state[i] in self.rabin.deadlock:
                next_rabin_state[i] = self.rabin.init_state
                deadlock[i] = True

        new_positions = [i.tolist() for i in new_positions]
        next_rabin_state = [int(i) for i in next_rabin_state]
        
        new_state_3d = [tuple(new_positions[i] + [next_rabin_state[i]]) for i in range(len(new_positions))]
        new_state = [np.ravel_multi_index( i, self.shape) for i in new_state_3d]
        
        is_done = [i in self.terminal_states for i in new_state_3d]

        reward_list = []

        for i in range(len(new_state)):
            if is_done[i]:
                reward_list += [100]
            elif next_rabin_state[i] in self.rabin.accept:
                reward_list += [10]
            elif deadlock[i] == True:
                reward_list += [-100]
            elif next_rabin_state[i] in self.rabin.reject:
                reward_list += [-10]
            else:
                reward_list += [-1]



        return [(0.7, new_state[0], reward_list[0], is_done[0]), 
                (0.1, new_state[1], reward_list[1], is_done[1]), 
                (0.1, new_state[2], reward_list[2], is_done[2]), 
                (0.1, new_state[3], reward_list[3], is_done[3])] 
#                 ,(0.1, new_state[4], reward_list[4], is_done[4])]


    def __init__(self, ltl):
        self.start_coord = (4, 1)
        self.terminal_coord = (9, 9)
        self.shape = (10, 10)
        
        ap_dict = {"A":[(2, 7)], "B":[(5, 2)], "C":[(i, i) for i in range(2, 7)]}
        ap_dict["T"] = [self.terminal_coord]

        coord_dict = defaultdict(lambda x: "")

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                coord_dict[(i,j)] = []

        for i in ap_dict.items():
            for j in i[1]:
                coord_dict[j] += i[0]
        
        self.coord_dict = coord_dict
        self.ap_dict = ap_dict
                
        ltl = ltl + " && <>[] T"
        
        self.rabin = Rabin_Automaton(ltl, self.coord_dict)
        self.shape = (self.shape[0], self.shape[1], self.rabin.num_of_nodes)
        
        nS = np.prod(self.shape)
        nA = 4
        
        self.start_state = tuple( list(self.start_coord) + [self.rabin.init_state] )
        self.terminal_states = [tuple(list(self.terminal_coord) + [i]) for i in self.rabin.accept]        
        
        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])
#             P[s][NONE] = self._calculate_transition_prob(position, [0, 0])

#       Set start point
        isd = np.zeros(nS)
        isd[np.ravel_multi_index(self.start_state, self.shape)] = 1.0

        super(CurrentWorld, self).__init__(nS, nA, P, isd)
        


def limit_coordinates(coord,world):
    coord[0] = min(coord[0], np.shape(world)[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], np.shape(world)[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord

