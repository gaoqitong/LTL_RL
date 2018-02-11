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
NONE = 4

class CurrentWorld(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current[:2]) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        current_rabin_state = current[-1]
        next_rabin_state = self.rabin.next_state(current, tuple(new_position))
        deadlock = False
        
        if next_rabin_state in self.rabin.deadlock:
            next_rabin_state = self.rabin.init_state
            deadlock = True

        new_position = new_position.tolist()
        next_rabin_state = int(next_rabin_state)
        
        new_state_3d = tuple(new_position + [next_rabin_state])
        new_state = np.ravel_multi_index( new_state_3d, self.shape)
        
        is_done = new_state_3d in self.terminal_states
        
        if is_done:
            return [(1.0, new_state, 100, is_done)]
        elif next_rabin_state in self.rabin.accept:
            return [(1.0, new_state, 10, is_done)]
        elif deadlock == True:
            return [(1.0, new_state, -100, is_done)]
        elif next_rabin_state in self.rabin.reject:
            return [(1.0, new_state, -10, is_done)]
        return [(1.0, new_state, -1, is_done)]


    def __init__(self, ltl):
        self.start_coord = (2, 2)
        self.terminal_coord = (19, 19)
        self.shape = (20, 20)
        
        ap_dict = {"B":[(6, 0)], "A":[(1, 12)], 
                   "C":[(0, 0), (0, 8), (1, 8), (1, 2), (1, 3), (8, 3), (5, 3),
                        (6, 3), (7, 3), (2, 7), (2, 8), (2, 9), (15, 6), (15, 1), 
                        (15, 2), (15, 3), (15, 4), (15, 5), (15, 7), (10, 15),
                        (11, 15), (12, 15), (13, 15), (14, 15), (15, 15), (16, 15), 
                        (17, 15), (18, 15)], 
                   "D":[(18, 5)]}
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
            P[s][NONE] = self._calculate_transition_prob(position, [0, 0])

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


