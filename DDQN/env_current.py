from __future__ import division
import numpy as np
from copy import deepcopy
from collections import defaultdict
import itertools
from numpy.linalg import norm
import gym
from gym.envs.toy_text import discrete
import sys
from scipy.spatial.distance import euclidean

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

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.multiply(1, self.winds_convert_dict[winds[tuple(current)]])
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == self.terminal_state
        if is_done:
            return [(1.0, new_state, 0, is_done)]
#         if delta == [0,0]:
#             return [(1.0, new_state, -1, is_done)]
        return [(1.0, new_state, -1, is_done)]
#         if is_done:
#             return [(1.0, new_state, 10000000, is_done)]
#         if delta == [0,0]:
#             return [(1.0, new_state, -euclidean(np.unravel_index(new_state,self.shape),self.terminal_state), is_done)]
#         return [(1.0, new_state, -100*euclidean(np.unravel_index(new_state,self.shape),self.terminal_state), is_done)]



    def __init__(self):
        self.shape = (20, 20)

        nS = np.prod(self.shape)
        nA = 5

        # Wind strength
        winds = np.zeros(self.shape)
#         winds = add_square_current((10,10),5,self.shape)
#         winds += add_square_current((10,10),4,self.shape)
#         winds += add_square_current((10,10),3,self.shape)
#         winds += add_square_current((10,10),2,self.shape)
        
#         for i in range(1,6):
#             winds += add_square_current((30,30),i,self.shape)
        self.winds = winds
        self.winds_convert_dict = {0:[0,0],1:[0,1],2:[1,0],3:[0,-1],4:[-1,0]}
        self.start_state = (12,5)
        self.terminal_state = (15,18)
        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)
            P[s][NONE] = self._calculate_transition_prob(position, [0, 0], winds)

#       Set start point
        isd = np.zeros(nS)
        isd[np.ravel_multi_index(self.start_state, self.shape)] = 1.0

        super(CurrentWorld, self).__init__(nS, nA, P, isd)
        
    def show_img(self):
        world = deepcopy(self.winds)
        world[np.unravel_index(self.s,self.shape)] = 7
        plt.imshow(world)


def limit_coordinates(coord,world):
    coord[0] = min(coord[0], np.shape(world)[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], np.shape(world)[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord

def add_square_current(center,radius,world_shape,direction='cw'):
    world = np.zeros(world_shape)
    vertical_boundary = [center[0]-radius, center[0]+radius]
    vertical_boundary = limit_coordinates(vertical_boundary,world)
    horizontal_boundary = [center[1]-radius, center[1]+radius]
    horizontal_boundary = limit_coordinates(horizontal_boundary,world)
    vertical_index = np.linspace(vertical_boundary[0],vertical_boundary[1],vertical_boundary[1]+1-vertical_boundary[0])
    horizontal_index = np.linspace(horizontal_boundary[0],horizontal_boundary[1],horizontal_boundary[1]+1-horizontal_boundary[0])
    xx, yy = np.meshgrid(vertical_index,horizontal_index)
    x_range = np.array([np.amin(xx),np.amax(xx)]).astype(int)
    y_range = np.array([np.amin(yy),np.amax(yy)]).astype(int)
    xxyy = zip(xx,yy)
    coord = []
    for i,j in xxyy:
        coord += zip(i,j)
    coord = np.array([np.array(i).astype(int) for i in coord])
    boundary_index = []
    for i,j in coord:
        if i in x_range or j in y_range:
            boundary_index += [[i,j]]
#   1:right, 2:down, 3:left, 4:up, 0:no current
    if direction == 'cw':
        world[y_range[0]:y_range[1]+1,x_range[0]] = 4
        world[y_range[0]:y_range[1],x_range[1]] = 2
        world[y_range[0],x_range[0]:x_range[1]] = 1
        world[y_range[1],x_range[0]+1:x_range[1]+1] = 3
    elif direction == 'ccw':
        world[y_range[0]:y_range[1]+1,x_range[0]] = 2
        world[y_range[0]:y_range[1],x_range[1]] = 4
        world[y_range[0],x_range[0]:x_range[1]] = 3
        world[y_range[1],x_range[0]+1:x_range[1]+1] = 1
    else:
        raise AttributeError("Direction input is not correct, please use 'cw' or 'ccw'")
    return world
