from __future__ import division
import time
import matplotlib.pyplot as plt
import numpy as np


def plot_path(env, LTL, saved_path):
    env = CurrentWorld(LTL)

    state_dim = 3
    action_dim = 4
    state = env.reset()
    done = False

    while not done:
        tf.reset_default_graph()
        with tf.Session() as sess:
            Qnet = QNet(sess, state_dim, action_dim, LEARNING_RATE, TAU, MINIBATCH_SIZE, SAVE_DIR)
            state = np.reshape(list(np.unravel_index(state, env.shape)), (1, state_dim))
            state_for_plot = tuple(state[0][:2])
            render(env, state_for_plot)
            action = Qnet.predict_a_from_save(state, saved_path)
            next_state,_,done,_ = env.step(action[0])
            state = next_state

def render(env, state):
    world = np.zeros((env.shape[0], env.shape[1]))
    color_dict = {ap: color+1 for color, ap in enumerate(env.ap_dict.keys())}
    for i in env.coord_dict.keys():
        if len(env.coord_dict[i]) >=1:
            world[i] = color_dict[env.coord_dict[i][0]]
    world[state] = len(env.ap_dict) + 1
    fig, ax = plt.subplots()
    ax.clear()
    ax.imshow(world)
    for i in env.ap_dict.keys():
        for j in env.ap_dict[i]:
            ax.annotate(i, xy=(j[1] - 0.13, j[0] + 0.13), fontsize=20, color=(1,1,1))
    ax.annotate("R", xy=(state[1] - 0.13, state[0] + 0.13), fontsize=20, color=(1,0,0))
    plt.pause(0.0001)