from __future__ import division
import numpy as np
from graphviz import Source
from qnetwork import *
from utils import *
import matplotlib.pyplot as plt
from env import *
import tensorflow as tf

def train(sess, env, qnet):
    
    global EXPLORATION_RATE
  
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    
    qnet.update_target()
    
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    
    for num_epi in range(MAX_EPISODES):

        s = env.reset()
        s = [list(np.unravel_index(s, env.shape))]

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(MAX_EPISODE_LEN):

            a = np.argmax(qnet.predict_q(np.reshape(s, (1, qnet.state_dim))))
    
            if np.random.rand(1) < EXPLORATION_RATE:
                s2, r, terminal, info = env.step(np.random.randint(0,qnet.action_dim))
            else:
                s2, r, terminal, info = env.step(a)
            
            s2 = list(np.unravel_index(s2, env.shape))

            replay_buffer.add(np.reshape(s, (qnet.state_dim,)), np.reshape(a, (1,)), r,
                              terminal, np.reshape(s2, (qnet.state_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = qnet.predect_target(s2_batch)

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * np.amax(target_q[k]))

                # Update the critic given the targets
                predicted_q_value, _ = qnet.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)), num_epi)

                ep_ave_max_q += np.amax(predicted_q_value)
                
                # Update target networks
                qnet.update_target()

            s = s2
            ep_reward += r

            if terminal or j == MAX_EPISODE_LEN-1:
                
                if EXPLORATION_RATE > 0.02 and terminal:
                    EXPLORATION_RATE = EXPLORATION_RATE*0.96

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j),
                    summary_vars[2]: EXPLORATION_RATE
                })

                writer.add_summary(summary_str, num_epi)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Exploration: {:.6f} '.format(int(ep_reward), \
                        num_epi, (ep_ave_max_q / float(j)), EXPLORATION_RATE))
                
                f = open("stats.txt", "ab")
                f.write("| Reward: " + str(int(ep_reward)) 
                        +" | Episode: " + str(num_epi) 
                        + " | Qmax: " + str(ep_ave_max_q / float(j)) 
                        + " | Exploration: " + str(EXPLORATION_RATE) + "\n")
                f.close()
                
                break
                
        if num_epi%10 == 0:
            state_list = []
            action_list = []
            world = np.zeros(env.shape)
            for state in range(env.nS):
                state = np.unravel_index(state, env.shape)
                action = qnet.predict_q(np.reshape(state, (1,state_dim)))
                action = np.argmax(action)
                state_list.append(state)
                action_list.append(action)
                
#             print np.reshape(action_list, env.shape)
                
            f = open("action.txt","ab")
            act_string = np.array_str(np.reshape(action_list, env.shape))
            f.write(act_string)
            f.write("---------------------------\n")
            f.close()
    

LEARNING_RATE = 0.0015
GAMMA = 0.99
# GAMMA = 0.7
TAU = 0.001
BUFFER_SIZE = 10**6
MINIBATCH_SIZE = 64
RANDOM_SEED = 272
MAX_EPISODES = 50000
MAX_EPISODE_LEN = 3000
file_appendix = time.ctime()[4:16].replace("  ","").replace(" ","_").replace(":","-")
SUMMARY_DIR = './results/tf_ddqn_' + file_appendix
SAVE_DIR = "./saved_model/" + file_appendix + "/ddqn.ckpt"
EXPLORATION_RATE = 0.7

LTL = "<>(A && <>B) && []!C"

env = CurrentWorld(LTL)

with tf.Session() as sess:
    
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    
    state_dim = 3
    action_dim = 5
    
    Qnet = QNet(sess, state_dim, action_dim, LEARNING_RATE, TAU, MINIBATCH_SIZE, SAVE_DIR)
    
    train(sess, env, Qnet)

    
