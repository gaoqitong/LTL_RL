from __future__ import division
import tensorflow as tf
import numpy as np


class QNet(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size, save_path):
        self.sess = sess
        self.learning_rate = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.save_path = save_path
        
        self.inputs, self.q_values, self.a_predict = self.build_net()
        self.net_params = tf.trainable_variables()
        
        self.target_inputs, self.target_q_values, self.target_a_predict = self.build_net()
        self.target_net_params = tf.trainable_variables()[len(self.net_params):]
        
        self.update_target_net_params = [self.target_net_params[i]
                                         .assign(tf.multiply(self.tau, self.net_params[i])
                                                 + tf.multiply((1.-self.tau), self.target_net_params[i]) ) 
                                         for i in range(len(self.target_net_params))]
        
        self.true_q_value = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        
        gather_indices = tf.range(self.batch_size) * tf.shape(self.q_values)[1] + tf.reshape(self.action, [-1])
        self.action_correlated_q = tf.gather(tf.reshape(self.q_values,[-1]), gather_indices)
        
        self.loss = tf.losses.mean_squared_error(tf.reshape(self.true_q_value, [-1]), self.action_correlated_q)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.saver = tf.train.Saver()
        self.last_num_epi = -1
        
    def build_net(self):
        s_inputs = tf.placeholder(shape = [None, self.state_dim], dtype = tf.float32)
        W1 = tf.Variable(tf.random_uniform([self.state_dim, 400], 0, 0.1))
        B1 = tf.Variable(tf.zeros([400]))
        L1 = tf.add(tf.matmul(s_inputs, W1), B1)
        L1 = tf.layers.batch_normalization(L1)
        L1 = tf.nn.relu(L1)
        W2 = tf.Variable(tf.random_uniform([400, 300], 0, 0.1))
        B2 = tf.Variable(tf.zeros([300]))
        L2 = tf.add(tf.matmul(L1, W2), B2)
        L2 = tf.layers.batch_normalization(L2)
        L2 = tf.nn.relu(L2)
        W3 = tf.Variable(tf.random_uniform([300, self.action_dim], 0, 0.01))
#         B3 = tf.Variable(tf.random_uniform([self.action_dim], -0.003, 0.003))
#         q_values = tf.add(tf.matmul(L2, W3), B3)
        q_values = tf.matmul(L2, W3)  
        a_predict = tf.argmax(q_values,1)
        
        regularizer = tf.contrib.layers.l2_regularizer(0.01)
        tf.contrib.layers.apply_regularization(regularizer,[W1, B1, W2, B2, W3])
        return s_inputs, q_values, a_predict
    
    def train(self, states, action, true_q, num_epi):
        if num_epi%20 == 0 and num_epi!=self.last_num_epi:
            self.saver.save(self.sess, self.save_path)
            print "DDQN Saved"
            self.last_num_epi = num_epi
            
        return self.sess.run([self.q_values, self.optimizer], 
                             feed_dict={self.inputs: states, self.true_q_value: true_q, self.action: action})
    
    def predict_q(self, states):
        return self.sess.run(self.q_values, feed_dict={self.inputs: states})
    
    def predict_a(self, states):
        return self.sess.run(self.a_predict, feed_dict={self.inputs: states})
    
    def predect_target(self, states):
        return self.sess.run(self.target_q_values, feed_dict={self.target_inputs: states})
    
    def update_target(self):
        self.sess.run(self.update_target_net_params)
        
        