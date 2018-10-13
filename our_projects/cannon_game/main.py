import tensorflow as tf
import game
import numpy as np
import random

from collections import deque

import gameObject 
import cnn

from game_reinforcement_env import init_env_data, update_env_by_action
from neural_network_utils import save_and_load_network, train_network_by_batch
from reinforcement_utils import act_with_greedy_policy

ACTIONS = 5 # 유효한 액션 수 (left, right, up, down)
GAMMA = 0.99 # decay rate (강화학습에 있는 개념)
OBSERVE = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000 # 이전 행동을 기억하는 메모리 크기(행동 갯수)
BATCH = 32 # 배치 크기
FRAME_PER_ACTION = 1
GAME = "rl game"


def print_info(t, epsilon, action_index, r_t, readout_t):
    """학습에 관한 변수 값들을 출력합니다"""
    state = ""
    if t <= OBSERVE:
        state = "observe"
        print("observe")
    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        state = "explore"
    else:
        state = "train"
    print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
    

def trainNetwork(s, readout, h_fc1, sess):
    
    a = tf.placeholder("float", [None, 5])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = game.GameState()
    D = deque()
    s_t = init_env_data(game_state)
    saver = save_and_load_network(sess)
    epsilon = INITIAL_EPSILON
    terminal = False
    t = 0
    
    while True:
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([5])
        action_index = 0

        if t % FRAME_PER_ACTION == 0:
            act_with_greedy_policy(epsilon, readout_t, a_t)
        else:
            a_t[0] = 1 # do nothing

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        s_t1, r_t, terminal = update_env_by_action(game_state, s_t, a_t)
        
        D.append((s_t, a_t, r_t, s_t1, terminal))

        if len(D) > REPLAY_MEMORY:
            D.popleft()


        if t > OBSERVE:
            # D 큐에서 학습에 필요한  데이터를 샘플링함
            minibatch = random.sample(D, BATCH)
            train_network_by_batch(minibatch, readout, train_step, s, a, y)
            
        s_t = s_t1
        t += 1
        
        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'save_networks/' + GAME + '-dqn', global_step = t)
        
        print_info(t, epsilon, action_index, r_t, readout_t)

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = cnn.createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()








