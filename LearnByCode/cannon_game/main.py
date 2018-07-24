import tensorflow as tf
import game
import numpy as np

from collections import deque

import gameObject 
import cnn

from game_reinforcement_env import init_env_data, update_env_by_action
from neural_network_utils import get_cost, save_and_load_network, train_network_by_batch
from reinforcement_utils import act_with_greedy_policy

ACTIONS = 2 # 유효한 액션 수 (뛰기, 그대로 있기)
GAMMA = 0.99 # decay rate (강화학습에 있는 개념)
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # 마지막 epsilon 값 (epsilon은 모험을 하는 정도)
INITIAL_EPSILON = 0.0001 # 초기 epsilon 값
REPLAY_MEMORY = 50000 # 이전 행동을 기억하는 메모리 크기(행동 갯수)
BATCH = 32 # 배치 크기
FRAME_PER_ACTION = 1


def trainNetwork(s, readout, h_fc1, sess):
    
    cost = get_cost(readout)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    game_state = game.GameState()
    D = deque()
    s_t = init_env_data(game_state)
    saver = save_and_load_network(sess)

    epsilon = INITIAL_EPSILON
    t = 0


    while true:
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([2])
        action_index = 0

        if not game_state.cannon.is_inside():
            act_with_greedy_policy(epsilon, readout_t, a_t)
        else:
            a_t[0] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        s_t1, r_t, terminal = update_env_by_action(game_state, s_t, a_t)














