#!/usr/bin/env python
from __future__ import print_function
# 학습을 위한 툴인 텐서플로우를 불러옵니다
import tensorflow as tf


# 파일 경로 접근을 위한 sys라이브러리
import sys
sys.path.append("game/")

#게임 폴더를 불러옵니다.
import wrapped_flappy_bird as game
import numpy as np

from collections import deque

#import network
import convolution as cnn

from game_reinforcement_env import init_env_data, update_env_by_action
from neural_network_utils import get_cost, save_and_load_network, train_network_by_batch
from reinforcement_utils import act_with_greedy_policy

GAME = 'bird' # 게임 이름
ACTIONS = 2 # 유효한 액션 수 (뛰기, 그대로 있기)
GAMMA = 0.99 # decay rate (강화학습에 있는 개념)
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # 마지막 epsilon 값 (epsilon은 모험을 하는 정도)
INITIAL_EPSILON = 0.0001 # 초기 epsilon 값
REPLAY_MEMORY = 50000 # 이전 행동을 기억하는 메모리 크기(행동 갯수)
BATCH = 32 # 배치 크기
FRAME_PER_ACTION = 1

def print_info(t, epsilon, action_index, r_t, readout_t):
    """학습에 관한 변수 값들을 출력합니다"""
    state = ""
    if t <= OBSERVE:
        state = "observe"
    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        state = "explore"
    else:
        state = "train"

    print("TIMESTEP", t, "/ STATE", state, \
        "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
        "/ Q_MAX %e" % np.max(readout_t))


def trainNetwork(s, readout, h_fc1, sess):

    cost = get_cost(readout)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # game을 불러옴
    game_state = game.GameState()

    # observe 상태일때 저장해서 모아둔 replay를 저장하는 큐
    D = deque()

    s_t = init_env_data(game_state)
    saver = save_and_load_network(sess)
    
    # start training
    epsilon = INITIAL_EPSILON
    t = 0

    # 무한 반복
    while "flappy bird" != "angry bird":
        # epsilon greedy 구현

        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        print(readout_t)
        a_t = np.zeros([ACTIONS])
        action_index = 0

        if t % FRAME_PER_ACTION == 0:
            act_with_greedy_policy(epsilon, readout_t, a_t)

        else:
            a_t[0] = 1 # do nothing


        # epsilon 감소

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        s_t1, r_t, terminal = update_env_by_action(game_state, s_t, a_t)

        # 처리된 데이터들을 큐에 넣음
        D.append((s_t, a_t, r_t, s_t1, terminal))

        if len(D) > REPLAY_MEMORY:
            D.popleft()


        # observe 상태가 끝나면 학습 시작

        if t > OBSERVE:
            # D 큐에서 학습에 필요한  데이터를 샘플링함
            minibatch = random.sample(D, BATCH)
            train_network_by_batch(minibatch)

        # timestep 이동 (상태 업데이트)
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)
        
        print_info(t, epsilon, action_index, r_t, readout_t)



def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = cnn.createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
