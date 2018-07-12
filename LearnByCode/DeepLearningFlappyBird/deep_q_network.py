#!/usr/bin/env python
from __future__ import print_function
# 학습을 위한 툴인 텐서플로우를 불러옵니다
import tensorflow as tf

# 게임 이미지데이터를 처리하기 위한 opencv 라이브러리 입니다.
import cv2

# 파일 경로 접근을 위한 sys라이브러리
import sys
sys.path.append("game/")

#게임 폴더를 불러옵니다.
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

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

def weight_variable(shape):
    """가중치 값 초기화"""
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias 값 초기화"""
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    """convolution 하는 함수
    x = 인풋데이터, W = 가중치, stride = window가 한번 움직일때 움직이는 정도"""
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    """max pooling하는 함수
    x = conv2d함수에 의해 convolve 된 값"""
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    """네트워크 각각에 해당되는 가중치(w)와 bias 값들에 대한 초기화를 한번에 해주는 함수
    총 3개의 convolution 네트워크를 가지고 있는 네트워크이다 """

    #첫번째 convolution 레이어에 필요한 w와 bias 생성
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    #두번째 convolution 레이어에 필요한 w와 bias 생성
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    #세번째 convolution 레이어에 필요한 w와 bias 생성
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    #첫번째 fully connect 레이어에 필요한 w와 bias 생성
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    #두번째 fully connect 레이어에 필요한 w와 bias 생성
    """ 이 레이어는 가장 말단이며 Flappy bird가 날지 가만히 있을지를 알려준다"""
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    #데이터를 받는 변수
    """ 80 * 80 크기의 사진데이터를 4개씩 NONE개 만큼 담는 placeholder 변수
	NONE표시는 몇개인지 정해지지 않았을때 쓴다"""
    s = tf.placeholder("float", [None, 80, 80, 4])

    # 히든 레이어 생성
    """ 인풋데이터 s를 conv2d함수로 convolve시키고 Wx + b 형태의 모델로 표현한뒤 
    relu activation시킨다"""
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    """ relu activation 시킨 값들을 max pooling 시키고 같은 형태의 모델로 표현한뒤 relu"""
    h_pool1 = max_pool_2x2(h_conv1)

    """ 위에 레이어에서 처리된 데이터를 또다시 convolve 시키고 같은 형태의 모델로 표현한뒤 relu"""
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    """ 위에 레이어에서 처리된 데이터를 또다시 convolve 시키고 같은 형태의 모델로 표현한뒤 relu"""
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    """ 처리된 데이터의 shape를 1열로 쭉 나열한다""" 
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    """ 나열된 데이터에 다시한번 Wx + b 모델에 넣고 relu activation"""
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 새의 행동을 결정하는 레이어 
    """ 똑같이 Wx + b 모델에 넣고 출력값을 통해 날지 안날지 결정"""
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    """ s = 데이터, readout = 뛸지 안뛸지 결정하도록 하는 데이터 값(아직 activation값임), 
    h_fc1 = readout 레이어를 통과하기 전 데이터"""

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    """ 위에서 만든 레이어 들을 본격적으로 학습시키는 함수"""
    """ readout ==(1,2)"""
    # cost 함수 설정
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])

    """ a * readout(wx + b) == 예측값(q), y == replay batch에서 나온값"""
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)

    """ 예측값과 batch에서 나온 값들의 오류로 학습"""
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # game을 불러옴
    game_state = game.GameState()

    # observe 상태일때 저장해서 모아둔 replay를 저장하는 큐
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # 초기 상태를 점프뛰지 않는 상태로 두고 이미지를 80 * 80 * 4 형태로 preprocessing함

    """ do_nothing == [1, 0]"""
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    """ game_state.frame_step => 행동 input을 기반으로 게임에 변화를 줌
    x_t == 이미지 데이터, r_0 == reward, terminal == 게임이 끝났는지 여부"""
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    """ 이미지를 흑백으로, 80 * 80 크기로 잘라냄"""
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)

    """ 이미지 임계처리"""
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

    """ 80 * 80 * 4 형태로 만듬
    80 * 80 크기의 이미지 4쌍이 한 세트""" 
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 네트워크 저장 및 로딩
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0

    # 무한 반복
    while "flappy bird" != "angry bird":
        # epsilon greedy 구현

        """ 
            readout_t == 신경망에 이미지를 넣어 계산한 행동 결과
            ex) readout_t == [0.56, 2.56]
        """
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]

        """ 변수 초기화 
        a_t => 행동 담는 변수
        action_index => 0 or 1 
        """
        a_t = np.zeros([ACTIONS])
        action_index = 0

        """ 설정해둔 프레임마다 행동 결정"""
        if t % FRAME_PER_ACTION == 0:

            """ epsilon 확률로 랜덤하게 행동"""
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1

            """ 아니면 예측된 값으로 행동"""
            else:
                """ 
                    action_index == 0이면 가만히 있기
                    action_index == 1이면 점프뛰기
                """
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

        """ 설정해둔 프레임이 아니면 가만히 있기"""
        else:
            a_t[0] = 1 # do nothing

        """
            a_t == [0, 1] => 점프
            a_t == [1, 0] => 가만히 있기
        """

        # epsilon 감소

        """ 처음에는 모험을 하는 지수를 늘리기 위해 epsilon값이 크지만
            서서히 학습해가면서 epsilon값을 줄이면서 모험하는 빈도를 낮춤"""
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 계산된 행동을 하고 그 행동에 따른 게임 상태를 받아오고 데이터 처리함

        """ x_t1_colored = 게임 이미지(컬러), 
            r_t = 행동에 따른 보상값,
            terminal = 끝난 여부
        """
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        """ 이전 데이터 처리와 같음"""
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 처리된 데이터들을 큐에 넣음
        D.append((s_t, a_t, r_t, :s_t1, terminal))

        """ 큐가 가득차면 제일 오래된 데이터 방출"""
        if len(D) > REPLAY_MEMORY:
            D.popleft()


        # observe 상태가 끝나면 학습 시작

        """ OBSERVE상태에서는 학습을 하지않고
            랜덤하게 행동을 취해서 그에 따른 데이터들을 
            수집하고 training 상태에서 데이터들을 사용하여
            학습을 시작함
        """
        if t > OBSERVE:
            # D 큐에서 학습에 필요한  데이터를 샘플링함
            minibatch = random.sample(D, BATCH)

            # batch로 변수 파싱
            """
                s_j_batch == 행동전 게임 상태,
                a_batch == 행동,
                r_batch == 보상,
                s_j1_batch == 행동에 따른 게임상태
            """
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []

            """ batch 데이터로 예측한 readout 값(wx + b)"""
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})

            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 상태가 terminal이면 보상값, 아니면 계산값 (q learning 계산값)
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # 계산값과 batch에서 얻은 데이터 차이를 통한 학습
            train_step.run(feed_dict = {
                y : y_batch, """ q learning으로 계산한 값"""
                a : a_batch, """ batch에서 가져온 행동값"""
                s : s_j_batch} """ batch에서 가져온 상태값"""
            )

        # timestep 이동 (상태 업데이트)
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
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
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
