import tensorflow as tf
import numpy as np

GAMMA = 0.99
# 상태에 따른 행동 값을 압축? 시켜서 전달
#    readout : 상태에 따른 행동 값
# prediction : 예측값
# y : 정답 값 (실 정답은 아니고, 가지고 있던 값)
def get_prediction(readout):
    action = tf.placeholder("float", [None, 5])
    y = tf.placeholder("float", [None])
    """action * prediction(wx + b) == 예측값(q),y == replay batch에서 나온값"""
    prediction = tf.reduce_sum(tf.multiply(readout, action), reduction_indices=1)
    return prediction, y

# (정답 - 예측)^2 으로 손실값을 줄임.
#     prediction : 상태에 따른 행동 값
# 손실값이 줄어든 tensor 를 되돌려줌



def train_network_by_batch(minibatch, readout, train_step, s, a, y):
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
        y : y_batch, # q learning으로 계산한 값"""
        a : a_batch, # batch에서 가져온 행동값"""
        s : s_j_batch} #batch에서 가져온 상태값"""
    )
    
def save_and_load_network(sess):

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("save_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    return saver
