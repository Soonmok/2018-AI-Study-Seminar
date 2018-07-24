import tensorflow as tf


# 상태에 따른 행동 값을 압축? 시켜서 전달
#    readout : 상태에 따른 행동 값
# prediction : 예측값
# y : 정답 값 (실 정답은 아니고, 가지고 있던 값)
def get_prediction(readout):
    action = tf.placeholder("float", [None, 2])
    y = tf.placeholder("float", [None])

    """action * prediction(wx + b) == 예측값(q),y == replay batch에서 나온값"""
    prediction = tf.reduce_sum(tf.multiply(readout, action), reduction_indices=1)
    return prediction, y

# (정답 - 예측)^2 으로 손실값을 줄임.
#     prediction : 상태에 따른 행동 값
# 손실값이 줄어든 tensor 를 되돌려줌
def get_cost(prediction):
    prediction_action, y = get_prediction(prediction)
    """ 예측값과 batch에서 나온 값들의 오류로 학습"""
    return tf.reduce_mean(tf.square(y - prediction_action))


def save_and_lod_network(sess):
    pass


# save_and_lod_network? train_netowrk_by_batch