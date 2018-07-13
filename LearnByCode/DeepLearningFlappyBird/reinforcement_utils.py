import random
import numpy as np

def act_with_greedy_policy(epsilon, readout_t, action):

    """ epsilon 확률로 랜덤하게 행동"""
    if random.random() <= epsilon:
        print("----------Random Action----------")
        action_index = random.randrange(2)
        action[random.randrange(2)] = 1

        """ 아니면 예측된 값으로 행동"""
    else:
        """ 
            action_index == 0이면 가만히 있기
            action_index == 1이면 점프뛰기
        """
        action_index = np.argmax(readout_t)
        action[action_index] = 1