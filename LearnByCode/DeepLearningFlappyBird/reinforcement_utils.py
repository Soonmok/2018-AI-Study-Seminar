import random
import numpy as np

def act_with_greedy_policy(epsilon, readout_t, action):
    """
        DQN은 다양하고 최적화가 된 학습을 위해 항상 가장 큰 값을 선택하기 보다는 
        epsilon 확률 변수를 최초에 설정해 가보지 못한 선택지에 대해서도 탐험을 한다.
        
        따라서 랜덤 값이 epsilon 확률에 속한다면 을 주어야 한다.
        그렇지 않다면 readout_t 에서 가장 높은 값을 가지는 행동을 해야한다. 
        
        매개변수에 대한 정보:
            1. epsilon : deep_q_network.py 에 설정이 되어있는 확률 변수 이다.
            2. readout_t : 행동(Jump or Stay)을 했을 때 얻을 수 있다고 예상되는 점수 값을 가지고 있다.
            3. action : 
                [1X2] 의 크기를 가지고 있는 변수이며, 최종 행동을 선택을 할 때 1인 값을 선택하게 된다.
                    [0] : stay, [1] : Jump
                를 가르키고 있으며, 둘 중 하나는 항상 1(True)을 가지고 있어야 한다.

<<<<<<< HEAD
        함수에 대한 정보
        1. random.random() -> 0~1 사이의 실수 값을 랜덤하게 되돌려 준다.
            random.randrange(MAX) -> 0 ~ MAX-1 사이의 정수 값을 랜덤하게 되돌려 준다.
        2. np.argmax(list) -> list에서 가장 높은 값을 가지는 index를 되돌려 준다.
    """
    if random.random() < epsilon :
||||||| merged common ancestors
    """ epsilon 확률로 랜덤하게 행동"""
    if random.random() <= epsilon:
=======
        함수에 대한 정보
        1. random.random() -> 0~1 사이의 실수 값을 랜덤하게 되돌려 준다.
            random.random(MAX) -> 0 ~ MAX-1 사이의 정수 값을 랜덤하게 되돌려 준다.
        2. np.argmax(list) -> list에서 가장 높은 값을 가지는 index를 되돌려 준다.
    """
    if random.random() < epsilon :
>>>>>>> upstream/master
        print("----------Random Action----------")
        action_index = random.randrange(2)
        action[random.randrange(2)] = 1
    else:
        action_index = np.argmax(readout_t)
        action[action_index] = 1
