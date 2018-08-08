import random
import numpy as np
import math
def act_with_greedy_policy(epsilon, readout_t, action):
    if random.random() < epsilon :
        print("----------Random Action----------")
        action_index = random.randrange(2)
        action[random.randrange(5)] = 1
    else:
        action_index = np.argmax(readout_t)
        action[action_index] = 1
    return  action
