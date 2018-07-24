import random
import numpy as np

def act_with_greedy_policy(epsilon, readout_t, action):

    if random.random() < epsilon :
        print("----------Random Action----------")
        action_index = random.randrange(2)
        action[random.randrange(2)] = 1
    else:
        action_index = np.argmax(readout_t)
        action[action_index] = 1