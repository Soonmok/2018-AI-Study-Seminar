import random
import numpy as np

def act_with_greedy_policy(epsilon, readout_t, action):

    if random.random() < epsilon :
        print("----------Random Action----------")
        action = [random.random(0, 300), random.random(0, 300)]
    else:
        action = readout_t