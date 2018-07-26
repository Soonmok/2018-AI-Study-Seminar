import random
import numpy as np
import math
def act_with_greedy_policy(epsilon, readout_t, action):

    if random.random() < epsilon :
        print("----------Random Action----------")
        thetha = random.randrange(0, 90)
        radian = math.cos(math.radians(thetha))
        action = [100 * math.cos(radian),100 * math.sin(radian)]
        print("Actions Value ",end="")
        print(action[0], action[1])
        print("Thetha : ", thetha)
    else:
        action = readout_t
        #print(readout_t)
