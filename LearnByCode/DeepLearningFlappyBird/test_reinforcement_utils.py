from reinforcement_utils import act_with_greedy_policy
import numpy as np
import random as random


if __name__ == "__main__":
    for i in range(10):
        epsilon = -1.0
        readout = [random.randrange(100) for _ in range(2)]
        action = [0,0]
        act_with_greedy_policy(epsilon, readout, action)

        assert(np.sum(action) == 1)
        assert(action[np.argmax(readout)] == 1)

        print("Test %d Succeed" % i)


    for i in range(5):
        action = [0,0]
        epsilon = 1.
        readout = [random.randrange(100) for _ in range(2)]
        act_with_greedy_policy(epsilon, readout, action)
        
        assert(np.sum(action) == 1)

        print("Test %d Succeed" % (i+10))
