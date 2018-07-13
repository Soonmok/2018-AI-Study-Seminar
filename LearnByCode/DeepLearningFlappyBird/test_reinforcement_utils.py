from reinforcement_utils import act_with_greedy_policy
import numpy as np
import random as random

def test_act_with_greed_policy():
    for i in range(10):
        epsilon = -1.0
        readout = [random.randrange(100) for _ in range(2)]
        action = [0,0]

        act_with_greedy_policy(epsilon, readout, action)

        assert(np.sum(action) == 1)
        assert action[np.argmax(readout)] == 1


    for i in range(5):
        epsilon = 1.0
        readout = [random.randrange(100) for _ in range(2)]

if __name__ == "__main__":
    test_act_with_greed_policy()
