from reinforcement_utils import act_with_greedy_policy
import numpy as np
import random as random

def test_answer(epsilon, readout, action):
    act_with_greedy_policy(epsilon, readout, action)
    BefoEpsilon = epsilon

    assert(np.sum(action) == 1)
    assert(BefoEpsilon == epsilon)

    return readout, action

if __name__ == "__main__":
    for i in range(10):
        epsilon = -1.0
        readout = [random.randrange(100) for _ in range(2)]
        action = [0,0]
        readout, action = test_answer(epsilon, readout, action)
        
        assert action[np.argmax(readout)] == 1
        print("Test %d Succeed" % i)


    for i in range(5):
        assert (test_answer(1, [random.randrange(100) for _ in range(2)], [0,0]))
        print("Test %d Succeed" % (i+10))
