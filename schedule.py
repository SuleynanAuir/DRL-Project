# Credit to Stanford CS 234 Winter 2021 team for part of this code:
# Guillaume Genthial and Shuhui Qu, Haojun Li and Garrett Thomas
import numpy as np
from utils.test_env import EnvTest

class LinearSchedule(object):
    def __init__(self, val_begin, val_end, nsteps):
        """
        Args:
            val_begin: initial value
            val_end: end value
            nsteps: number of steps between the two values
        """
        self.curr_val = val_begin
        self.val_begin = val_begin
        self.val_end = val_end
        self.nsteps = nsteps

    def update(self, t):
        """
        Updates self.curr_val

        Args:
            t: int
                frame number
        """
        ##############################################################
        """
        TODO: modify self.curr_val such that
			  it is a linear interpolation from self.val_begin to
			  self.val_end as t goes from 0 to self.nsteps
			  For t > self.nsteps self.curr_val remains constant
        """
        ##############################################################
        ################ YOUR CODE HERE - 3-4 lines ##################
        if self.nsteps <= 0:
            self.curr_val = self.val_end
            return
        alpha = min(float(t) / float(self.nsteps), 1.0)
        self.curr_val = self.val_begin + alpha * (self.val_end - self.val_begin)

        ##############################################################
        ######################## END YOUR CODE ############## ########


class ExplorationSchedule(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(ExplorationSchedule, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, q_vals):
        """
        Returns a random action with prob curr_val, otherwise returns
        the best_action

        Args:
            q_vals: list or numpy array
                Q values for all actions

        Returns:
            an action
        """
        ##############################################################
        """
        TODO: with probability self.curr_val, return a random action
                else, return the best action (i.e. the argmax of Q 
                values)

                you can access the environment via self.env

                you may use env.action_space.sample() to generate
                a random action
        """
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines ##################
        if np.random.rand() < self.curr_val:
            return self.env.action_space.sample()
        return int(np.argmax(q_vals))

        ##############################################################
        ######################## END YOUR CODE #######################


def test1():
    env = EnvTest((5, 5, 1))
    exp_strat = ExplorationSchedule(env, 1, 0, 10)
    num_actions = env.action_space.n

    found_diff = False
    for i in range(10):
        rnd_act = exp_strat.get_action(np.zeros((num_actions,)))
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_strat = ExplorationSchedule(env, 1, 0, 10)
    exp_strat.update(5)
    assert exp_strat.curr_val == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_strat = ExplorationSchedule(env, 1, 0.5, 10)
    exp_strat.update(20)
    assert exp_strat.curr_val == 0.5, "Test 3 failed"
    print("Test3: ok")


def your_test():
    """
    Use this to implement your own tests if you'd like (not required)
    """
    pass


if __name__ == "__main__":
    test1()
    test2()
    test3()
    your_test()
