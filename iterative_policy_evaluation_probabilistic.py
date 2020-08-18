# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import windy_grid, action_space

SMALL_ENOUGH = 1e-3


# Functions to print the values and policy in each cell of the grid

def print_values(V,g):
    for i in range(g.rows):
        print("--------------------------")
        for j in range(g.cols):
            v=V.get((i,j),0)
            if v>=0:
                print(" %.3f|"%v,end=" ")
            else:
                print("%.3f|"%v,end=" ")# negatives tke an extra space
        print("")
        
def print_policy(P, g):
    for i in range(g.rows):
        print("------------------------")
        for j in range(g.cols):
            a = P.get((i,j), ' ')
            print("      %s       |" % a, end="")
        print("")



if __name__ == '__main__':
    ### define transition probabilities and grid ###
    # the key is (s, a, s'), the value is the probability
    # that is, transition_probs[(s, a, s')] = p(s' | s, a)
    # any key NOT present will considered to be impossible (i.e. probability 0)
    transition_probs = {}
  
    # to reduce the dimensionality of the dictionary, we'll use deterministic
    # rewards, r(s, a, s')
    # note: you could make it simpler by using r(s') since the reward doesn't
    # actually depend on (s, a)
    rewards = {}    
    
    
    grid = windy_grid()
    # Different to the previous ne as probs are new
    for (s,a),t_p in grid.probs.items():
        for s2,p in t_p.items():
            transition_probs[(s,a,s2)] = p
            rewards[(s,a,s2)] = grid.rewards.get(s2,0)
            
            
    ### Policy###
    policy = {
        (2, 0): {'U':1.0, 'R':0.0},
        (1, 0): {'U':1.0},
        (0, 0): {'R':1.0},
        (0, 1): {'R':1.0},
        (0, 2): {'R':1.0},
        (1, 2): {'U':1.0},
        (2, 1): {'R':1.0},
        (2, 2): {'U':1.0},
        (2, 3): {'L':1.0}}
    print_policy(policy, grid)
    
    # initialize V(s) = 0

    V = {}

    for s in grid.all_states():
        V[s] = 0

    # discount factor
    gamma = 0.9

    # Now iterating
    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in action_space:
                    for s2 in grid.all_states():
                        # assign action prob 1 as the action probability is deterministic
                        action_prob = policy[s].get(a,0)

                        # now the reward
                        r = rewards.get((s,a,s2),0)
                        # Updating the value of the iteration for the action_probs different from 0
                        new_v += action_prob*transition_probs.get((s,a,s2),0)*(r+gamma*V[s2])

                #updating the new_value
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v-new_v))
            print()    
            print("State",s)
            print_values(V,grid)
        print("iter:",it, "biggest_change",biggest_change)
        print('****************************')    

        it += 1
        if biggest_change < SMALL_ENOUGH:
            break
