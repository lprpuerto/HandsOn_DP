# Original source: https://github.com/lazyprogrammer/machine_learning_examples rl
import numpy as np

from grid_world import action_space, standard_grid

SMALL_ENOUGH = 1e-3 # convergence threshold


# Functions to print the values and policy in each cell of the grid

def print_values(V,g):
    for i in range(g.rows):
        print("--------------------------")
        for j in range(g.cols):
            v=V.get((i,j),0)
            if v>=0:
                print(" %.2f|"%v,end=" ")
            else:
                print("%.2f|"%v,end=" ")# negatives tke an extra space
        print("")
        
def print_policy(P, g):
    for i in range(g.rows):
        print("------------------------")
        for j in range(g.cols):
            a = P.get((i,j), ' ')
            print("  %s  |" % a, end="")
        print("")
        

        
if __name__ == '__main__':
    transition_probs = {}
    rewards = {}
    
    grid = standard_grid()
    for i in range(grid.rows):
        for j in range (grid.cols):
            s = (i,j)
            # All the non terminal states
            if not grid.is_terminal(s):
                for a in action_space:
                    # the possible next states
                    s2 = grid.get_next_state(s,a)
                    if s2 != s:
                        transition_probs[(s,a,s2)] = 1
                    if s2 in grid.rewards:
                        #Transition rewards
                        rewards[(s,a,s2)] = grid.rewards[s2]
                        
    ### fixed policy ###
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L'}
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
                        action_prob = 1 if policy[s] == a else 0

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
    print("\n\n")  
