# original source : https://github.com/lazyprogrammer/machine_learning_examples rl

import pandas as pd
import numpy as np


action_space = ('U','D','L','R')


class Grid:  # Enviroment
    def __init__(self,rows,cols,start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
    
    def set(self,rewards,actions):
        self.rewards = rewards
        self.actions = actions
        
    def set_state(self,s):
        self.i = s[0]
        self.j = s[1]
        
    def current_state(self):
        return (self.i,self.j)
    
    def is_terminal(self,s):
        return s not in self.actions
        
    def get_next_state(self,s,a):
        # Where would I end up if I am in state s and take action a  
        # this is not actually moving
        i,j = s[0],s[1]
        
        if a in self.actions[(i,j)]:
            if a =='U':
                i -= 1
            if a == 'D':
                i += 1
            if a == 'L':
                j -= 1
            if a == 'R':
                j += 1
        return i,j
    
    def move(self, action):
        # chek legal move first
        if action in self.actions[(self.current_state())]:
            if action =='U':
                self.i -= 1
            if action == 'D':
                self.i += 1
            if action == 'L':
                self.j -= 1
            if action == 'R':
                self.j += 1
            #Here we return the reward as the state has already been changed
        else: print('unavailable move')
        return self.rewards.get((self.i,self.j),0)
    
    def undo_move(self,action):
        if action in self.actions:
            if action =='U':
                self.i += 1
            if action == 'D':
                self.i -= 1
            if action == 'L':
                self.j += 1
            if action == 'R':
                self.j -= 1        
        assert(self.current_state() in self.all_states())
    
    def game_over(self):
        # Checking with the actions set as the reward can be extended to all states, 
        #as giving a negative reward to all staates to look for the shortest path
        return self.current_state() not in self.actions
    
    def all_states(self):
        # grouping the states defined in agtions and in rewards
        return set(self.actions.keys()) | set(self.rewards.keys())
     

def standard_grid():
    # Defu=ining a grid whith start in s, a wall in x and rewards 1,-1 in the presented positions
    #...1
    #.x.-1
    #s...
    grid = Grid(3,4,(2,0))
    reward = {(0,3):1,(1,3):-1}
    action = {
        (0,0):('D','R'),
        (0,1):('L','R'),
        (0,2):('L','R','D'),
        (1,0):('D','U'),
        (1,2):('D','U','R'),
        (2,0):('U','R'),
        (2,1):('L','R'),
        (2,2):('U','R','L'),
        (2,3):('L','U')
             }
    grid.set(reward,action)
    return grid


class WindyGrid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]
  
    def set(self, rewards, actions, probs):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions
        self.probs = probs     # Now we must define the transition probabilities for a given state and action
  
    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]
  
    def current_state(self):
        return (self.i, self.j)
  
    def is_terminal(self, s):
        return s not in self.actions
  
    def move(self, action):
        s = (self.i, self.j)
        a = action
        # now the move result is probabilistic, we must perform a random sort among the target states
        next_state_probs = self.probs[(s, a)]
        print(s)
        print(a)
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())

        s2 = next_states[np.random.choice(len(next_probs),p=next_probs)] #np.random.choice(next_states, p=next_probs)
    
        # update the current state
        self.i, self.j = s2
    
        # return a reward (if any)
        return self.rewards.get(s2, 0)


    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions
  
    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())    
      


def windy_grid():
    g = WindyGrid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
      (0, 0): ('D', 'R'),
      (0, 1): ('L', 'R'),
      (0, 2): ('L', 'R', 'D'),
      (1, 0): ('D', 'U'),
      (1, 2): ('D', 'U', 'R'),
      (2, 0): ('U', 'R'),
      (2, 1): ('L', 'R'),
      (2, 2): ('U', 'R', 'L'),
      (2, 3): ('L', 'U')
    }

    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
    probs = {
      ((2, 0), 'U'): {(1, 0): 1.0},
      ((2, 0), 'D'): {(2, 0): 1.0},
      ((2, 0), 'L'): {(2, 0): 1.0},
      ((2, 0), 'R'): {(2, 1): 1.0},
      ((1, 0), 'U'): {(0, 0): 1.0},
      ((1, 0), 'D'): {(2, 0): 1.0},
      ((1, 0), 'L'): {(1, 0): 1.0},
      ((1, 0), 'R'): {(1, 0): 1.0},
      ((0, 0), 'U'): {(0, 0): 1.0},
      ((0, 0), 'D'): {(1, 0): 1.0},
      ((0, 0), 'L'): {(0, 0): 1.0},
      ((0, 0), 'R'): {(0, 1): 1.0},
      ((0, 1), 'U'): {(0, 1): 1.0},
      ((0, 1), 'D'): {(0, 1): 1.0},
      ((0, 1), 'L'): {(0, 0): 1.0},
      ((0, 1), 'R'): {(0, 2): 1.0},
      ((0, 2), 'U'): {(0, 2): 1.0},
      ((0, 2), 'D'): {(1, 2): 1.0},
      ((0, 2), 'L'): {(0, 1): 1.0},
      ((0, 2), 'R'): {(0, 3): 1.0},
      ((2, 1), 'U'): {(2, 1): 1.0},
      ((2, 1), 'D'): {(2, 1): 1.0},
      ((2, 1), 'L'): {(2, 0): 1.0},
      ((2, 1), 'R'): {(2, 2): 1.0},
      ((2, 2), 'U'): {(1, 2): 1.0},
      ((2, 2), 'D'): {(2, 2): 1.0},
      ((2, 2), 'L'): {(2, 1): 1.0},
      ((2, 2), 'R'): {(2, 3): 1.0},
      ((2, 3), 'U'): {(1, 3): 1.0},
      ((2, 3), 'D'): {(2, 3): 1.0},
      ((2, 3), 'L'): {(2, 2): 1.0},
      ((2, 3), 'R'): {(2, 3): 1.0},
      ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
      ((1, 2), 'D'): {(2, 2): 1.0},
      ((1, 2), 'L'): {(1, 2): 1.0},
      ((1, 2), 'R'): {(1, 3): 1.0}
    }
    g.set(rewards, actions, probs)
    return g
    
    
def windy_grid_penalized(step_cost=-0.1):
  g = WindyGrid(3, 4, (2, 0))
  rewards = {
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (0, 3): 1,
    (1, 3): -1
  }
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }

  # p(s' | s, a) represented as:
  # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
  probs = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
  g.set(rewards, actions, probs)
  return g
