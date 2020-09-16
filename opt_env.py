import numpy as np
from scipy.optimize import minimize, linprog


class RORLenv():
  """
  This class defines the environment for the optimization problem with uncertainty 
  to be used for the RORL approach. 
  """
  def __init__(self, dim):

    # dim is the dimension of the driving factor space. 
    self.state = None
    self.state_val = None
    self.action_space_bounds=np.ones((1,dim), dtype=int)
    self.obs_space_n = dim
    self.action_space_n = dim
    # self.state_values = dict()

  def action(self):
    temp = 2* (np.random.rand(1, self.action_space_n) - 0.5)
    temp *= self.action_space_bounds.reshape(1, self.action_space_n)
    return temp



  def env_LP(self, s):
    """
    This function solves the optimization problem for a given state s. Here, the optimization problem is an LP. 
    We use the LP solver of scipy.
    """
    g1, g2, g3, g4, g5 , g6, g7, g8, g9, g10= s.reshape(10)
    

    # c=[-1,-1,-1,-1,-1,-1,-1]
    c=[-1+g6**2,-1+g7,-1,-1-g8,-1+g8**2,-1+g9,-1-g10]

    A=[[1,1,1,0,0,0,0], 
      [1,0,0,1,0,0,0],
      [0,0,0,1,0,1,0],
      [0,1,0,0,1,0,0],
      [0,0,0,0,1,0,0],
      [0,0,1,0,0,1,1],
      [0,0,0,0,0,0,1]]
    b=[100-g1+g2**2, 90+g3**2+g1, 200-g4**3-g3+g5, 80-g5**2, 100, 80+g1**2+g5, 90-g1+g4+g3**2]
    res = linprog(c, A_ub=A, b_ub=b, bounds = [(0,None)] * 7)

    if res.status == 2:
      return (10 ** 10, True)
      # return (10 ** 10, False)
    elif res.status == 0:
      return (res.fun, False)

  def reset(self, initial = [0,0,0,0,0,0,0,0,0,0]):
    self.state = np.array(initial).reshape(1,self.obs_space_n)
    val, flag = self.env_LP(self.state)
    self.state_val = val
    return self.state

  def step(self, action):
    """
    Given the self.state and the action, this function calculates the reward of the action and returns the next state.
    """
    pre_state = self.state[0]
    cur_state = pre_state + action
    val, flag = self.env_LP(cur_state)
    updated_value = self.state_val -val
    self.state = np.array([cur_state])
    self.state_val= val
    
    return [np.array([cur_state]),  updated_value , flag]
  
