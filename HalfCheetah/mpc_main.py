import numpy as np
import matplotlib.pyplot as plt
import os
from cheetah_mpc import MPC,new_cheetah
from mpc_nn import PolicyNetwork
import torch
#note:Nt keeps around 50
# Global variables

h=0.2 # time step
nx=18 # number of state ,include rootx 
nu=6 # number of control
Tfinal = 5 # final time 
Nt =int((Tfinal/h)+1) # number of time steps
M = 20 # number of perturbed trajectories  
env = new_cheetah(render_mode=None, exclude_current_positions_from_observation=False)
#get initial states
initial_state,_=env.reset(seed=0)
#6,51
policy_net=PolicyNetwork(nx,nu)
desired_states=np.array([10, -0.01222431, 0.07171958, 0.03947361, -0.08116453, 2, 0.05222794, 0.05721286, -0.07437727, 2.5, 0.0879398, 0.07777919, 0.00660307, 0.11272412, 0.04675093, -0.08592925, 0.03687508, -0.09588826])
mpc = MPC(policy_network=policy_net, horizon=Nt, learning_rate=1e-2,desired_states=desired_states,M=M,nx=nx,nu=nu)

mpc.optimize_policy(initial_state=initial_state, max_iterations=2000, convergence_threshold=1e-6)

#testing model,uncomment to sim

# model = PolicyNetwork(state_dim=nx,action_dim=nu)
# model.load_state_dict(torch.load('policy100.pth'))
# model.eval()
# input_state = initial_state
# with torch.no_grad(): 
#     actions = model(input_state)
    #simulation(actions=actions)