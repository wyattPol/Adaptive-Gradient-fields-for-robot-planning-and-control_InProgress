import numpy as np
import matplotlib.pyplot as plt
import os
from cheetah_mpc import MPC,simulation,animation,new_cheetah
from mpc_nn import PolicyNetwork
import torch


os.environ["WANDB_MODE"] = "disabled"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h=0.2 # time step
nx=18 # number of state ,include rootx 
nu=6 # number of control
Tfinal = 5 # final time 
Nt =int((Tfinal/h)+1) # number of time steps
M = 20 # number of perturbed trajectories  
env = new_cheetah(render_mode="human", exclude_current_positions_from_observation=False)
#get initial states
state,_=env.reset(seed=42)
#6,51
model = PolicyNetwork(nx, nu).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

total_reward = 0

for step in range(1000):  # run for 100 steps
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action = model(state_tensor)
    
    action_np = action.cpu().numpy()
    next_state, reward, done, _, _ = env.step(action_np)
    
    total_reward += reward
    # print(f"Step {step}: Action = {action_np}, Reward = {reward}")
    
    state = next_state
    
    if done:
        break

print(f"Simulation finished. Total reward: {total_reward}")

env.close()

# model.load_state_dict(torch.load('model_200.pth'))
# model.eval()
# input_state = initial_state
# with torch.no_grad(): 
#     actions = model(input_state)
#     simulation(actions=actions)
    #animation(actions=actions)
    # pip install mujoco==2.3.7 for cheetah,
    #for allgero mujoco .3.2.2